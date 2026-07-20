#!/usr/bin/env python3

"""Evaluate SceneGen-style reconstruction metrics for predicted and GT GLBs."""

from __future__ import annotations

import argparse
import csv
from itertools import permutations
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils.metric_utils import (  # noqa: E402
    bbox_iou_3d,
    bbox_overlap_volume,
    bounds_from_mesh_or_scene,
    compute_cd_and_f_score,
    compute_IoU,
    load_mesh_or_scene,
    scene_to_single_mesh,
    similarity_transform_umeyama,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", type=Path, required=True, help="COM4D inference output directory.")
    ap.add_argument("--gt-dir", type=Path, required=True, help="Ground-truth GLB root.")
    ap.add_argument("--output-dir", type=Path, default=None, help="Defaults to <pred-dir>/eval_reconstruction.")
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--metric", default="l2")
    ap.add_argument("--iou-num-grids", type=int, default=64)
    ap.add_argument("--iou-scale", type=float, default=2.0)
    ap.add_argument(
        "--skip-raw-metrics",
        action="store_true",
        help="With alignment enabled, compute only aligned reconstruction metrics.",
    )
    ap.add_argument("--skip-voxel-iou", action="store_true", help="Skip voxelization and report voxel IoU as NaN.")
    ap.add_argument(
        "--max-voxel-vertices",
        type=int,
        default=2_000_000,
        help="Skip voxel IoU when either mesh exceeds this vertex count; 0 disables the guard.",
    )
    ap.add_argument(
        "--max-voxel-faces",
        type=int,
        default=4_000_000,
        help="Skip voxel IoU when either mesh exceeds this face count; 0 disables the guard.",
    )
    ap.add_argument(
        "--max-voxel-cells",
        type=int,
        default=16_777_216,
        help="Skip voxel IoU when either mesh's estimated bounding grid exceeds this cell count; 0 disables the guard.",
    )
    ap.add_argument(
        "--alignment",
        choices=("none", "translation", "similarity", "first_frame_similarity"),
        default="none",
        help="Optional transform from predicted coordinates to GT coordinates. first_frame_similarity matches the COM4D paper protocol.",
    )
    ap.add_argument(
        "--object-assignment",
        choices=("fixed", "best"),
        default="fixed",
        help="Match object tracks by ID or by lowest sequence-level center distance.",
    )
    ap.add_argument(
        "--com-method",
        choices=("vertex_centroid", "volume_center_mass", "bbox_center"),
        default="vertex_centroid",
        help=(
            "Predicted/GT mesh center proxy for translation/similarity alignment and best object assignment. "
            "Matches evaluate_physics.py; bbox_center is retained for old behavior."
        ),
    )
    ap.add_argument(
        "--alignment-samples-per-frame",
        type=int,
        default=1024,
        help="Number of deterministic mesh points used by first_frame_similarity alignment.",
    )
    ap.add_argument(
        "--pred-scene-glob",
        default="dynamic/dynamic_scene_frame_*.glb",
        help="Glob relative to --pred-dir for full-scene frame GLBs.",
    )
    ap.add_argument(
        "--gt-scene-glob",
        default="**/frame_*.glb",
        help="Glob relative to --gt-dir for full-scene GT frame GLBs.",
    )
    ap.add_argument(
        "--skip-object-level",
        action="store_true",
        help="Only compute full-scene metrics.",
    )
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write NaN summaries instead of failing when no GT/pred pairs are found.",
    )
    return ap.parse_args()


def frame_index(path: Path) -> int | None:
    patterns = [
        r"dynamic_scene_frame_(\d+)\.glb$",
        r"frame_(\d+)\.glb$",
        r"frame_(\d+)\.json$",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.name)
        if match is not None:
            return int(match.group(1))
    return None


def path_has_mesh_geometry(path: Path) -> bool:
    try:
        loaded = load_mesh_or_scene(path)
        if isinstance(loaded, trimesh.Trimesh):
            return len(loaded.vertices) > 0
        if isinstance(loaded, trimesh.Scene):
            return any(
                isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0
                for mesh in loaded.dump(concatenate=False)
            )
    except Exception:
        return False
    return False


def dynamic_object_paths(root: Path) -> dict[str, dict[int, Path]]:
    dynamic_dir = root / "dynamic"
    tracks: dict[str, dict[int, Path]] = {}
    if not dynamic_dir.exists():
        return tracks
    for obj_dir in sorted(dynamic_dir.glob("object_*")):
        if not obj_dir.is_dir():
            continue
        frame_map = {}
        for path in sorted(obj_dir.glob("frame_*.glb")):
            idx = frame_index(path)
            if idx is not None and path_has_mesh_geometry(path):
                frame_map[idx] = path
        if frame_map:
            tracks[obj_dir.name] = frame_map
    return tracks


def dynamic_metadata_names(root: Path) -> list[str]:
    metadata_path = root / "physics_metadata.json"
    if not metadata_path.is_file():
        return []
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    objects = metadata.get("objects")
    if isinstance(objects, dict) and objects:
        names = [
            name
            for name, spec in objects.items()
            if isinstance(spec, dict) and bool(spec.get("dynamic", name.startswith("ball_")))
        ]
        render_objects = metadata.get("render_objects")
        if isinstance(render_objects, list):
            order = {name: idx for idx, name in enumerate(render_objects)}
            return sorted(names, key=lambda name: order.get(name, len(order)))
        return sorted(names)
    render_objects = metadata.get("render_objects")
    if isinstance(render_objects, list):
        return [name for name in render_objects if isinstance(name, str) and name.startswith("ball_")]
    return []


def metadata_gt_tracks(root: Path) -> dict[str, dict[int, tuple[Path, Path, str]]]:
    """Return GT object tracks from raw renderer outputs.

    Raw synthetic GT stores canonical meshes as meshes/<object>.glb and per-frame
    poses as transforms/frame_*.json, not necessarily one GLB per frame.
    """
    mesh_dir = root / "meshes"
    transform_dir = root / "transforms"
    object_names = dynamic_metadata_names(root)
    if not object_names:
        object_names = [path.stem for path in sorted(mesh_dir.glob("ball_*.glb"))]
    object_map = {
        f"object_{idx:03d}": (name, mesh_dir / f"{name}.glb")
        for idx, name in enumerate(object_names)
    }
    if not transform_dir.is_dir() or not object_map or not all(path.is_file() for _, path in object_map.values()):
        return {}

    tracks: dict[str, dict[int, tuple[Path, Path, str]]] = {object_id: {} for object_id in object_map}
    for transform_path in sorted(transform_dir.glob("frame_*.json")):
        idx = frame_index(transform_path)
        if idx is None:
            continue
        for object_id, (object_key, mesh_path) in object_map.items():
            tracks[object_id][idx] = (mesh_path, transform_path, object_key)
    return {object_id: frames for object_id, frames in tracks.items() if frames}


def metadata_gt_position_tracks(root: Path) -> dict[str, dict[int, np.ndarray]]:
    metadata_path = root / "physics_metadata.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    object_names = dynamic_metadata_names(root)
    if not object_names:
        return {}
    tracks: dict[str, dict[int, np.ndarray]] = {f"object_{idx:03d}": {} for idx, _name in enumerate(object_names)}
    frames = metadata.get("frames")
    if not isinstance(frames, list):
        return {}
    for frame_data in frames:
        if not isinstance(frame_data, dict):
            continue
        frame = frame_data.get("frame")
        if not isinstance(frame, int):
            continue
        objects = frame_data.get("objects") if isinstance(frame_data.get("objects"), dict) else frame_data
        for idx, object_name in enumerate(object_names):
            item = objects.get(object_name) if isinstance(objects, dict) else None
            if not isinstance(item, dict) or "position" not in item:
                continue
            position = np.asarray(item["position"], dtype=np.float64)
            if position.shape == (3,) and np.isfinite(position).all():
                tracks[f"object_{idx:03d}"][frame] = position
    return {object_id: frames for object_id, frames in tracks.items() if frames}


def frame_paths(root: Path, pattern: str) -> dict[int, Path]:
    paths = {}
    for path in sorted(root.glob(pattern)):
        if any(parent.name.startswith("object_") for parent in path.parents):
            continue
        idx = frame_index(path)
        if idx is not None and idx not in paths and path_has_mesh_geometry(path):
            paths[idx] = path
    return paths


def metadata_gt_scene_frames(root: Path) -> dict[int, list[tuple[Path, Path, str]]]:
    tracks = metadata_gt_tracks(root)
    frames: dict[int, list[tuple[Path, Path, str]]] = {}
    for frame_map in tracks.values():
        for frame, spec in frame_map.items():
            frames.setdefault(frame, []).append(spec)
    return {frame: specs for frame, specs in frames.items() if specs}


def transform_matrix_from_json(transform_path: Path, ball_key: str) -> np.ndarray:
    with transform_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    objects = data.get("objects")
    item = objects.get(ball_key) if isinstance(objects, dict) else data.get(ball_key)
    if item is None:
        raise KeyError(f"Missing transform for {ball_key!r} in {transform_path}")
    matrix = trimesh.transformations.quaternion_matrix(item["quaternion_blender_wxyz"])
    matrix[:3, 3] = np.asarray(item["location"], dtype=np.float64)
    return matrix


def load_transformed_mesh(mesh_path: Path, transform_path: Path, ball_key: str) -> trimesh.Trimesh:
    loaded = load_mesh_or_scene(mesh_path)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        meshes = [item.copy() for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    gltf_to_blender = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    mesh.apply_transform(gltf_to_blender)
    mesh.apply_transform(transform_matrix_from_json(transform_path, ball_key))
    return mesh


def load_gt_scene(specs: list[tuple[Path, Path, str]]) -> trimesh.Trimesh:
    meshes = [load_transformed_mesh(mesh_path, transform_path, ball_key) for mesh_path, transform_path, ball_key in specs]
    return trimesh.util.concatenate(meshes)


def mesh_center(mesh: trimesh.Trimesh, method: str = "vertex_centroid") -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    bounds_center = bounds.mean(axis=0)
    if method == "bbox_center":
        return bounds_center
    try:
        if method == "volume_center_mass":
            center = np.asarray(mesh.center_mass, dtype=np.float64)
        else:
            center = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
        if center.shape == (3,) and np.isfinite(center).all():
            return center
    except Exception:
        pass
    return bounds_center


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def deterministic_mesh_points(mesh: trimesh.Trimesh, max_points: int) -> np.ndarray:
    """Return deterministic mesh point samples for alignment."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return vertices.reshape(0, 3)
    if len(vertices) <= max_points:
        return vertices
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    indices = np.linspace(0, len(order) - 1, max_points, dtype=np.int64)
    return vertices[order[indices]]


def mesh_points_from_path(path: Path, max_points: int) -> np.ndarray:
    mesh = scene_to_single_mesh(load_mesh_or_scene(path))
    return deterministic_mesh_points(mesh, max_points)


def mesh_points_from_gt_spec(spec: Path | tuple[Path, Path, str], max_points: int) -> np.ndarray:
    return deterministic_mesh_points(gt_track_mesh(spec), max_points)


def first_common_frame(left: set[int], right: set[int]) -> int | None:
    common = sorted(left.intersection(right))
    return common[0] if common else None


def collect_first_frame_point_clouds(
    pred_scenes: dict[int, Path],
    gt_scenes: dict[int, Path],
    gt_metadata_scenes: dict[int, list[tuple[Path, Path, str]]],
    pred_tracks: dict[str, dict[int, Path]],
    gt_tracks: dict[str, dict[int, Path | tuple[Path, Path, str]]],
    assignment: dict[str, str],
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = first_common_frame(set(pred_scenes), set(gt_scenes))
    if frame is not None:
        return mesh_points_from_path(pred_scenes[frame], max_points), mesh_points_from_path(gt_scenes[frame], max_points)

    frame = first_common_frame(set(pred_scenes), set(gt_metadata_scenes))
    if frame is not None:
        gt_mesh = load_gt_scene(gt_metadata_scenes[frame])
        return mesh_points_from_path(pred_scenes[frame], max_points), deterministic_mesh_points(gt_mesh, max_points)

    pred_points = []
    gt_points = []
    per_object_points = max(16, max_points // max(1, len(assignment)))
    for pred_id, gt_id in sorted(assignment.items()):
        pred_frames = pred_tracks.get(pred_id, {})
        gt_frames = gt_tracks.get(gt_id, {})
        frame = first_common_frame(set(pred_frames), set(gt_frames))
        if frame is None:
            continue
        pred_points.append(mesh_points_from_path(pred_frames[frame], per_object_points))
        gt_points.append(mesh_points_from_gt_spec(gt_frames[frame], per_object_points))
    if not pred_points or not gt_points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.concatenate(pred_points, axis=0), np.concatenate(gt_points, axis=0)


def first_frame_similarity_transform(pred_points: np.ndarray, gt_points: np.ndarray) -> np.ndarray:
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.eye(4, dtype=np.float64)
    try:
        matrix, _transformed, _cost = trimesh.registration.icp(
            pred_points,
            gt_points,
            threshold=1e-6,
            max_iterations=50,
            scale=True,
        )
        return np.asarray(matrix, dtype=np.float64)
    except Exception:
        count = min(len(pred_points), len(gt_points))
        if count == 0:
            return np.eye(4, dtype=np.float64)
        pred_idx = np.linspace(0, len(pred_points) - 1, count, dtype=np.int64)
        gt_idx = np.linspace(0, len(gt_points) - 1, count, dtype=np.int64)
        return similarity_transform_umeyama(pred_points[pred_idx], gt_points[gt_idx])


def translation_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    if len(src) and len(dst):
        transform[:3, 3] = dst.mean(axis=0) - src.mean(axis=0)
    return transform


def gt_track_mesh(spec: Path | tuple[Path, Path, str]) -> trimesh.Trimesh:
    if isinstance(spec, tuple):
        mesh_path, transform_path, ball_key = spec
        return load_transformed_mesh(mesh_path, transform_path, ball_key)
    return scene_to_single_mesh(load_mesh_or_scene(spec)).copy()


def gt_track_label(spec: Path | tuple[Path, Path, str]) -> str:
    if isinstance(spec, tuple):
        mesh_path, transform_path, ball_key = spec
        return f"{mesh_path}@{transform_path}:{ball_key}"
    return str(spec)


def collect_object_center_pairs(
    pred_tracks: dict[str, dict[int, Path]],
    gt_tracks: dict[str, dict[int, Path | tuple[Path, Path, str]]],
    assignment: dict[str, str] | None = None,
    center_method: str = "vertex_centroid",
) -> tuple[np.ndarray, np.ndarray]:
    pred_centers = []
    gt_centers = []
    if assignment is None:
        assignment = {object_id: object_id for object_id in sorted(set(pred_tracks).intersection(gt_tracks))}
    for pred_id, gt_id in sorted(assignment.items()):
        pred_frames = pred_tracks.get(pred_id, {})
        gt_frames = gt_tracks.get(gt_id, {})
        for frame in sorted(set(pred_frames).intersection(gt_frames)):
            pred_mesh = scene_to_single_mesh(load_mesh_or_scene(pred_frames[frame]))
            gt_mesh = gt_track_mesh(gt_frames[frame])
            pred_centers.append(mesh_center(pred_mesh, center_method))
            gt_centers.append(mesh_center(gt_mesh, center_method))
    return np.asarray(pred_centers, dtype=np.float64), np.asarray(gt_centers, dtype=np.float64)


def collect_object_center_pairs_from_positions(
    pred_tracks: dict[str, dict[int, Path]],
    gt_position_tracks: dict[str, dict[int, np.ndarray]],
    assignment: dict[str, str] | None = None,
    center_method: str = "vertex_centroid",
) -> tuple[np.ndarray, np.ndarray]:
    pred_centers = []
    gt_centers = []
    if assignment is None:
        assignment = {object_id: object_id for object_id in sorted(set(pred_tracks).intersection(gt_position_tracks))}
    for pred_id, gt_id in sorted(assignment.items()):
        pred_frames = pred_tracks.get(pred_id, {})
        gt_frames = gt_position_tracks.get(gt_id, {})
        for frame in sorted(set(pred_frames).intersection(gt_frames)):
            pred_mesh = scene_to_single_mesh(load_mesh_or_scene(pred_frames[frame]))
            pred_centers.append(mesh_center(pred_mesh, center_method))
            gt_centers.append(gt_frames[frame])
    return np.asarray(pred_centers, dtype=np.float64), np.asarray(gt_centers, dtype=np.float64)


def collect_scene_center_pairs(
    pred_scenes: dict[int, Path],
    gt_scenes: dict[int, Path],
    gt_metadata_scenes: dict[int, list[tuple[Path, Path, str]]],
    center_method: str = "vertex_centroid",
) -> tuple[np.ndarray, np.ndarray]:
    pred_centers = []
    gt_centers = []
    for frame in sorted(set(pred_scenes).intersection(gt_scenes)):
        pred_mesh = scene_to_single_mesh(load_mesh_or_scene(pred_scenes[frame]))
        gt_mesh = scene_to_single_mesh(load_mesh_or_scene(gt_scenes[frame]))
        pred_centers.append(mesh_center(pred_mesh, center_method))
        gt_centers.append(mesh_center(gt_mesh, center_method))
    for frame in sorted(set(pred_scenes).intersection(gt_metadata_scenes)):
        pred_mesh = scene_to_single_mesh(load_mesh_or_scene(pred_scenes[frame]))
        gt_mesh = load_gt_scene(gt_metadata_scenes[frame])
        pred_centers.append(mesh_center(pred_mesh, center_method))
        gt_centers.append(mesh_center(gt_mesh, center_method))
    return np.asarray(pred_centers, dtype=np.float64), np.asarray(gt_centers, dtype=np.float64)


def alignment_transform(method: str, pred_points: np.ndarray, gt_points: np.ndarray) -> np.ndarray:
    if method == "none":
        return np.eye(4, dtype=np.float64)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.eye(4, dtype=np.float64)
    if method == "translation":
        return translation_transform(pred_points, gt_points)
    if method == "similarity":
        return similarity_transform_umeyama(pred_points, gt_points)
    if method == "first_frame_similarity":
        return first_frame_similarity_transform(pred_points, gt_points)
    raise ValueError(f"Unsupported alignment method: {method}")


def best_object_assignment(
    pred_tracks: dict[str, dict[int, Path]],
    gt_tracks: dict[str, dict[int, Path | tuple[Path, Path, str]]],
    pred_to_gt_transform: np.ndarray,
    center_method: str = "vertex_centroid",
) -> dict[str, str]:
    pred_ids = sorted(pred_tracks)
    gt_ids = sorted(gt_tracks)
    if not pred_ids or len(gt_ids) < len(pred_ids):
        return {object_id: object_id for object_id in sorted(set(pred_tracks).intersection(gt_tracks))}

    best_perm = None
    best_cost = float("inf")
    for perm in permutations(gt_ids, len(pred_ids)):
        cost = 0.0
        count = 0
        for pred_id, gt_id in zip(pred_ids, perm):
            pred_frames = pred_tracks[pred_id]
            gt_frames = gt_tracks[gt_id]
            for frame in sorted(set(pred_frames).intersection(gt_frames)):
                pred_mesh = scene_to_single_mesh(load_mesh_or_scene(pred_frames[frame]))
                gt_mesh = gt_track_mesh(gt_frames[frame])
                pred_center = transform_points(mesh_center(pred_mesh, center_method)[None, :], pred_to_gt_transform)[0]
                cost += float(np.linalg.norm(pred_center - mesh_center(gt_mesh, center_method)))
                count += 1
        if count == 0:
            continue
        cost /= count
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    if best_perm is None:
        return {object_id: object_id for object_id in sorted(set(pred_tracks).intersection(gt_tracks))}
    return dict(zip(pred_ids, best_perm))


def best_object_assignment_from_positions(
    pred_tracks: dict[str, dict[int, Path]],
    gt_position_tracks: dict[str, dict[int, np.ndarray]],
    pred_to_gt_transform: np.ndarray,
    center_method: str = "vertex_centroid",
) -> dict[str, str]:
    pred_ids = sorted(pred_tracks)
    gt_ids = sorted(gt_position_tracks)
    if not pred_ids or len(gt_ids) < len(pred_ids):
        return {object_id: object_id for object_id in sorted(set(pred_tracks).intersection(gt_position_tracks))}

    best_perm = None
    best_cost = float("inf")
    for perm in permutations(gt_ids, len(pred_ids)):
        cost = 0.0
        count = 0
        for pred_id, gt_id in zip(pred_ids, perm):
            pred_frames = pred_tracks[pred_id]
            gt_frames = gt_position_tracks[gt_id]
            for frame in sorted(set(pred_frames).intersection(gt_frames)):
                pred_mesh = scene_to_single_mesh(load_mesh_or_scene(pred_frames[frame]))
                pred_center = transform_points(mesh_center(pred_mesh, center_method)[None, :], pred_to_gt_transform)[0]
                cost += float(np.linalg.norm(pred_center - gt_frames[frame]))
                count += 1
        if count == 0:
            continue
        cost /= count
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    if best_perm is None:
        return {object_id: object_id for object_id in sorted(set(pred_tracks).intersection(gt_position_tracks))}
    return dict(zip(pred_ids, best_perm))


def estimated_voxel_cells(mesh: trimesh.Trimesh, pitch: float) -> int:
    if pitch <= 0.0 or mesh.is_empty:
        return 0
    extents = np.maximum(np.asarray(mesh.extents, dtype=np.float64), 0.0)
    grid_shape = np.maximum(np.ceil(extents / pitch).astype(np.int64) + 1, 1)
    return int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2])


def voxel_iou_or_nan(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    num_grids: int,
    scale: float,
    *,
    skip: bool = False,
    max_vertices: int = 0,
    max_faces: int = 0,
    max_cells: int = 0,
) -> float:
    if skip:
        return float("nan")
    if num_grids <= 0 or scale <= 0.0:
        print("[warn] skipped voxel IoU: --iou-num-grids and --iou-scale must be positive", flush=True)
        return float("nan")
    for label, mesh in (("prediction", pred_mesh), ("ground truth", gt_mesh)):
        if max_vertices > 0 and len(mesh.vertices) > max_vertices:
            print(f"[warn] skipped voxel IoU: {label} has {len(mesh.vertices)} vertices (limit {max_vertices})", flush=True)
            return float("nan")
        if max_faces > 0 and len(mesh.faces) > max_faces:
            print(f"[warn] skipped voxel IoU: {label} has {len(mesh.faces)} faces (limit {max_faces})", flush=True)
            return float("nan")
        cells = estimated_voxel_cells(mesh, scale / num_grids)
        if max_cells > 0 and cells > max_cells:
            print(f"[warn] skipped voxel IoU: {label} bounding grid has about {cells} cells (limit {max_cells})", flush=True)
            return float("nan")
    try:
        return float(compute_IoU(pred_mesh, gt_mesh, num_grids=num_grids, scale=scale))
    except Exception as exc:
        print(f"[warn] voxel IoU failed: {exc}", flush=True)
        return float("nan")


def mesh_metrics(
    pred_path: Path,
    gt_path: Path,
    num_samples: int,
    threshold: float,
    metric: str,
    iou_num_grids: int,
    iou_scale: float,
    pred_transform: np.ndarray | None = None,
    **voxel_options: Any,
) -> dict[str, Any]:
    pred_geom = load_mesh_or_scene(pred_path)
    gt_geom = load_mesh_or_scene(gt_path)
    pred_mesh = scene_to_single_mesh(pred_geom).copy()
    if pred_transform is not None:
        pred_mesh.apply_transform(pred_transform)
    gt_mesh = scene_to_single_mesh(gt_geom)
    cd, f_score = compute_cd_and_f_score(pred_mesh, gt_mesh, num_samples=num_samples, threshold=threshold, metric=metric)
    pred_bounds = bounds_from_mesh_or_scene(pred_mesh)
    gt_bounds = bounds_from_mesh_or_scene(gt_geom)
    return {
        "chamfer_distance": float(cd),
        "f_score": float(f_score),
        "voxel_iou": voxel_iou_or_nan(pred_mesh, gt_mesh, iou_num_grids, iou_scale, **voxel_options),
        "bbox_iou_3d": bbox_iou_3d(pred_bounds, gt_bounds),
        "bbox_overlap_volume": bbox_overlap_volume(pred_bounds, gt_bounds),
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
    }


def mesh_metrics_for_gt_mesh(
    pred_path: Path,
    gt_mesh: trimesh.Trimesh,
    gt_label: str,
    num_samples: int,
    threshold: float,
    metric: str,
    iou_num_grids: int,
    iou_scale: float,
    pred_transform: np.ndarray | None = None,
    **voxel_options: Any,
) -> dict[str, Any]:
    pred_geom = load_mesh_or_scene(pred_path)
    pred_mesh = scene_to_single_mesh(pred_geom).copy()
    if pred_transform is not None:
        pred_mesh.apply_transform(pred_transform)
    cd, f_score = compute_cd_and_f_score(pred_mesh, gt_mesh, num_samples=num_samples, threshold=threshold, metric=metric)
    pred_bounds = bounds_from_mesh_or_scene(pred_mesh)
    gt_bounds = bounds_from_mesh_or_scene(gt_mesh)
    return {
        "chamfer_distance": float(cd),
        "f_score": float(f_score),
        "voxel_iou": voxel_iou_or_nan(pred_mesh, gt_mesh, iou_num_grids, iou_scale, **voxel_options),
        "bbox_iou_3d": bbox_iou_3d(pred_bounds, gt_bounds),
        "bbox_overlap_volume": bbox_overlap_volume(pred_bounds, gt_bounds),
        "pred_path": str(pred_path),
        "gt_path": gt_label,
    }


METRIC_KEYS = ("chamfer_distance", "f_score", "voxel_iou", "bbox_iou_3d", "bbox_overlap_volume")


def add_metric_family(row: dict[str, Any], family: str, metrics: dict[str, Any]) -> None:
    for key in METRIC_KEYS:
        row[f"{family}_{key}"] = metrics[key]


def combined_metric_row(raw_metrics: dict[str, Any] | None, aligned_metrics: dict[str, Any] | None) -> dict[str, Any]:
    primary_metrics = raw_metrics if raw_metrics is not None else aligned_metrics
    if primary_metrics is None:
        raise ValueError("At least one metric family must be computed")
    row = dict(primary_metrics)
    if raw_metrics is not None:
        add_metric_family(row, "raw", raw_metrics)
    if aligned_metrics is not None:
        add_metric_family(row, "aligned", aligned_metrics)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def add_prefixed_means(summary: dict[str, Any], prefix: str, rows: list[dict[str, Any]]) -> None:
    for key in METRIC_KEYS:
        summary[f"{prefix}_{key}_mean"] = mean_or_nan([float(row[key]) for row in rows if key in row])
    summary[f"{prefix}_per_frame_chamfer_distance_mean"] = summary[f"{prefix}_chamfer_distance_mean"]
    summary[f"{prefix}_per_frame_iou_mean"] = summary[f"{prefix}_voxel_iou_mean"]

    for family in ("raw", "aligned"):
        if not any(f"{family}_chamfer_distance" in row for row in rows):
            continue
        for key in METRIC_KEYS:
            family_key = f"{family}_{key}"
            summary[f"{prefix}_{family_key}_mean"] = mean_or_nan([float(row[family_key]) for row in rows if family_key in row])
        summary[f"{prefix}_{family}_per_frame_chamfer_distance_mean"] = summary[f"{prefix}_{family}_chamfer_distance_mean"]
        summary[f"{prefix}_{family}_per_frame_iou_mean"] = summary[f"{prefix}_{family}_voxel_iou_mean"]


def main() -> None:
    args = parse_args()
    pred_dir = args.pred_dir.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()
    output_dir = (args.output_dir or (pred_dir / "eval_reconstruction")).expanduser().resolve()

    scene_rows = []
    pred_scenes = frame_paths(pred_dir, args.pred_scene_glob)
    gt_scenes = frame_paths(gt_dir, args.gt_scene_glob)
    gt_metadata_scenes = {} if gt_scenes else metadata_gt_scene_frames(gt_dir)

    pred_tracks = dynamic_object_paths(pred_dir)
    gt_file_tracks = dynamic_object_paths(gt_dir)
    gt_object_tracks: dict[str, dict[int, Path | tuple[Path, Path, str]]] = gt_file_tracks or metadata_gt_tracks(gt_dir)
    gt_position_tracks = metadata_gt_position_tracks(gt_dir)
    alignment_source = "none"

    object_assignment = {object_id: object_id for object_id in sorted(set(pred_tracks).intersection(gt_object_tracks))}
    if not object_assignment and gt_position_tracks:
        object_assignment = {object_id: object_id for object_id in sorted(set(pred_tracks).intersection(gt_position_tracks))}

    if args.alignment == "first_frame_similarity":
        alignment_pred_points, alignment_gt_points = collect_first_frame_point_clouds(
            pred_scenes,
            gt_scenes,
            gt_metadata_scenes,
            pred_tracks,
            gt_object_tracks,
            object_assignment,
            args.alignment_samples_per_frame,
        )
        pred_to_gt_transform = alignment_transform(args.alignment, alignment_pred_points, alignment_gt_points)
    else:
        alignment_pred_points, alignment_gt_points = collect_object_center_pairs_from_positions(
            pred_tracks, gt_position_tracks, center_method=args.com_method
        )
        alignment_source = "metadata_object_positions" if len(alignment_pred_points) else alignment_source
        if len(alignment_pred_points) == 0:
            alignment_pred_points, alignment_gt_points = collect_object_center_pairs(
                pred_tracks, gt_object_tracks, center_method=args.com_method
            )
            alignment_source = "object_center_tracks" if len(alignment_pred_points) else alignment_source
        if len(alignment_pred_points) == 0:
            alignment_pred_points, alignment_gt_points = collect_scene_center_pairs(
                pred_scenes, gt_scenes, gt_metadata_scenes, center_method=args.com_method
            )
            alignment_source = "scene_centers" if len(alignment_pred_points) else alignment_source
        pred_to_gt_transform = alignment_transform(args.alignment, alignment_pred_points, alignment_gt_points)

    if args.object_assignment == "best" and (gt_object_tracks or gt_position_tracks):
        if gt_position_tracks:
            object_assignment = best_object_assignment_from_positions(
                pred_tracks, gt_position_tracks, pred_to_gt_transform, center_method=args.com_method
            )
        else:
            object_assignment = best_object_assignment(
                pred_tracks, gt_object_tracks, pred_to_gt_transform, center_method=args.com_method
            )
        if args.alignment == "first_frame_similarity":
            alignment_pred_points, alignment_gt_points = collect_first_frame_point_clouds(
                pred_scenes,
                gt_scenes,
                gt_metadata_scenes,
                pred_tracks,
                gt_object_tracks,
                object_assignment,
                args.alignment_samples_per_frame,
            )
            pred_to_gt_transform = alignment_transform(args.alignment, alignment_pred_points, alignment_gt_points)
        elif args.alignment != "none":
            if gt_position_tracks:
                alignment_pred_points, alignment_gt_points = collect_object_center_pairs_from_positions(
                    pred_tracks, gt_position_tracks, object_assignment, center_method=args.com_method
                )
                alignment_source = "metadata_object_positions" if len(alignment_pred_points) else alignment_source
            else:
                alignment_pred_points, alignment_gt_points = collect_object_center_pairs(
                    pred_tracks, gt_object_tracks, object_assignment, center_method=args.com_method
                )
                alignment_source = "object_center_tracks" if len(alignment_pred_points) else alignment_source
            pred_to_gt_transform = alignment_transform(args.alignment, alignment_pred_points, alignment_gt_points)

    aligned_transform = None if args.alignment == "none" else pred_to_gt_transform
    skip_raw_metrics = bool(args.skip_raw_metrics and aligned_transform is not None)
    if args.skip_raw_metrics and aligned_transform is None:
        print("[warn] --skip-raw-metrics ignored because --alignment=none", flush=True)
    voxel_options = {
        "skip": args.skip_voxel_iou,
        "max_vertices": args.max_voxel_vertices,
        "max_faces": args.max_voxel_faces,
        "max_cells": args.max_voxel_cells,
    }

    scene_file_frames = sorted(set(pred_scenes).intersection(gt_scenes))
    scene_metadata_frames = sorted(set(pred_scenes).intersection(gt_metadata_scenes))
    total_scene_frames = len(scene_file_frames) + len(scene_metadata_frames)
    scene_progress = 0
    for frame in scene_file_frames:
        scene_progress += 1
        print(f"[scene {scene_progress}/{total_scene_frames}] frame {frame}", flush=True)
        raw_metrics = None if skip_raw_metrics else mesh_metrics(
            pred_scenes[frame],
            gt_scenes[frame],
            args.num_samples,
            args.threshold,
            args.metric,
            args.iou_num_grids,
            args.iou_scale,
            **voxel_options,
        )
        aligned_metrics = None if aligned_transform is None else mesh_metrics(
            pred_scenes[frame],
            gt_scenes[frame],
            args.num_samples,
            args.threshold,
            args.metric,
            args.iou_num_grids,
            args.iou_scale,
            pred_transform=aligned_transform,
            **voxel_options,
        )
        row = combined_metric_row(raw_metrics, aligned_metrics)
        row.update({"level": "scene", "frame": frame})
        scene_rows.append(row)

    for frame in scene_metadata_frames:
        scene_progress += 1
        print(f"[scene {scene_progress}/{total_scene_frames}] frame {frame}", flush=True)
        specs = gt_metadata_scenes[frame]
        gt_mesh = load_gt_scene(specs)
        gt_label = "+".join(f"{mesh_path}@{transform_path}:{ball_key}" for mesh_path, transform_path, ball_key in specs)
        raw_metrics = None if skip_raw_metrics else mesh_metrics_for_gt_mesh(
            pred_scenes[frame],
            gt_mesh,
            gt_label,
            args.num_samples,
            args.threshold,
            args.metric,
            args.iou_num_grids,
            args.iou_scale,
            **voxel_options,
        )
        aligned_metrics = None if aligned_transform is None else mesh_metrics_for_gt_mesh(
            pred_scenes[frame],
            gt_mesh,
            gt_label,
            args.num_samples,
            args.threshold,
            args.metric,
            args.iou_num_grids,
            args.iou_scale,
            pred_transform=aligned_transform,
            **voxel_options,
        )
        row = combined_metric_row(raw_metrics, aligned_metrics)
        row.update({"level": "scene", "frame": frame})
        scene_rows.append(row)

    object_rows = []
    if not args.skip_object_level:
        if args.object_assignment == "fixed":
            object_assignment = {object_id: object_id for object_id in sorted(set(pred_tracks).intersection(gt_object_tracks))}
        object_pairs = [
            (pred_object_id, gt_object_id, frame)
            for pred_object_id, gt_object_id in sorted(object_assignment.items())
            for frame in sorted(set(pred_tracks.get(pred_object_id, {})).intersection(gt_object_tracks.get(gt_object_id, {})))
        ]
        for object_progress, (pred_object_id, gt_object_id, frame) in enumerate(object_pairs, start=1):
            print(
                f"[object {object_progress}/{len(object_pairs)}] {pred_object_id}->{gt_object_id} frame {frame}",
                flush=True,
            )
            pred_frames = pred_tracks.get(pred_object_id, {})
            gt_frames = gt_object_tracks.get(gt_object_id, {})
            gt_spec = gt_frames[frame]
            gt_mesh = gt_track_mesh(gt_spec)
            gt_label = gt_track_label(gt_spec)
            raw_metrics = None if skip_raw_metrics else mesh_metrics_for_gt_mesh(
                pred_frames[frame],
                gt_mesh,
                gt_label,
                args.num_samples,
                args.threshold,
                args.metric,
                args.iou_num_grids,
                args.iou_scale,
                **voxel_options,
            )
            aligned_metrics = None if aligned_transform is None else mesh_metrics_for_gt_mesh(
                pred_frames[frame],
                gt_mesh,
                gt_label,
                args.num_samples,
                args.threshold,
                args.metric,
                args.iou_num_grids,
                args.iou_scale,
                pred_transform=aligned_transform,
                **voxel_options,
            )
            row = combined_metric_row(raw_metrics, aligned_metrics)
            row.update({
                "level": "object",
                "object_id": pred_object_id,
                "gt_object_id": gt_object_id,
                "frame": frame,
            })
            object_rows.append(row)

    if not args.allow_empty and not scene_rows and not object_rows:
        raise RuntimeError(
            "No reconstruction GT/pred GLB pairs were found. For raw synthetic GT, expected "
            f"dynamic object meshes under {gt_dir / 'meshes'} and "
            f"{gt_dir / 'transforms' / 'frame_0000.json'}. Pass --allow-empty to write NaN summaries."
        )

    summary: dict[str, Any] = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "num_scene_pairs": len(scene_rows),
        "num_object_pairs": len(object_rows),
        "num_samples": args.num_samples,
        "threshold": args.threshold,
        "iou_num_grids": args.iou_num_grids,
        "iou_scale": args.iou_scale,
        "skip_raw_metrics": skip_raw_metrics,
        "skip_voxel_iou": args.skip_voxel_iou,
        "max_voxel_vertices": args.max_voxel_vertices,
        "max_voxel_faces": args.max_voxel_faces,
        "max_voxel_cells": args.max_voxel_cells,
        "alignment": args.alignment,
        "object_assignment": args.object_assignment,
        "com_method": args.com_method,
        "alignment_num_points": int(len(alignment_pred_points)),
        "pred_to_gt_transform": pred_to_gt_transform.tolist(),
        "reconstruction_alignment": args.alignment,
        "reconstruction_alignment_scope": (
            "first_frame_point_cloud" if args.alignment == "first_frame_similarity"
            else alignment_source if args.alignment != "none"
            else "none"
        ),
        "reconstruction_alignment_num_points": int(len(alignment_pred_points)),
        "reconstruction_alignment_center_method": args.com_method,
        "reconstruction_alignment_scale": float(np.cbrt(abs(np.linalg.det(pred_to_gt_transform[:3, :3])))) if args.alignment != "none" else 1.0,
        "reconstruction_alignment_rotation": (
            (pred_to_gt_transform[:3, :3] / float(np.cbrt(abs(np.linalg.det(pred_to_gt_transform[:3, :3]))))).tolist()
            if args.alignment != "none" and abs(float(np.linalg.det(pred_to_gt_transform[:3, :3]))) > 0.0
            else np.eye(3, dtype=np.float64).tolist()
        ),
        "reconstruction_alignment_translation_x": float(pred_to_gt_transform[0, 3]),
        "reconstruction_alignment_translation_y": float(pred_to_gt_transform[1, 3]),
        "reconstruction_alignment_translation_z": float(pred_to_gt_transform[2, 3]),
        "object_assignment_map": object_assignment,
    }
    add_prefixed_means(summary, "scene", scene_rows)
    add_prefixed_means(summary, "object", object_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "summary": summary,
                "scene": scene_rows,
                "objects": object_rows,
            },
            f,
            indent=2,
            allow_nan=True,
        )
    write_csv(output_dir / "scene_metrics.csv", scene_rows)
    write_csv(output_dir / "object_metrics.csv", object_rows)
    write_csv(output_dir / "summary.csv", [summary])
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

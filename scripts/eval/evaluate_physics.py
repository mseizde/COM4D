#!/usr/bin/env python3

"""Evaluate simple physical/geometric failure modes for COM4D GLB outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from curvesimilarities import dfd as discrete_frechet_distance
except Exception:  # pragma: no cover - optional dependency fallback
    discrete_frechet_distance = None

from src.utils.metric_utils import (  # noqa: E402
    align_umeyama,
    axis_index,
    bbox_iou_3d,
    bbox_overlap_area_xz,
    bbox_overlap_volume,
    bounds_from_mesh_or_scene,
    center_from_bounds,
    floor_penetration_depth,
    floor_support_error,
    is_floating,
    load_mesh_or_scene,
    scene_to_single_mesh,
    size_from_bounds,
    trajectory_acceleration_stats,
    trajectory_speed_stats,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inference-dir", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=None, help="Optional PyBullet physics_metadata.json.")
    ap.add_argument("--output-dir", type=Path, default=None, help="Defaults to <inference-dir>/eval_physics.")
    ap.add_argument("--floor-height", type=float, default=0.0)
    ap.add_argument("--up-axis", choices=("x", "y", "z"), default="y")
    ap.add_argument(
        "--metadata-up-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Up axis used in physics_metadata.json. PyBullet/Blender two-ball metadata is z-up.",
    )
    ap.add_argument("--support-tol", type=float, default=0.05)
    ap.add_argument("--overlap-volume-tol", type=float, default=1e-6)
    ap.add_argument("--fps", type=float, default=None, help="Override FPS. Defaults to metadata fps or 30.")
    ap.add_argument(
        "--com-method",
        choices=("vertex_centroid", "volume_center_mass", "bbox_center"),
        default="vertex_centroid",
        help=(
            "Predicted object COM proxy. vertex_centroid avoids bbox bias and is fast; "
            "volume_center_mass uses trimesh's uniform-density volume COM but can be very slow on large GLBs."
        ),
    )
    ap.add_argument(
        "--skip-basic-physics-summary",
        action="store_true",
        help="Omit legacy penetration/collision/scale/speed summary fields; detailed CSVs are still written.",
    )
    return ap.parse_args()


def frame_index(path: Path) -> int:
    match = re.search(r"frame_(\d+)\.glb$", path.name)
    if match is None:
        raise ValueError(f"Could not parse frame index from {path}")
    return int(match.group(1))


def dynamic_object_paths(inference_dir: Path) -> dict[str, list[Path]]:
    dynamic_dir = inference_dir / "dynamic"
    tracks: dict[str, list[Path]] = {}
    if not dynamic_dir.exists():
        return tracks
    for obj_dir in sorted(dynamic_dir.glob("object_*")):
        if obj_dir.is_dir():
            frames = sorted(obj_dir.glob("frame_*.glb"), key=frame_index)
            if frames:
                tracks[obj_dir.name] = frames
    return tracks


def load_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.expanduser().open("r") as f:
        return json.load(f)


def dynamic_metadata_keys(metadata: dict[str, Any] | None) -> list[str]:
    if metadata is None:
        return []
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
    radii = metadata.get("ball_radii")
    if isinstance(radii, dict):
        return sorted(radii.keys())
    frames = metadata.get("frames", [])
    if frames:
        first = frames[0].get("objects") if isinstance(frames[0].get("objects"), dict) else frames[0]
        return sorted(key for key in first if key.startswith("ball_"))
    return []


def object_to_metadata_key(object_ids: list[str], metadata: dict[str, Any] | None) -> dict[str, str]:
    keys = dynamic_metadata_keys(metadata)
    return {object_id: keys[idx] for idx, object_id in enumerate(sorted(object_ids)) if idx < len(keys)}


def metadata_frame_map(metadata: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if metadata is None:
        return {}
    return {int(frame["frame"]): frame for frame in metadata.get("frames", [])}


def expected_radius(metadata: dict[str, Any] | None, key: str) -> float | None:
    if metadata is None:
        return None
    objects = metadata.get("objects")
    if isinstance(objects, dict) and isinstance(objects.get(key), dict) and "radius" in objects[key]:
        return float(objects[key]["radius"])
    radii = metadata.get("ball_radii")
    if isinstance(radii, dict) and key in radii:
        return float(radii[key])
    if "ball_radius" in metadata:
        return float(metadata["ball_radius"])
    return None


def metadata_expected_supported(
    metadata_frames: dict[int, dict[str, Any]],
    frame: int,
    key: str,
    radius: float | None,
    metadata_up_axis: str,
    support_tol: float,
) -> bool | None:
    if radius is None or frame not in metadata_frames:
        return None
    frame_data = metadata_frames[frame]
    objects = frame_data.get("objects")
    obj = objects.get(key) if isinstance(objects, dict) else frame_data.get(key)
    if not isinstance(obj, dict) or "position" not in obj:
        return None
    up = axis_index(metadata_up_axis)
    bottom = float(obj["position"][up]) - radius
    return bool(abs(bottom) <= support_tol)


def metadata_object_position(
    metadata_frames: dict[int, dict[str, Any]],
    frame: int,
    key: str,
) -> np.ndarray | None:
    frame_data = metadata_frames.get(frame)
    if frame_data is None:
        return None
    objects = frame_data.get("objects")
    obj = objects.get(key) if isinstance(objects, dict) else frame_data.get(key)
    if not isinstance(obj, dict) or "position" not in obj:
        return None
    position = np.asarray(obj["position"], dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        return None
    return position


def mesh_center_of_mass(geom: Any, method: str) -> tuple[np.ndarray, str]:
    bounds_center = center_from_bounds(bounds_from_mesh_or_scene(geom))
    if method == "bbox_center":
        return bounds_center, "bbox_center"
    try:
        mesh = scene_to_single_mesh(geom)
        if method == "volume_center_mass":
            center = np.asarray(mesh.center_mass, dtype=np.float64)
            label = "volume_center_mass"
        else:
            center = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
            label = "vertex_centroid"
        if center.shape == (3,) and np.isfinite(center).all():
            return center, label
    except Exception:
        pass
    return bounds_center, "bbox_center_fallback"


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (rotation @ points.T).T + translation


def trajectory_error_stats(errors: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    if len(errors) == 0:
        return {"rmse": float("nan"), "mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
    }


def trajectory_discrete_frechet(pred_aligned: np.ndarray, gt: np.ndarray) -> float:
    if discrete_frechet_distance is None or len(pred_aligned) == 0 or len(gt) == 0:
        return float("nan")
    try:
        return float(discrete_frechet_distance(np.asarray(pred_aligned, dtype=np.float64), np.asarray(gt, dtype=np.float64)))
    except Exception:
        return float("nan")


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


def max_or_nan(values: list[float]) -> float:
    return float(np.max(values)) if values else float("nan")


def finite_rate(flags: list[bool]) -> float:
    return float(np.mean(np.asarray(flags, dtype=np.float32))) if flags else float("nan")


def main() -> None:
    args = parse_args()
    inference_dir = args.inference_dir.expanduser().resolve()
    output_dir = (args.output_dir or (inference_dir / "eval_physics")).expanduser().resolve()
    metadata = load_metadata(args.metadata)
    fps = float(args.fps or (metadata or {}).get("fps", 30.0))

    tracks = dynamic_object_paths(inference_dir)
    metadata_keys = object_to_metadata_key(list(tracks), metadata)
    metadata_frames = metadata_frame_map(metadata)

    frame_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    records_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    centers_by_object: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    gt_centers_by_object: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)

    for object_id, paths in tracks.items():
        metadata_key = metadata_keys.get(object_id)
        radius = expected_radius(metadata, metadata_key) if metadata_key else None
        expected_diameter = None if radius is None else 2.0 * radius

        for path in paths:
            frame = frame_index(path)
            geom = load_mesh_or_scene(path)
            bounds = bounds_from_mesh_or_scene(geom)
            center, center_method = mesh_center_of_mass(geom, args.com_method)
            bbox_center = center_from_bounds(bounds)
            size = size_from_bounds(bounds)
            gt_center = metadata_object_position(metadata_frames, frame, metadata_key) if metadata_key else None
            support_error = floor_support_error(bounds, args.floor_height, args.up_axis)
            penetration = floor_penetration_depth(bounds, args.floor_height, args.up_axis)
            floating = is_floating(bounds, args.floor_height, args.support_tol, args.up_axis)
            expected_supported = metadata_expected_supported(
                metadata_frames,
                frame,
                metadata_key,
                radius,
                args.metadata_up_axis,
                args.support_tol,
            ) if metadata_key else None
            scale_error = float("nan")
            if expected_diameter is not None and expected_diameter > 0.0:
                scale_error = float(abs(float(np.max(size)) - expected_diameter) / expected_diameter)

            row = {
                "object_id": object_id,
                "metadata_key": metadata_key,
                "frame": frame,
                "path": str(path),
                "support_error": support_error,
                "floor_penetration_depth": penetration,
                "is_floating": floating,
                "expected_floor_supported": expected_supported,
                "support_consistent": None if expected_supported is None else (abs(support_error) <= args.support_tol),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "center_method": center_method,
                "bbox_center_x": float(bbox_center[0]),
                "bbox_center_y": float(bbox_center[1]),
                "bbox_center_z": float(bbox_center[2]),
                "gt_center_x": None if gt_center is None else float(gt_center[0]),
                "gt_center_y": None if gt_center is None else float(gt_center[1]),
                "gt_center_z": None if gt_center is None else float(gt_center[2]),
                "has_gt_center": gt_center is not None,
                "bbox_min_x": float(bounds[0, 0]),
                "bbox_min_y": float(bounds[0, 1]),
                "bbox_min_z": float(bounds[0, 2]),
                "bbox_max_x": float(bounds[1, 0]),
                "bbox_max_y": float(bounds[1, 1]),
                "bbox_max_z": float(bounds[1, 2]),
                "bbox_size_x": float(size[0]),
                "bbox_size_y": float(size[1]),
                "bbox_size_z": float(size[2]),
                "scale_error": scale_error,
            }
            object_rows.append(row)
            records_by_frame[frame].append({"object_id": object_id, "bounds": bounds})
            centers_by_object[object_id].append((frame, center))
            if gt_center is not None:
                gt_centers_by_object[object_id].append((frame, gt_center))

    pair_rows: list[dict[str, Any]] = []
    collision_flags = []
    for frame, records in sorted(records_by_frame.items()):
        for i, rec_a in enumerate(records):
            for rec_b in records[i + 1 :]:
                overlap_volume = bbox_overlap_volume(rec_a["bounds"], rec_b["bounds"])
                overlaps = overlap_volume > args.overlap_volume_tol
                collision_flags.append(overlaps)
                pair_rows.append(
                    {
                        "frame": frame,
                        "object_a": rec_a["object_id"],
                        "object_b": rec_b["object_id"],
                        "bbox_overlap_volume": overlap_volume,
                        "bbox_overlap_area_xz": bbox_overlap_area_xz(rec_a["bounds"], rec_b["bounds"]),
                        "bbox_iou_3d": bbox_iou_3d(rec_a["bounds"], rec_b["bounds"]),
                        "bbox_collision": overlaps,
                    }
                )

    global_pred_points = []
    global_gt_points = []
    for object_id, items in sorted(centers_by_object.items()):
        gt_by_frame = {frame: center for frame, center in gt_centers_by_object.get(object_id, [])}
        for frame, pred_center in sorted(items, key=lambda item: item[0]):
            gt_center = gt_by_frame.get(frame)
            if gt_center is not None:
                global_pred_points.append(pred_center)
                global_gt_points.append(gt_center)

    global_alignment_scale = float("nan")
    global_alignment_rotation = np.eye(3, dtype=np.float64)
    global_alignment_translation = np.full(3, float("nan"), dtype=np.float64)
    global_alignment_points = len(global_pred_points)
    if global_alignment_points:
        try:
            global_alignment_scale, global_alignment_rotation, global_alignment_translation = align_umeyama(
                np.stack(global_gt_points, axis=0),
                np.stack(global_pred_points, axis=0),
                known_scale=False,
            )
        except ValueError:
            global_alignment_points = 0

    trajectory_rows: list[dict[str, Any]] = []
    for object_id, items in sorted(centers_by_object.items()):
        items = sorted(items, key=lambda item: item[0])
        frames = [frame for frame, _ in items]
        centers = np.stack([center for _, center in items], axis=0)
        speed = trajectory_speed_stats(centers, fps)
        accel = trajectory_acceleration_stats(centers, fps)
        expected_frames = set(range(min(frames), max(frames) + 1)) if frames else set()
        missing = len(expected_frames.difference(frames))

        gt_items = sorted(gt_centers_by_object.get(object_id, []), key=lambda item: item[0])
        gt_by_frame = {frame: center for frame, center in gt_items}
        pred_by_frame = {frame: center for frame, center in items}
        common_frames = sorted(set(pred_by_frame).intersection(gt_by_frame))
        pred_traj = np.stack([pred_by_frame[frame] for frame in common_frames], axis=0) if common_frames else np.empty((0, 3), dtype=np.float64)
        gt_traj = np.stack([gt_by_frame[frame] for frame in common_frames], axis=0) if common_frames else np.empty((0, 3), dtype=np.float64)
        pred_aligned = pred_traj
        ate_errors = np.asarray([], dtype=np.float64)
        if len(common_frames) >= 1 and global_alignment_points:
            pred_aligned = apply_similarity(
                pred_traj,
                global_alignment_scale,
                global_alignment_rotation,
                global_alignment_translation,
            )
            ate_errors = np.linalg.norm(gt_traj - pred_aligned, axis=1)
        ate = trajectory_error_stats(ate_errors)
        frechet = trajectory_discrete_frechet(pred_aligned, gt_traj) if len(common_frames) >= 1 and global_alignment_points else float("nan")
        gt_frame_set = set(gt_by_frame)
        pred_frame_set = set(pred_by_frame)
        trajectory_rows.append(
            {
                "object_id": object_id,
                "num_frames": len(frames),
                "first_frame": min(frames) if frames else None,
                "last_frame": max(frames) if frames else None,
                "missing_frame_count": missing,
                "missing_frame_rate": float(missing / len(expected_frames)) if expected_frames else 0.0,
                "gt_num_frames": len(gt_items),
                "matched_frame_count": len(common_frames),
                "unmatched_pred_frame_count": len(pred_frame_set.difference(gt_frame_set)),
                "unmatched_gt_frame_count": len(gt_frame_set.difference(pred_frame_set)),
                "same_frame_count_as_gt": len(pred_frame_set) == len(gt_frame_set),
                "same_frame_indices_as_gt": pred_frame_set == gt_frame_set,
                "first_matched_frame": min(common_frames) if common_frames else None,
                "last_matched_frame": max(common_frames) if common_frames else None,
                "speed_mean": speed["mean"],
                "speed_max": speed["max"],
                "speed_std": speed["std"],
                "acceleration_mean": accel["mean"],
                "acceleration_max": accel["max"],
                "acceleration_std": accel["std"],
                "ate_alignment": "global_sim3_umeyama",
                "ate_alignment_scope": "all_dynamic_object_trajectories",
                "ate_alignment_num_points": global_alignment_points,
                "ate_num_frames": len(common_frames),
                "ate_rmse": ate["rmse"],
                "ate_mean": ate["mean"],
                "ate_median": ate["median"],
                "ate_max": ate["max"],
                "ate_scale": global_alignment_scale,
                "ate_translation_x": float(global_alignment_translation[0]),
                "ate_translation_y": float(global_alignment_translation[1]),
                "ate_translation_z": float(global_alignment_translation[2]),
                "discrete_frechet_aligned": frechet,
                "frechet_implementation": "curvesimilarities.dfd" if discrete_frechet_distance is not None else "unavailable",
            }
        )

    penetration_values = [float(row["floor_penetration_depth"]) for row in object_rows]
    floating_flags = [bool(row["is_floating"]) for row in object_rows]
    support_flags = [bool(row["support_consistent"]) for row in object_rows if row["support_consistent"] is not None]
    scale_values = [float(row["scale_error"]) for row in object_rows if np.isfinite(float(row["scale_error"]))]

    summary = {
        "inference_dir": str(inference_dir),
        "metadata": str(args.metadata.expanduser().resolve()) if args.metadata else None,
        "num_objects": len(tracks),
        "num_object_frames": len(object_rows),
        "num_pair_frames": len(pair_rows),
        "com_method": args.com_method,
        "trajectory_ate_rmse_mean": mean_or_nan([float(row["ate_rmse"]) for row in trajectory_rows if np.isfinite(float(row["ate_rmse"]))]),
        "trajectory_ate_mean_mean": mean_or_nan([float(row["ate_mean"]) for row in trajectory_rows if np.isfinite(float(row["ate_mean"]))]),
        "trajectory_ate_median_mean": mean_or_nan([float(row["ate_median"]) for row in trajectory_rows if np.isfinite(float(row["ate_median"]))]),
        "trajectory_ate_max_mean": mean_or_nan([float(row["ate_max"]) for row in trajectory_rows if np.isfinite(float(row["ate_max"]))]),
        "trajectory_discrete_frechet_aligned_mean": mean_or_nan([float(row["discrete_frechet_aligned"]) for row in trajectory_rows if np.isfinite(float(row["discrete_frechet_aligned"]))]),
        "trajectory_frame_indices_match_rate": finite_rate([bool(row["same_frame_indices_as_gt"]) for row in trajectory_rows if int(row["gt_num_frames"]) > 0]),
        "trajectory_alignment": "global_sim3_umeyama",
        "trajectory_alignment_scope": "all_dynamic_object_trajectories",
        "trajectory_alignment_num_points": global_alignment_points,
        "trajectory_alignment_scale": global_alignment_scale,
        "trajectory_alignment_translation_x": float(global_alignment_translation[0]),
        "trajectory_alignment_translation_y": float(global_alignment_translation[1]),
        "trajectory_alignment_translation_z": float(global_alignment_translation[2]),
        "trajectory_alignment_rotation": global_alignment_rotation.tolist(),
    }
    if not args.skip_basic_physics_summary:
        summary.update(
            {
                "floor_penetration_mean": mean_or_nan(penetration_values),
                "floor_penetration_max": max_or_nan(penetration_values),
                "floating_rate": finite_rate(floating_flags),
                "bbox_collision_rate": finite_rate(collision_flags),
                "support_consistency_rate": finite_rate(support_flags),
                "scale_error_mean": mean_or_nan(scale_values),
                "scale_error_max": max_or_nan(scale_values),
                "trajectory_speed_max_mean": mean_or_nan([float(row["speed_max"]) for row in trajectory_rows]),
                "trajectory_acceleration_max_mean": mean_or_nan([float(row["acceleration_max"]) for row in trajectory_rows]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "summary": summary,
                "objects": object_rows,
                "pairs": pair_rows,
                "trajectories": trajectory_rows,
            },
            f,
            indent=2,
            allow_nan=True,
        )
    write_csv(output_dir / "object_metrics.csv", object_rows)
    write_csv(output_dir / "pair_metrics.csv", pair_rows)
    write_csv(output_dir / "trajectory_metrics.csv", trajectory_rows)
    write_csv(output_dir / "summary.csv", [summary])
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

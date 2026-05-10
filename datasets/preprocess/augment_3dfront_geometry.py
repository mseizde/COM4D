#!/usr/bin/env python3
"""
Augment a COM4D/MIDI 3D-FRONT manifest with room-shell and geometry metadata.

The existing COM4D 3D-FRONT manifest stores one entry per rendered view and points
at MIDI assets:
    .../3D-FRONT-SCENE/<house_id>/<room_id>.glb
    .../3D-FRONT-RENDER/<house_id>/<room_id>/render_0000.webp

This script adds optional, backward-compatible fields:
    - house_id, room_id
    - room_scene_dir
    - floor_path, wall_path, ceiling_path
    - depth_path, normal_path, semantic_path, render_meta_path
    - geometry_metadata_path

It also writes one compact sidecar JSON per room with room-shell paths and
floor/wall/ceiling plane summaries extracted from the original 3D-FRONT JSON zip.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    def tqdm(iterable, **kwargs):
        return iterable


SHELL_NAMES = {
    "floor.glb": "floor_path",
    "wall.glb": "wall_path",
    "ceil.glb": "ceiling_path",
    "ceiling.glb": "ceiling_path",
}
EXCLUDED_OBJECT_GLB_NAMES = set(SHELL_NAMES) | {"others.glb"}
GEOMETRY_TYPES = {"Floor", "Ceiling"}
IMAGE_INDEX_RE = re.compile(r"(?:re)?render_(\d+)\.webp$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Augment COM4D 3D-FRONT manifest with shell paths and original JSON geometry metadata."
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/mseizde/com4d/COM4D/dataset_json/3dfront.json"),
        help="Input COM4D 3D-FRONT manifest.",
    )
    ap.add_argument(
        "--front-zip",
        type=Path,
        default=Path("/mnt/mocap_b/work/com4d/datasets/raw/3D-FRONT/3D-FRONT-ORIGINAL/3D-FRONT.zip"),
        help="Original 3D-FRONT.zip containing 3D-FRONT/<house_id>.json files.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/data/mseizde/com4d/COM4D/dataset_json/3dfront_augmented_geometry.json"),
        help="Output augmented manifest path.",
    )
    ap.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("/mnt/mocap_b/work/com4d/datasets/processed/3dfront/geometry_metadata_v2"),
        help="Output directory for one compact geometry sidecar JSON per room.",
    )
    ap.add_argument(
        "--original-json-subset-root",
        type=Path,
        default=Path("/mnt/mocap_b/work/com4d/datasets/raw/3D-FRONT/original_subset/3D-FRONT"),
        help="Optional directory where selected original scene JSONs are extracted.",
    )
    ap.add_argument(
        "--no-extract-original-jsons",
        action="store_true",
        help="Do not write selected original scene JSON files into --original-json-subset-root.",
    )
    ap.add_argument(
        "--indent",
        type=int,
        default=0,
        help="JSON indentation for generated files. Default is compact JSON because the augmented manifest is large.",
    )
    return ap.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def dump_json(data: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=indent if indent > 0 else None)


def resolve_room_ids(entry: Dict[str, Any]) -> Tuple[str, str, Path]:
    mesh_path = Path(entry["mesh_path"])
    house_id = mesh_path.parent.name
    room_id = mesh_path.stem
    room_scene_dir = mesh_path.parent / room_id
    return house_id, room_id, room_scene_dir


def image_index(image_path: str) -> Optional[int]:
    match = IMAGE_INDEX_RE.search(Path(image_path).name)
    if match is None:
        return None
    return int(match.group(1))


def existing_path(path: Path) -> Optional[str]:
    return str(path.resolve()) if path.exists() else None


def collect_shell_paths(room_scene_dir: Path) -> Dict[str, Optional[str]]:
    paths: Dict[str, Optional[str]] = {
        "floor_path": None,
        "wall_path": None,
        "ceiling_path": None,
    }
    for filename, key in SHELL_NAMES.items():
        path = room_scene_dir / filename
        if path.exists():
            paths[key] = str(path.resolve())
    return paths


def collect_object_glbs(room_scene_dir: Path) -> List[str]:
    if not room_scene_dir.is_dir():
        return []
    glbs = []
    for path in sorted(room_scene_dir.glob("*.glb")):
        if path.name.lower() in EXCLUDED_OBJECT_GLB_NAMES:
            continue
        glbs.append(str(path.resolve()))
    return glbs


def collect_render_paths(render_dir: Path, idx: Optional[int]) -> Dict[str, Optional[str]]:
    paths = {"depth_path": None, "normal_path": None, "semantic_path": None, "render_meta_path": None}
    if not render_dir.is_dir():
        return paths
    paths["render_meta_path"] = existing_path(render_dir / "meta.json")
    if idx is None:
        return paths
    paths["depth_path"] = existing_path(render_dir / f"depth_{idx:04d}.exr")
    paths["normal_path"] = existing_path(render_dir / f"normal_{idx:04d}.webp")
    paths["semantic_path"] = existing_path(render_dir / f"semantic_{idx:04d}.png")
    return paths


def quat_to_matrix_xyzw(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_vertices(vertices: np.ndarray, child: Dict[str, Any]) -> np.ndarray:
    scale = np.asarray(child.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    rot = quat_to_matrix_xyzw(child.get("rot", [0.0, 0.0, 0.0, 1.0]))
    pos = np.asarray(child.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    return (vertices * scale) @ rot.T + pos


def transform_normals(normals: np.ndarray, child: Dict[str, Any]) -> np.ndarray:
    rot = quat_to_matrix_xyzw(child.get("rot", [0.0, 0.0, 0.0, 1.0]))
    out = normals @ rot.T
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def triangle_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(vertices) == 0 or len(faces) == 0:
        return 0.0
    tris = vertices[faces]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return float((np.linalg.norm(cross, axis=1) * 0.5).sum())


def summarize_mesh_plane(mesh: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    vertices = np.asarray(mesh.get("xyz", []), dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh.get("faces", []), dtype=np.int64).reshape(-1, 3)
    normals_raw = np.asarray(mesh.get("normal", []), dtype=np.float64).reshape(-1, 3)

    world_vertices = transform_vertices(vertices, child)
    if len(normals_raw) > 0:
        world_normals = transform_normals(normals_raw, child)
        normal = world_normals.mean(axis=0)
    elif len(faces) > 0:
        tris = world_vertices[faces]
        normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]).mean(axis=0)
    else:
        normal = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    normal_norm = float(np.linalg.norm(normal))
    if normal_norm > 1e-12:
        normal = normal / normal_norm

    center = world_vertices.mean(axis=0) if len(world_vertices) else np.zeros(3, dtype=np.float64)
    offset = -float(np.dot(normal, center)) if normal_norm > 1e-12 else 0.0
    area = triangle_area(world_vertices, faces)

    return {
        "type": mesh.get("type"),
        "uid": mesh.get("uid"),
        "instanceid": mesh.get("instanceid"),
        "child_instanceid": child.get("instanceid"),
        "normal": [round(float(v), 8) for v in normal.tolist()],
        "offset": round(float(offset), 8),
        "center": [round(float(v), 8) for v in center.tolist()],
        "area": round(area, 8),
        "num_vertices": int(len(world_vertices)),
        "num_faces": int(len(faces)),
    }


def geometry_class(mesh_type: Optional[str]) -> Optional[str]:
    if mesh_type in {"Floor", "Ceiling"}:
        return mesh_type
    if isinstance(mesh_type, str) and mesh_type.startswith("Wall"):
        return "Wall"
    return None


def extract_room_geometry(scene: Dict[str, Any], house_id: str, room_id: str) -> Dict[str, Any]:
    rooms = scene.get("scene", {}).get("room", [])
    room = next((r for r in rooms if r.get("instanceid") == room_id), None)
    if room is None:
        return {
            "house_id": house_id,
            "room_id": room_id,
            "found_in_original_json": False,
            "geometry_source": "missing_original_room",
        }

    mesh_by_uid = {m.get("uid"): m for m in scene.get("mesh", []) if isinstance(m, dict)}
    planes_by_type: Dict[str, List[Dict[str, Any]]] = {"Floor": [], "Wall": [], "Ceiling": []}
    for child in room.get("children", []):
        mesh = mesh_by_uid.get(child.get("ref"))
        if mesh is None:
            continue
        cls = geometry_class(mesh.get("type"))
        if cls is None:
            continue
        planes_by_type[cls].append(summarize_mesh_plane(mesh, child))

    floor_planes = sorted(planes_by_type["Floor"], key=lambda p: p["area"], reverse=True)
    wall_planes = sorted(planes_by_type["Wall"], key=lambda p: p["area"], reverse=True)
    ceiling_planes = sorted(planes_by_type["Ceiling"], key=lambda p: p["area"], reverse=True)

    return {
        "house_id": house_id,
        "room_id": room_id,
        "room_type": room.get("type"),
        "room_size": room.get("size"),
        "found_in_original_json": True,
        "geometry_source": "gt_3dfront_original_json",
        "gravity": [0.0, -1.0, 0.0],
        "ground_planes": floor_planes,
        "wall_planes": wall_planes,
        "ceiling_planes": ceiling_planes,
        "counts": {
            "floor": len(floor_planes),
            "wall": len(wall_planes),
            "ceiling": len(ceiling_planes),
        },
    }


def room_metadata_path(metadata_root: Path, house_id: str, room_id: str) -> Path:
    return metadata_root / house_id / f"{room_id}.json"


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)

    rooms: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in manifest:
        house_id, room_id, room_scene_dir = resolve_room_ids(entry)
        render_dir = Path(entry["image_path"]).parent
        rooms.setdefault(
            (house_id, room_id),
            {
                "room_scene_dir": room_scene_dir,
                "render_dir": render_dir,
            },
        )

    missing_original_jsons: List[str] = []
    missing_original_rooms: List[Tuple[str, str]] = []
    current_house_id: Optional[str] = None
    current_scene: Optional[Dict[str, Any]] = None
    with zipfile.ZipFile(args.front_zip) as z:
        names = set(z.namelist())
        if not args.no_extract_original_jsons:
            args.original_json_subset_root.mkdir(parents=True, exist_ok=True)

        for house_id, room_id in tqdm(sorted(rooms), desc="Room geometry"):
            zip_name = f"3D-FRONT/{house_id}.json"
            original_subset_path = args.original_json_subset_root / f"{house_id}.json"
            if zip_name not in names:
                missing_original_jsons.append(house_id)
                geometry = {
                    "house_id": house_id,
                    "room_id": room_id,
                    "found_in_original_json": False,
                    "geometry_source": "missing_original_json",
                }
            else:
                if current_house_id != house_id:
                    raw = z.read(zip_name)
                    current_scene = json.loads(raw)
                    current_house_id = house_id
                else:
                    raw = None
                if not args.no_extract_original_jsons and not original_subset_path.exists():
                    if raw is None:
                        raw = z.read(zip_name)
                    original_subset_path.write_bytes(raw)
                assert current_scene is not None
                geometry = extract_room_geometry(current_scene, house_id, room_id)
                if not geometry.get("found_in_original_json", False):
                    missing_original_rooms.append((house_id, room_id))

            room_info = rooms[(house_id, room_id)]
            room_scene_dir = Path(room_info["room_scene_dir"])
            render_dir = Path(room_info["render_dir"])
            shell_paths = collect_shell_paths(room_scene_dir)
            object_glb_paths = collect_object_glbs(room_scene_dir)
            metadata = {
                **geometry,
                "room_scene_dir": str(room_scene_dir.resolve()) if room_scene_dir.exists() else str(room_scene_dir),
                "render_dir": str(render_dir.resolve()) if render_dir.exists() else str(render_dir),
                "room_shell_paths": shell_paths,
                "object_glb_paths": object_glb_paths,
                "original_scene_json_path": (
                    str(original_subset_path.resolve())
                    if (not args.no_extract_original_jsons and original_subset_path.exists())
                    else None
                ),
            }
            meta_path = room_metadata_path(args.metadata_root, house_id, room_id)
            dump_json(metadata, meta_path, indent=args.indent)
            room_info["metadata_path"] = meta_path
            room_info["shell_paths"] = shell_paths
            room_info["num_object_glbs"] = len(object_glb_paths)
            room_info["found_in_original_json"] = bool(metadata.get("found_in_original_json", False))

    augmented = []
    for entry in tqdm(manifest, desc="Manifest entries"):
        out_entry = dict(entry)
        house_id, room_id, room_scene_dir = resolve_room_ids(entry)
        render_dir = Path(entry["image_path"]).parent
        room_info = rooms[(house_id, room_id)]
        idx = image_index(entry["image_path"])

        out_entry.update(
            {
                "house_id": house_id,
                "room_id": room_id,
                "room_scene_dir": str(room_scene_dir.resolve()) if room_scene_dir.exists() else str(room_scene_dir),
                "geometry_metadata_path": str(Path(room_info["metadata_path"]).resolve()),
                "num_object_glbs": int(room_info["num_object_glbs"]),
            }
        )
        out_entry.update(room_info["shell_paths"])
        out_entry.update(collect_render_paths(render_dir, idx))
        augmented.append(out_entry)

    dump_json(augmented, args.output, indent=args.indent)

    rooms_with_shell = sum(
        1
        for info in rooms.values()
        if any(info["shell_paths"].values())
    )
    rooms_with_original = sum(
        1
        for info in rooms.values()
        if info.get("found_in_original_json", False)
    )
    print(f"Wrote augmented manifest: {args.output}")
    print(f"Wrote room metadata root: {args.metadata_root}")
    print(f"Manifest entries: {len(augmented)}")
    print(f"Unique rooms: {len(rooms)}")
    print(f"Rooms with MIDI shell GLBs: {rooms_with_shell}")
    print(f"Rooms matched in original JSON: {rooms_with_original}")
    print(f"Missing original JSONs: {len(set(missing_original_jsons))}")
    print(f"Missing rooms inside matched JSONs: {len(missing_original_rooms)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Convert synthetic physics pipeline outputs into COM4D 4D training data.

Input layout can be either one sequence directory:
  outputs/two_ball_dataset/sample_000001/physics_metadata.json

or a root containing many sequence directories:
  outputs/two_ball_dataset/sample_000001/physics_metadata.json
  outputs/two_ball_dataset/sample_000002/physics_metadata.json

This writes:
  <output-root>/glb/<sequence>/frame_0000.glb
  <output-root>/preprocessed/<sequence>_frame_0000/points.npy
  <output-root>/render/<sequence>/frame_0000.png
  <json-output>

The JSON format matches src/datasets/animated_frame.py.

The preprocessor is backward-compatible with the legacy two-ball metadata, and
also supports generalized metadata with an ``objects`` dictionary. In generalized
metadata, all objects with ``dynamic: true`` are reconstructed as parts. Static
objects can optionally be included as interaction parts for physics-focused
training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "two_ball_dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "processed" / "two_ball"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "dataset_json" / "two_ball.json"

sys.path.append(str(PROJECT_ROOT))
from src.utils.data_utils import mesh_to_surface  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--num-points", type=int, default=204800)
    parser.add_argument("--sphere-subdivisions", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy render_rgb frames into the training render layout.",
    )
    parser.add_argument(
        "--write-glb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write posed per-frame GLBs for inspection/reuse.",
    )
    parser.add_argument(
        "--include-parts",
        action="store_true",
        help="Also store per-ball surface samples under the points.npy 'parts' key.",
    )
    parser.add_argument(
        "--include-static-parts",
        action="store_true",
        help="Include interaction-relevant static geometry after dynamic parts: floor, wall, and occluder_box when present.",
    )
    parser.add_argument(
        "--floor-size",
        type=float,
        default=5.0,
        help="Side length for generated floor plane when --include-static-parts is enabled.",
    )
    return parser.parse_args()


def find_sequence_dirs(input_root: Path, sequence_glob: str) -> list[Path]:
    input_root = input_root.expanduser().resolve()
    if (input_root / "physics_metadata.json").is_file():
        return [input_root]
    return sorted(
        path
        for path in input_root.glob(sequence_glob)
        if path.is_dir() and (path / "physics_metadata.json").is_file()
    )


def frame_name(frame_idx: int) -> str:
    return f"frame_{frame_idx:04d}"


def get_radius(metadata: dict, ball_name: str) -> float:
    return float(metadata.get("ball_radii", {}).get(ball_name, metadata.get("ball_radius", 0.25)))


def legacy_object_specs(metadata: dict) -> dict[str, dict]:
    return {
        "ball_0": {
            "type": "sphere",
            "dynamic": True,
            "radius": get_radius(metadata, "ball_0"),
        },
        "ball_1": {
            "type": "sphere",
            "dynamic": True,
            "radius": get_radius(metadata, "ball_1"),
        },
    }


def object_specs(metadata: dict) -> dict[str, dict]:
    objects = metadata.get("objects")
    if isinstance(objects, dict) and objects:
        return objects
    render_objects = metadata.get("render_objects")
    object_properties = metadata.get("object_properties")
    if isinstance(render_objects, list) and isinstance(object_properties, dict):
        radius = float(object_properties.get("ball_radius", metadata.get("ball_radius", 0.25)))
        specs = {}
        for name in render_objects:
            if not isinstance(name, str):
                continue
            if name.startswith("ball_"):
                specs[name] = {"type": "sphere", "dynamic": True, "radius": radius}
            elif name in {"wall", "occluder_box"}:
                half_extents = object_properties.get(f"{name}_half_extents", [0.5, 0.5, 0.5])
                specs[name] = {
                    "type": "box",
                    "dynamic": False,
                    "size": [2.0 * float(value) for value in half_extents],
                }
            elif name == "floor":
                specs[name] = {"type": "plane", "dynamic": False}
        if specs:
            return specs
    return legacy_object_specs(metadata)


def dynamic_object_specs(metadata: dict) -> dict[str, dict]:
    return {
        name: spec
        for name, spec in object_specs(metadata).items()
        if bool(spec.get("dynamic", True))
    }


def interaction_static_object_specs(metadata: dict) -> dict[str, dict]:
    specs = object_specs(metadata)
    render_order = metadata.get("render_objects")
    names = [name for name in render_order if isinstance(name, str)] if isinstance(render_order, list) else list(specs)
    wanted = {"floor", "wall", "occluder_box"}
    return {
        name: specs[name]
        for name in names
        if name in wanted and name in specs and not bool(specs[name].get("dynamic", name.startswith("ball_")))
    }


def transform_matrix(position: list[float], quat_xyzw: list[float]) -> np.ndarray:
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    translation = np.eye(4)
    translation[:3, 3] = np.asarray(position, dtype=np.float64)
    return translation @ rotation


def object_pose(name: str, spec: dict, frame: dict) -> tuple[list[float], list[float]]:
    frame_objects = frame.get("objects")
    frame_pose = frame_objects.get(name) if isinstance(frame_objects, dict) else frame.get(name)
    if isinstance(frame_pose, dict):
        position = frame_pose.get("position", spec.get("position", [0.0, 0.0, 0.0]))
        quaternion = frame_pose.get("quaternion", spec.get("quaternion", [0.0, 0.0, 0.0, 1.0]))
        return list(position), list(quaternion)
    return list(spec.get("position", [0.0, 0.0, 0.0])), list(spec.get("quaternion", [0.0, 0.0, 0.0, 1.0]))


def make_sphere_mesh(spec: dict, position: list[float], quat_xyzw: list[float], subdivisions: int) -> trimesh.Trimesh:
    radius = float(spec.get("radius", 0.25))
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_transform(transform_matrix(position, quat_xyzw))
    return mesh


def make_box_mesh(spec: dict, position: list[float], quat_xyzw: list[float]) -> trimesh.Trimesh:
    size = spec.get("size", spec.get("extents", [1.0, 1.0, 1.0]))
    mesh = trimesh.creation.box(extents=np.asarray(size, dtype=np.float64))
    mesh.apply_transform(transform_matrix(position, quat_xyzw))
    return mesh


def make_floor_mesh(size: float) -> trimesh.Trimesh:
    half = float(size) * 0.5
    return trimesh.Trimesh(
        vertices=[
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )


def make_object_mesh(
    name: str,
    spec: dict,
    frame: dict,
    sphere_subdivisions: int,
    floor_size: float,
) -> trimesh.Trimesh:
    position, quat_xyzw = object_pose(name, spec, frame)
    object_type = str(spec.get("type", "sphere")).lower()
    if object_type in {"sphere", "ball"}:
        return make_sphere_mesh(spec, position, quat_xyzw, sphere_subdivisions)
    if object_type in {"box", "cube", "cuboid"}:
        return make_box_mesh(spec, position, quat_xyzw)
    if object_type in {"plane", "floor"}:
        mesh = make_floor_mesh(floor_size)
        mesh.apply_transform(transform_matrix(position, quat_xyzw))
        return mesh
    raise ValueError(f"Unsupported object type for {name}: {object_type!r}")


def build_frame_meshes(
    metadata: dict,
    frame: dict,
    subdivisions: int,
    include_static_parts: bool,
    floor_size: float,
) -> list[tuple[str, trimesh.Trimesh]]:
    specs = dict(dynamic_object_specs(metadata))
    if include_static_parts:
        specs.update(interaction_static_object_specs(metadata))
    if not specs:
        raise ValueError("Metadata does not define any dynamic objects.")
    return [
        (name, make_object_mesh(name, spec, frame, subdivisions, floor_size))
        for name, spec in specs.items()
    ]


def surface_dict(mesh: trimesh.Trimesh, num_points: int) -> dict[str, np.ndarray]:
    data = mesh_to_surface(mesh, num_pc=num_points, return_dict=True)
    return {
        "surface_points": np.asarray(data["surface_points"], dtype=np.float32),
        "surface_normals": np.asarray(data["surface_normals"], dtype=np.float32),
    }


def write_points(
    output_path: Path,
    object_mesh: trimesh.Trimesh,
    part_meshes: list[trimesh.Trimesh],
    num_points: int,
    include_parts: bool,
) -> None:
    data = {
        "object": surface_dict(object_mesh, num_points),
        "parts": [],
    }
    if include_parts:
        data["parts"] = [surface_dict(part, num_points) for part in part_meshes]
    np.save(output_path, data)


def process_sequence(
    sequence_dir: Path,
    output_root: Path,
    num_points: int,
    sphere_subdivisions: int,
    frame_limit: int | None,
    overwrite: bool,
    copy_rgb: bool,
    write_glb: bool,
    include_parts: bool,
    include_static_parts: bool,
    floor_size: float,
) -> tuple[str, list[dict]]:
    sequence_name = sequence_dir.name
    metadata_path = sequence_dir / "physics_metadata.json"
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    processed_metadata_path = output_root / "metadata" / sequence_name / "physics_metadata.json"
    processed_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not processed_metadata_path.exists():
        shutil.copy2(metadata_path, processed_metadata_path)

    frames = metadata["frames"]
    if frame_limit is not None:
        frames = frames[:frame_limit]

    glb_dir = output_root / "glb" / sequence_name
    preproc_root = output_root / "preprocessed"
    render_dir = output_root / "render" / sequence_name
    if write_glb:
        glb_dir.mkdir(parents=True, exist_ok=True)
    if copy_rgb:
        render_dir.mkdir(parents=True, exist_ok=True)
    preproc_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for frame in frames:
        idx = int(frame["frame"])
        name = frame_name(idx)
        frame_preproc_dir = preproc_root / f"{sequence_name}_{name}"
        points_path = frame_preproc_dir / "points.npy"
        num_parts_path = frame_preproc_dir / "num_parts.json"
        frame_preproc_dir.mkdir(parents=True, exist_ok=True)

        if overwrite or not points_path.exists():
            named_part_meshes = build_frame_meshes(
                metadata,
                frame,
                sphere_subdivisions,
                include_static_parts=include_static_parts,
                floor_size=floor_size,
            )
            part_meshes = [mesh for _, mesh in named_part_meshes]
            object_mesh = trimesh.util.concatenate(part_meshes) if len(part_meshes) > 1 else part_meshes[0].copy()
            write_points(
                output_path=points_path,
                object_mesh=object_mesh,
                part_meshes=part_meshes,
                num_points=num_points,
                include_parts=include_parts,
            )

            with num_parts_path.open("w") as f:
                json.dump(
                    {
                        "num_parts": len(part_meshes),
                        "part_names": [part_name for part_name, _ in named_part_meshes],
                        "mesh_path": str((glb_dir / f"{name}.glb").resolve()) if write_glb else None,
                        "source_metadata": str(processed_metadata_path.resolve()),
                    },
                    f,
                    separators=(",", ":"),
                )

            if write_glb:
                scene = trimesh.Scene()
                for part_name, part_mesh in named_part_meshes:
                    scene.add_geometry(part_mesh, geom_name=part_name)
                scene.export(glb_dir / f"{name}.glb", file_type="glb")

        src_rgb = sequence_dir / "render_rgb" / f"{name}.png"
        dst_rgb = render_dir / f"{name}.png"
        if copy_rgb:
            if not src_rgb.exists():
                raise FileNotFoundError(f"Missing RGB frame: {src_rgb}")
            if overwrite or not dst_rgb.exists():
                shutil.copy2(src_rgb, dst_rgb)
            image_path = dst_rgb
        else:
            image_path = src_rgb

        entries.append(
            {
                "surface_path": str(points_path.resolve()),
                "image_path": str(image_path.resolve()),
                "iou_mean": 0.0,
                "iou_max": 0.0,
            }
        )

    return sequence_name, entries


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    json_output = args.json_output.expanduser().resolve()

    sequence_dirs = find_sequence_dirs(input_root, args.sequence_glob)
    if not sequence_dirs:
        raise FileNotFoundError(f"No synthetic physics sequences found under {input_root}")

    dataset_index = {}
    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        results = [
            process_sequence(
                sequence_dir=sequence_dir,
                output_root=output_root,
                num_points=args.num_points,
                sphere_subdivisions=args.sphere_subdivisions,
                frame_limit=args.frame_limit,
                overwrite=args.overwrite,
                copy_rgb=args.copy_rgb,
                write_glb=args.write_glb,
                include_parts=args.include_parts,
                include_static_parts=args.include_static_parts,
                floor_size=args.floor_size,
            )
            for sequence_dir in tqdm(sequence_dirs, desc="Preprocessing synthetic physics sequences")
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    process_sequence,
                    sequence_dir,
                    output_root,
                    args.num_points,
                    args.sphere_subdivisions,
                    args.frame_limit,
                    args.overwrite,
                    args.copy_rgb,
                    args.write_glb,
                    args.include_parts,
                    args.include_static_parts,
                    args.floor_size,
                )
                for sequence_dir in sequence_dirs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing synthetic physics sequences"):
                results.append(future.result())

    for sequence_name, entries in results:
        if entries:
            dataset_index[sequence_name] = entries

    json_output.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w") as f:
        json.dump(dict(sorted(dataset_index.items())), f, indent=2)

    total_frames = sum(len(entries) for entries in dataset_index.values())
    print(f"Wrote {json_output} with {len(dataset_index)} sequences and {total_frames} frames.")
    print(f"Processed data root: {output_root}")


if __name__ == "__main__":
    main()

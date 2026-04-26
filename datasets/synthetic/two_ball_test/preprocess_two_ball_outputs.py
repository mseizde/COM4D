#!/usr/bin/env python3

"""Convert two-ball pipeline outputs into COM4D 4D training data.

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


def make_ball_mesh(radius: float, position: list[float], quat_xyzw: list[float], subdivisions: int) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    translation = np.eye(4)
    translation[:3, 3] = np.asarray(position, dtype=np.float64)
    mesh.apply_transform(translation @ rotation)
    return mesh


def build_frame_meshes(metadata: dict, frame: dict, subdivisions: int) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    ball_0 = make_ball_mesh(
        radius=get_radius(metadata, "ball_0"),
        position=frame["ball_0"]["position"],
        quat_xyzw=frame["ball_0"]["quaternion"],
        subdivisions=subdivisions,
    )
    ball_1 = make_ball_mesh(
        radius=get_radius(metadata, "ball_1"),
        position=frame["ball_1"]["position"],
        quat_xyzw=frame["ball_1"]["quaternion"],
        subdivisions=subdivisions,
    )
    return ball_0, ball_1


def surface_dict(mesh: trimesh.Trimesh, num_points: int) -> dict[str, np.ndarray]:
    data = mesh_to_surface(mesh, num_pc=num_points, return_dict=True)
    return {
        "surface_points": np.asarray(data["surface_points"], dtype=np.float32),
        "surface_normals": np.asarray(data["surface_normals"], dtype=np.float32),
    }


def write_points(
    output_path: Path,
    object_mesh: trimesh.Trimesh,
    part_meshes: tuple[trimesh.Trimesh, trimesh.Trimesh],
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
) -> tuple[str, list[dict]]:
    sequence_name = sequence_dir.name
    metadata_path = sequence_dir / "physics_metadata.json"
    with metadata_path.open("r") as f:
        metadata = json.load(f)

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
            ball_0_mesh, ball_1_mesh = build_frame_meshes(metadata, frame, sphere_subdivisions)
            object_mesh = trimesh.util.concatenate([ball_0_mesh, ball_1_mesh])
            write_points(
                output_path=points_path,
                object_mesh=object_mesh,
                part_meshes=(ball_0_mesh, ball_1_mesh),
                num_points=num_points,
                include_parts=include_parts,
            )

            with num_parts_path.open("w") as f:
                json.dump(
                    {
                        "num_parts": 2,
                        "mesh_path": str((glb_dir / f"{name}.glb").resolve()) if write_glb else None,
                        "source_metadata": str(metadata_path.resolve()),
                    },
                    f,
                    separators=(",", ":"),
                )

            if write_glb:
                scene = trimesh.Scene()
                scene.add_geometry(ball_0_mesh, geom_name="ball_0")
                scene.add_geometry(ball_1_mesh, geom_name="ball_1")
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
        raise FileNotFoundError(f"No two-ball sequences found under {input_root}")

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
            )
            for sequence_dir in tqdm(sequence_dirs, desc="Preprocessing two-ball sequences")
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
                )
                for sequence_dir in sequence_dirs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing two-ball sequences"):
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

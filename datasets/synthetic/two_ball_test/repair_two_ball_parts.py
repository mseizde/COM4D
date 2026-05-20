#!/usr/bin/env python3
"""Add explicit per-ball `parts` to an existing processed two-ball dataset.

This is for datasets that were already rendered/preprocessed without
`--include-parts`. It preserves the existing `object` samples and RGB renders,
then loads each per-frame GLB scene and writes `parts=[ball_0, ball_1]` into the
existing `points.npy`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re

import numpy as np
import trimesh
from tqdm.auto import tqdm

from preprocess_two_ball_outputs import surface_dict


FRAME_RE = re.compile(r"^(?P<sample>.+)_frame_(?P<frame>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
        help="Existing processed two-ball root containing preprocessed/ and glb/.",
    )
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite parts even when points.npy already has non-empty parts.",
    )
    return parser.parse_args()


def frame_dir_to_glb(processed_root: Path, frame_dir: Path) -> Path:
    match = FRAME_RE.match(frame_dir.name)
    if match is None:
        raise ValueError(f"Unexpected frame directory name: {frame_dir.name}")
    sample = match.group("sample")
    frame = match.group("frame")
    return processed_root / "glb" / sample / f"frame_{frame}.glb"


def load_ball_parts(glb_path: Path) -> list[trimesh.Trimesh]:
    scene = trimesh.load(glb_path, process=False)
    if not isinstance(scene, trimesh.Scene):
        raise ValueError(f"Expected GLB scene with ball_0/ball_1 geometries: {glb_path}")
    missing = [name for name in ("ball_0", "ball_1") if name not in scene.geometry]
    if missing:
        raise ValueError(f"Missing {missing} in {glb_path}; found {list(scene.geometry.keys())}")
    return [scene.geometry["ball_0"], scene.geometry["ball_1"]]


def repair_one(task: tuple[Path, Path, int, bool]) -> tuple[str, str]:
    processed_root, points_path, num_points, overwrite = task
    data = np.load(points_path, allow_pickle=True).item()
    if not isinstance(data, dict) or "object" not in data:
        raise ValueError(f"Unrecognized points.npy schema: {points_path}")
    if data.get("parts") and not overwrite:
        return str(points_path), "skipped"

    glb_path = frame_dir_to_glb(processed_root, points_path.parent)
    part_meshes = load_ball_parts(glb_path)
    data["parts"] = [surface_dict(mesh, num_points) for mesh in part_meshes]
    np.save(points_path, data)
    return str(points_path), "repaired"


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.expanduser().resolve()
    preprocessed_root = processed_root / "preprocessed"
    if not preprocessed_root.is_dir():
        raise FileNotFoundError(f"Missing preprocessed directory: {preprocessed_root}")

    points_paths = sorted(preprocessed_root.glob("*/points.npy"))
    if not points_paths:
        raise FileNotFoundError(f"No points.npy files found under {preprocessed_root}")

    tasks = [(processed_root, path, int(args.num_points), bool(args.overwrite)) for path in points_paths]
    counts = {"repaired": 0, "skipped": 0}
    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        iterator = (repair_one(task) for task in tasks)
        for _, status in tqdm(iterator, total=len(tasks), desc="Repairing parts"):
            counts[status] = counts.get(status, 0) + 1
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(repair_one, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Repairing parts"):
                _, status = future.result()
                counts[status] = counts.get(status, 0) + 1

    print(f"Processed root: {processed_root}")
    print(f"Repaired: {counts.get('repaired', 0)}")
    print(f"Skipped: {counts.get('skipped', 0)}")


if __name__ == "__main__":
    main()

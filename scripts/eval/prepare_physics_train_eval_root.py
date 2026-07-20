#!/usr/bin/env python3

"""Build an eval-style dataset root from processed physics training samples."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


DEFAULT_MANIFEST = Path("dataset_json/physics_train_100_static.json")
DEFAULT_PROCESSED_ROOT = Path("/mnt/mocap_b/work/com4d/datasets/processed/physics/processed_train_100_static")
DEFAULT_GLB_ROOT = Path("/mnt/mocap_b/work/com4d/datasets/processed/physics/processed_train_100/glb")
DEFAULT_OUTPUT_ROOT = Path("/data/mseizde/com4d/outputs/eval_inputs/physics_train_100_static_subset")
DEFAULT_CASES = ("two_ball_collision", "rolling_occluder", "wall_impact", "ball_drop")


SCENARIO_CAMERA_DEFAULTS = {
    "two_ball_collision": {"target": [0.0, 0.0, 0.32], "distance": 3.0, "height": 0.85, "azimuth": 0.0, "focal_length": 38.0},
    "ball_drop": {"target": [0.15, 0.0, 0.75], "distance": 3.2, "height": 0.9, "azimuth": 8.0, "focal_length": 35.0},
    "rolling_occluder": {"target": [0.0, -0.35, 0.45], "distance": 3.5, "height": 0.8, "azimuth": 0.0, "focal_length": 38.0},
    "wall_impact": {"target": [0.45, 0.0, 0.45], "distance": 3.5, "height": 0.9, "azimuth": -10.0, "focal_length": 38.0},
}


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm <= 0.0:
        raise ValueError(f"Cannot normalize zero vector: {vec}")
    return [value / norm for value in vec]


def cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def synthetic_camera_for_scenario(scenario: str, resolution: int = 518) -> dict[str, object]:
    defaults = SCENARIO_CAMERA_DEFAULTS.get(scenario, SCENARIO_CAMERA_DEFAULTS["two_ball_collision"])
    target = [float(value) for value in defaults["target"]]
    distance = float(defaults["distance"])
    height = float(defaults["height"])
    azimuth = math.radians(float(defaults["azimuth"]))
    lens = float(defaults["focal_length"])
    location = [
        target[0] + distance * math.sin(azimuth),
        target[1] - distance * math.cos(azimuth),
        target[2] + height,
    ]

    forward = normalize([target[index] - location[index] for index in range(3)])
    camera_z = [-value for value in forward]
    world_up = [0.0, 0.0, 1.0]
    camera_x = normalize(cross(forward, world_up))
    camera_y = cross(camera_z, camera_x)
    camera_to_world = [
        [camera_x[0], camera_y[0], camera_z[0], location[0]],
        [camera_x[1], camera_y[1], camera_z[1], location[1]],
        [camera_x[2], camera_y[2], camera_z[2], location[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]

    sensor_width = 36.0
    sensor_height = 24.0
    fx = lens * resolution / sensor_width
    fy = fx
    cx = resolution * 0.5
    cy = resolution * 0.5
    return {
        "static": True,
        "coordinate_system": "blender_world",
        "camera_convention": "opencv_values_with_blender_camera_to_world_minus_z_forward_y_up",
        "location": location,
        "target": target,
        "camera_to_world": camera_to_world,
        "lens_mm": lens,
        "sensor_width_mm": sensor_width,
        "sensor_height_mm": sensor_height,
        "sensor_fit": "AUTO",
        "shift_x": 0.0,
        "shift_y": 0.0,
        "clip_start": 0.1,
        "clip_end": 1000.0,
        "resolution": [resolution, resolution],
        "intrinsics": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "recovery": {"profile": "prepare_physics_train_eval_root", **defaults},
    }


def write_metadata_with_camera(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    with src.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    camera = metadata.get("camera")
    if not isinstance(camera, dict) or not all(key in camera for key in ("camera_to_world", "intrinsics", "resolution")):
        metadata["camera"] = synthetic_camera_for_scenario(str(metadata.get("scenario", "two_ball_collision")))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--glb-root", type=Path, default=DEFAULT_GLB_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--case",
        action="append",
        choices=DEFAULT_CASES,
        default=None,
        help="Training case prefix to include. Repeatable. Defaults to all four cases.",
    )
    parser.add_argument(
        "--samples-per-case",
        type=int,
        default=2,
        help="Number of manifest samples to include per selected case.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Exact sample id to include. If provided, bypasses --case/--samples-per-case.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output root first.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object manifest: {path}")
    return data


def case_name(sample: str) -> str:
    return sample.rsplit("_", 1)[0]


def select_samples(manifest: dict[str, object], args: argparse.Namespace) -> list[str]:
    manifest_samples = list(manifest)
    if args.sample:
        missing = [sample for sample in args.sample if sample not in manifest]
        if missing:
            raise ValueError(f"Samples are not in manifest: {', '.join(missing)}")
        return list(dict.fromkeys(args.sample))

    selected_cases = args.case or list(DEFAULT_CASES)
    samples: list[str] = []
    for selected_case in selected_cases:
        matches = [sample for sample in manifest_samples if case_name(sample) == selected_case]
        samples.extend(matches[: args.samples_per_case])
    return samples


def reset_output_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    (path / "gt_raw").mkdir(parents=True, exist_ok=True)
    (path / "inference_input").mkdir(parents=True, exist_ok=True)


def symlink_path(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def add_sample(sample: str, processed_root: Path, glb_root: Path, output_root: Path) -> None:
    raw_dir = output_root / "gt_raw" / sample
    raw_dir.mkdir(parents=True, exist_ok=True)

    symlink_path(processed_root / "render" / sample, raw_dir / "render_rgb")
    symlink_path(processed_root / "masks_visible" / sample, raw_dir / "masks")
    write_metadata_with_camera(processed_root / "metadata" / sample / "physics_metadata.json", raw_dir / "physics_metadata.json")
    glb_sample_dir = glb_root / sample
    if not glb_sample_dir.is_dir():
        raise FileNotFoundError(glb_sample_dir)
    frame_glbs = sorted(glb_sample_dir.glob("frame_*.glb"))
    if not frame_glbs:
        raise FileNotFoundError(f"No frame_*.glb files found in {glb_sample_dir}")
    for frame_glb in frame_glbs:
        symlink_path(frame_glb, raw_dir / frame_glb.name)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    samples = select_samples(manifest, args)
    if not samples:
        raise RuntimeError("No samples selected.")

    reset_output_root(args.output_root, args.overwrite)
    for sample in samples:
        add_sample(sample, args.processed_root, args.glb_root, args.output_root)

    print(f"Wrote eval-style dataset root: {args.output_root}")
    print("Samples:")
    for sample in samples:
        print(f"  {sample}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Generate varied two-ball comparison cases and prepare COM4D inference inputs."""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
RUN_ONE = REPO_ROOT / "datasets" / "synthetic" / "two_ball_test" / "run_two_ball_pipeline.py"
PREPARE_INPUT = REPO_ROOT / "scripts" / "eval" / "prepare_two_ball_inference_input.py"
DEFAULT_ROOT = Path("/mnt/mocap_b/work/com4d/datasets/synthetic/two_ball_compare")
DEFAULT_BLENDER = PROJECT_ROOT / "tools" / "blender-3.6.5-linux-x64" / "blender"


def parse_range(raw: list[float], name: str) -> tuple[float, float]:
    if len(raw) != 2:
        raise ValueError(f"{name} must have exactly two values")
    low, high = float(raw[0]), float(raw[1])
    return (low, high) if low <= high else (high, low)


def sample_range(rng: random.Random, bounds: tuple[float, float]) -> float:
    return rng.uniform(bounds[0], bounds[1])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--start-index", type=int, default=1)
    ap.add_argument("--num-samples", type=int, default=9)
    ap.add_argument("--name-template", default="two_ball_eval_{index:03d}")
    ap.add_argument("--seed", type=int, default=124)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    ap.add_argument("--device", choices=("AUTO", "CPU", "GPU"), default="CPU")
    ap.add_argument("--blender-threads", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=518)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--input-mode", choices=("symlink", "copy"), default="copy")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-save-depth", action="store_true")
    ap.add_argument("--no-save-normals", action="store_true")

    ap.add_argument("--radius-range", type=float, nargs=2, default=[0.18, 0.38])
    ap.add_argument("--mass-range", type=float, nargs=2, default=[0.6, 1.8])
    ap.add_argument("--height-range", type=float, nargs=2, default=[0.0, 0.24])
    ap.add_argument("--x-extent-range", type=float, nargs=2, default=[0.65, 0.95])
    ap.add_argument("--y-offset-range", type=float, nargs=2, default=[-0.18, 0.18])
    ap.add_argument("--speed-range", type=float, nargs=2, default=[1.6, 2.8])
    ap.add_argument("--lateral-speed-range", type=float, nargs=2, default=[-0.35, 0.35])
    ap.add_argument("--vertical-speed-range", type=float, nargs=2, default=[0.0, 0.35])
    ap.add_argument("--angular-speed-range", type=float, nargs=2, default=[-6.0, 6.0])
    ap.add_argument("--restitution-range", type=float, nargs=2, default=[0.65, 0.98])
    ap.add_argument("--friction-range", type=float, nargs=2, default=[0.05, 0.45])
    ap.add_argument("--looks", default="red,blue,green,orange,white,black,basketball,football")
    ap.add_argument("--floor-looks", default="gray,light_gray,blue_gray")
    ap.add_argument("--camera-distance-range", type=float, nargs=2, default=[4.0, 6.0])
    ap.add_argument("--camera-height-range", type=float, nargs=2, default=[1.5, 3.2])
    ap.add_argument("--camera-azimuth-range", type=float, nargs=2, default=[0.0, 360.0])
    ap.add_argument("--light-energy-range", type=float, nargs=2, default=[350.0, 750.0])
    ap.add_argument("--light-size-range", type=float, nargs=2, default=[2.0, 6.0])
    ap.add_argument("--light-distance-range", type=float, nargs=2, default=[3.0, 5.5])
    ap.add_argument("--light-height-range", type=float, nargs=2, default=[3.0, 6.0])
    return ap.parse_args()


def extend_vec(cmd: list[str], flag: str, values: list[float]) -> None:
    cmd.append(flag)
    cmd.extend(str(value) for value in values)


def sample_params(args: argparse.Namespace, sample_index: int) -> dict[str, object]:
    rng = random.Random(args.seed + sample_index)
    radius_range = parse_range(args.radius_range, "--radius-range")
    mass_range = parse_range(args.mass_range, "--mass-range")
    height_range = parse_range(args.height_range, "--height-range")
    x_extent_range = parse_range(args.x_extent_range, "--x-extent-range")
    y_offset_range = parse_range(args.y_offset_range, "--y-offset-range")
    speed_range = parse_range(args.speed_range, "--speed-range")
    lateral_speed_range = parse_range(args.lateral_speed_range, "--lateral-speed-range")
    vertical_speed_range = parse_range(args.vertical_speed_range, "--vertical-speed-range")
    angular_speed_range = parse_range(args.angular_speed_range, "--angular-speed-range")
    restitution_range = parse_range(args.restitution_range, "--restitution-range")
    friction_range = parse_range(args.friction_range, "--friction-range")
    camera_distance_range = parse_range(args.camera_distance_range, "--camera-distance-range")
    camera_height_range = parse_range(args.camera_height_range, "--camera-height-range")
    camera_azimuth_range = parse_range(args.camera_azimuth_range, "--camera-azimuth-range")
    light_energy_range = parse_range(args.light_energy_range, "--light-energy-range")
    light_size_range = parse_range(args.light_size_range, "--light-size-range")
    light_distance_range = parse_range(args.light_distance_range, "--light-distance-range")
    light_height_range = parse_range(args.light_height_range, "--light-height-range")
    looks = [look.strip() for look in args.looks.split(",") if look.strip()]
    floor_looks = [look.strip() for look in args.floor_looks.split(",") if look.strip()]

    x0 = sample_range(rng, x_extent_range)
    x1 = sample_range(rng, x_extent_range)
    y0 = sample_range(rng, y_offset_range)
    y1 = sample_range(rng, y_offset_range)
    speed0 = sample_range(rng, speed_range)
    speed1 = sample_range(rng, speed_range)
    lateral0 = sample_range(rng, lateral_speed_range)
    lateral1 = sample_range(rng, lateral_speed_range)
    vertical0 = sample_range(rng, vertical_speed_range)
    vertical1 = sample_range(rng, vertical_speed_range)

    return {
        "seed": args.seed + sample_index,
        "ball_0_radius": sample_range(rng, radius_range),
        "ball_1_radius": sample_range(rng, radius_range),
        "ball_0_mass": sample_range(rng, mass_range),
        "ball_1_mass": sample_range(rng, mass_range),
        "ball_0_xy": [-x0, y0],
        "ball_1_xy": [x1, y1],
        "ball_0_height": sample_range(rng, height_range),
        "ball_1_height": sample_range(rng, height_range),
        "ball_0_velocity": [speed0, lateral0, vertical0],
        "ball_1_velocity": [-speed1, lateral1, vertical1],
        "ball_0_angular_velocity": [sample_range(rng, angular_speed_range) for _ in range(3)],
        "ball_1_angular_velocity": [sample_range(rng, angular_speed_range) for _ in range(3)],
        "restitution": sample_range(rng, restitution_range),
        "lateral_friction": sample_range(rng, friction_range),
        "camera_distance": sample_range(rng, camera_distance_range),
        "camera_height": sample_range(rng, camera_height_range),
        "camera_azimuth": sample_range(rng, camera_azimuth_range),
        "light_energy": sample_range(rng, light_energy_range),
        "light_size": sample_range(rng, light_size_range),
        "light_distance": sample_range(rng, light_distance_range),
        "light_height": sample_range(rng, light_height_range),
        "ball_0_look": rng.choice(looks),
        "ball_1_look": rng.choice(looks),
        "floor_look": rng.choice(floor_looks),
    }


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], check=True)


def build_pipeline_cmd(args: argparse.Namespace, raw_dir: Path, params: dict[str, object]) -> list[str]:
    cmd = [
        sys.executable,
        RUN_ONE,
        "--output-dir",
        raw_dir,
        "--blender-bin",
        args.blender_bin,
        "--num-frames",
        args.num_frames,
        "--fps",
        args.fps,
        "--seed",
        params["seed"],
        "--ball-0-radius",
        params["ball_0_radius"],
        "--ball-1-radius",
        params["ball_1_radius"],
        "--ball-0-mass",
        params["ball_0_mass"],
        "--ball-1-mass",
        params["ball_1_mass"],
        "--ball-0-height",
        params["ball_0_height"],
        "--ball-1-height",
        params["ball_1_height"],
        "--restitution",
        params["restitution"],
        "--lateral-friction",
        params["lateral_friction"],
        "--random-view",
        "--view-seed",
        params["seed"],
        "--camera-distance",
        params["camera_distance"],
        "--camera-height",
        params["camera_height"],
        "--camera-azimuth",
        params["camera_azimuth"],
        "--random-light",
        "--light-seed",
        params["seed"],
        "--light-energy",
        params["light_energy"],
        "--light-size",
        params["light_size"],
        "--light-distance",
        params["light_distance"],
        "--light-height",
        params["light_height"],
        "--ball-0-look",
        params["ball_0_look"],
        "--ball-1-look",
        params["ball_1_look"],
        "--floor-look",
        params["floor_look"],
        "--resolution",
        args.resolution,
        "--samples",
        args.samples,
        "--device",
        args.device,
        "--blender-threads",
        args.blender_threads,
    ]
    if not args.no_save_depth:
        cmd.append("--save-depth")
    if not args.no_save_normals:
        cmd.append("--save-normals")
    extend_vec(cmd, "--ball-0-xy", params["ball_0_xy"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-1-xy", params["ball_1_xy"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-0-velocity", params["ball_0_velocity"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-1-velocity", params["ball_1_velocity"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-0-angular-velocity", params["ball_0_angular_velocity"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-1-angular-velocity", params["ball_1_angular_velocity"])  # type: ignore[arg-type]
    return cmd


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    raw_root = root / "gt_raw"
    input_root = root / "inference_input"
    raw_root.mkdir(parents=True, exist_ok=True)
    input_root.mkdir(parents=True, exist_ok=True)

    for offset in range(args.num_samples):
        sample_index = args.start_index + offset
        sample_name = args.name_template.format(index=sample_index)
        raw_dir = raw_root / sample_name
        input_dir = input_root / sample_name
        if args.overwrite:
            for path in (raw_dir, input_dir):
                if path.exists() and not args.dry_run:
                    shutil.rmtree(path)
        params = sample_params(args, sample_index)
        run(build_pipeline_cmd(args, raw_dir, params), dry_run=args.dry_run)
        run(
            [
                sys.executable,
                PREPARE_INPUT,
                "--raw-dir",
                raw_dir,
                "--output-dir",
                input_dir,
                "--mode",
                args.input_mode,
                "--overwrite",
            ],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

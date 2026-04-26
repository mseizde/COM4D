#!/usr/bin/env python3

"""Generate a randomized two-ball physics dataset end to end.

This runs, per sample:
  1. PyBullet physics metadata
  2. Blender RGB rendering only

Then it converts all samples into COM4D training format:
  <processed-root>/glb/<sample>/frame_0000.glb
  <processed-root>/preprocessed/<sample>_frame_0000/points.npy
  <processed-root>/render/<sample>/frame_0000.png
  <json-output>

Example:
  micromamba run -n com4d python datasets/synthetic/two_ball_test/generate_two_ball_dataset.py \
    --num-samples 1000 \
    --num-frames 32 \
    --workers 4 \
    --device GPU \
    --gpu-ids 0,1,2,3 \
    --num-points 8192 \
    --overwrite
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/mseizde/com4d/datasets/processed")
DEFAULT_RAW_ROOT = DATA_ROOT / "physics" / "two_ball_raw"
DEFAULT_PROCESSED_ROOT = DATA_ROOT / "physics" / "two_ball"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "dataset_json" / "physics.json"
RUN_ONE = SCRIPT_DIR / "run_two_ball_pipeline.py"
PREPROCESS = SCRIPT_DIR / "preprocess_two_ball_outputs.py"


def parse_range(raw: list[float], name: str) -> tuple[float, float]:
    if len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    low, high = float(raw[0]), float(raw[1])
    if high < low:
        low, high = high, low
    return low, high


def sample_range(rng: random.Random, bounds: tuple[float, float]) -> float:
    return rng.uniform(bounds[0], bounds[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preprocess-workers", type=int, default=None)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--retries", type=int, default=0, help="Retry failed sample generation commands.")

    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--device", choices=("AUTO", "CPU", "GPU"), default="CPU")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU ids to cycle across workers.")
    parser.add_argument(
        "--blender-threads",
        type=int,
        default=1,
        help="Threads per Blender render process. Keep low when using multiple workers on CPU.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--sphere-subdivisions", type=int, default=4)
    parser.add_argument(
        "--write-masks",
        action="store_true",
        help="Keep Blender object-mask PNG outputs in the raw sample folders.",
    )

    parser.add_argument("--radius-range", type=float, nargs=2, default=[0.18, 0.38])
    parser.add_argument("--mass-range", type=float, nargs=2, default=[0.6, 1.8])
    parser.add_argument("--height-range", type=float, nargs=2, default=[0.0, 0.12])
    parser.add_argument("--x-extent-range", type=float, nargs=2, default=[0.65, 0.95])
    parser.add_argument("--y-offset-range", type=float, nargs=2, default=[-0.18, 0.18])
    parser.add_argument("--speed-range", type=float, nargs=2, default=[1.6, 2.8])
    parser.add_argument("--lateral-speed-range", type=float, nargs=2, default=[-0.35, 0.35])
    parser.add_argument("--vertical-speed-range", type=float, nargs=2, default=[0.0, 0.35])
    parser.add_argument("--angular-speed-range", type=float, nargs=2, default=[-6.0, 6.0])
    parser.add_argument("--restitution-range", type=float, nargs=2, default=[0.65, 0.98])
    parser.add_argument("--friction-range", type=float, nargs=2, default=[0.05, 0.45])

    parser.add_argument("--camera-distance-range", type=float, nargs=2, default=[4.0, 6.0])
    parser.add_argument("--camera-height-range", type=float, nargs=2, default=[1.5, 3.0])
    parser.add_argument("--camera-azimuth-range", type=float, nargs=2, default=[0.0, 360.0])
    parser.add_argument("--light-energy-range", type=float, nargs=2, default=[350.0, 750.0])
    parser.add_argument("--light-size-range", type=float, nargs=2, default=[2.0, 6.0])
    parser.add_argument("--light-distance-range", type=float, nargs=2, default=[3.0, 5.5])
    parser.add_argument("--light-height-range", type=float, nargs=2, default=[3.0, 6.0])
    parser.add_argument(
        "--looks",
        default="red,blue,green,orange,white,black,basketball,football",
        help="Comma-separated material look choices to sample.",
    )
    parser.add_argument(
        "--floor-looks",
        default="gray,light_gray,dark_gray,blue_gray,green_gray",
        help="Comma-separated floor material look choices to sample.",
    )
    return parser.parse_args()


def format_command(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def run_command(
    cmd: list[str],
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    if log_path is None:
        subprocess.run(cmd, check=True, env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] Running:\n")
        log.write(f"{format_command(cmd)}\n\n")
        log.flush()
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] Exit code: {result.returncode}\n")
        log.flush()

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def sample_params(args: argparse.Namespace, sample_idx: int) -> dict[str, object]:
    rng = random.Random(args.seed + sample_idx)
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
        "seed": args.seed + sample_idx,
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


def extend_vec(cmd: list[str], flag: str, values: list[float]) -> None:
    cmd.append(flag)
    cmd.extend(str(value) for value in values)


def build_sample_command(args: argparse.Namespace, sample_idx: int, sample_dir: Path) -> list[str]:
    params = sample_params(args, sample_idx)
    cmd = [
        sys.executable,
        str(RUN_ONE),
        "--output-dir",
        str(sample_dir),
        "--blender-bin",
        args.blender_bin,
        "--num-frames",
        str(args.num_frames),
        "--fps",
        str(args.fps),
        "--seed",
        str(params["seed"]),
        "--ball-0-radius",
        str(params["ball_0_radius"]),
        "--ball-1-radius",
        str(params["ball_1_radius"]),
        "--ball-0-mass",
        str(params["ball_0_mass"]),
        "--ball-1-mass",
        str(params["ball_1_mass"]),
        "--ball-0-height",
        str(params["ball_0_height"]),
        "--ball-1-height",
        str(params["ball_1_height"]),
        "--restitution",
        str(params["restitution"]),
        "--lateral-friction",
        str(params["lateral_friction"]),
        "--resolution",
        str(args.resolution),
        "--samples",
        str(args.samples),
        "--device",
        args.device,
        "--blender-threads",
        str(args.blender_threads),
        "--random-view",
        "--view-seed",
        str(params["seed"]),
        "--camera-distance",
        str(params["camera_distance"]),
        "--camera-height",
        str(params["camera_height"]),
        "--camera-azimuth",
        str(params["camera_azimuth"]),
        "--random-light",
        "--light-seed",
        str(params["seed"]),
        "--light-energy",
        str(params["light_energy"]),
        "--light-size",
        str(params["light_size"]),
        "--light-distance",
        str(params["light_distance"]),
        "--light-height",
        str(params["light_height"]),
        "--ball-0-look",
        str(params["ball_0_look"]),
        "--ball-1-look",
        str(params["ball_1_look"]),
        "--floor-look",
        str(params["floor_look"]),
        "--skip-transforms",
        "--skip-canonical-meshes",
    ]
    if not args.write_masks:
        cmd.append("--skip-masks")
    extend_vec(cmd, "--ball-0-xy", params["ball_0_xy"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-1-xy", params["ball_1_xy"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-0-velocity", params["ball_0_velocity"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-1-velocity", params["ball_1_velocity"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-0-angular-velocity", params["ball_0_angular_velocity"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-1-angular-velocity", params["ball_1_angular_velocity"])  # type: ignore[arg-type]
    return cmd


def run_sample(args: argparse.Namespace, sample_number: int, gpu_ids: list[str]) -> str:
    sample_idx = args.start_index + sample_number
    sample_name = f"two_ball_{sample_idx:06d}"
    sample_dir = args.raw_root / sample_name
    if sample_dir.exists() and args.overwrite:
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[sample_number % len(gpu_ids)]

    env["PYTHONUNBUFFERED"] = "1"
    if args.blender_threads > 0:
        thread_count = str(args.blender_threads)
        env["OMP_NUM_THREADS"] = thread_count
        env["OPENBLAS_NUM_THREADS"] = thread_count
        env["MKL_NUM_THREADS"] = thread_count
        env["NUMEXPR_NUM_THREADS"] = thread_count
    env["TMPDIR"] = str(sample_dir / "tmp")
    env["BLENDER_USER_CONFIG"] = str(sample_dir / "blender_user_config")
    env["BLENDER_USER_CACHE"] = str(sample_dir / "blender_user_cache")
    env["BLENDER_USER_DATAFILES"] = str(sample_dir / "blender_user_datafiles")
    for env_dir in (
        env["TMPDIR"],
        env["BLENDER_USER_CONFIG"],
        env["BLENDER_USER_CACHE"],
        env["BLENDER_USER_DATAFILES"],
    ):
        Path(env_dir).mkdir(parents=True, exist_ok=True)

    cmd = build_sample_command(args, sample_idx, sample_dir)
    log_path = sample_dir / "pipeline.log"
    max_attempts = max(1, int(args.retries) + 1)
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\nRetrying sample after failure: attempt {attempt}/{max_attempts}\n")
            run_command(cmd, env=env, log_path=log_path)
            return sample_name
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < max_attempts:
                continue

    raise RuntimeError(
        f"{sample_name} failed after {max_attempts} attempt(s). "
        f"See log: {log_path}"
    ) from last_error


def run_preprocess(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(PREPROCESS),
        "--input-root",
        str(args.raw_root),
        "--output-root",
        str(args.processed_root),
        "--json-output",
        str(args.json_output),
        "--num-points",
        str(args.num_points),
        "--sphere-subdivisions",
        str(args.sphere_subdivisions),
        "--workers",
        str(args.preprocess_workers if args.preprocess_workers is not None else args.workers),
        "--overwrite",
    ]
    run_command(cmd)


def main() -> None:
    args = parse_args()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.processed_root = args.processed_root.expanduser().resolve()
    args.json_output = args.json_output.expanduser().resolve()

    if args.overwrite:
        if args.raw_root.exists():
            shutil.rmtree(args.raw_root)
        if args.processed_root.exists():
            shutil.rmtree(args.processed_root)

    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.processed_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = []
    if args.gpu_ids:
        gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]

    worker_count = min(max(1, int(args.workers)), max(1, int(args.num_samples)))
    if worker_count == 1:
        for sample_number in tqdm(range(args.num_samples), desc="Generating two-ball samples"):
            run_sample(args, sample_number, gpu_ids)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(run_sample, args, sample_number, gpu_ids)
                for sample_number in range(args.num_samples)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating two-ball samples"):
                future.result()

    run_preprocess(args)

    if not args.keep_raw:
        shutil.rmtree(args.raw_root)

    print(f"Processed dataset: {args.processed_root}")
    print(f"Dataset JSON: {args.json_output}")


if __name__ == "__main__":
    main()

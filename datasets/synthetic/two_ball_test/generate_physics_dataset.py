#!/usr/bin/env python3

"""Generate a randomized synthetic physics dataset end to end.

This runs, per sample:
  1. PyBullet physics metadata
  2. Blender RGB rendering only

Then it converts all samples into COM4D training format:
  <processed-root>/glb/<sample>/frame_0000.glb
  <processed-root>/preprocessed/<sample>_frame_0000/points.npy
  <processed-root>/render/<sample>/frame_0000.png
  <json-output>

The generated points.npy files include explicit per-ball entries under the
`parts` key. That is required for COM4D physics batches that use true
spatio-temporal mixing over [frame x object] instances.

Example:
  micromamba run -n com4d python datasets/synthetic/two_ball_test/generate_physics_dataset.py \
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
RUN_ONE = SCRIPT_DIR / "run_physics_pipeline.py"
PREPROCESS = SCRIPT_DIR / "preprocess_physics_outputs.py"


SCENARIO_SAMPLING_DEFAULTS = {
    "two_ball_collision": {
        "radius_range": [0.18, 0.38],
        "mass_range": [0.6, 1.8],
        "height_range": [0.0, 0.12],
        "x_extent_range": [0.65, 0.95],
        "y_offset_range": [-0.18, 0.18],
        "speed_range": [1.6, 2.8],
        "lateral_speed_range": [-0.35, 0.35],
        "vertical_speed_range": [0.0, 0.35],
        "restitution_range": [0.65, 0.98],
        "friction_range": [0.05, 0.45],
        "camera_target": [0.0, 0.0, 0.35],
        "camera_focal_length": 35.0,
        "camera_distance_range": [3.4, 5.4],
        "camera_height_range": [0.8, 4.8],
        "camera_azimuth_range": [0.0, 360.0],
    },
    "ball_drop": {
        "radius_range": [0.20, 0.32],
        "mass_range": [0.7, 1.6],
        "height_range": [0.0, 0.08],
        "drop_height_range": [1.25, 1.85],
        "x_extent_range": [0.0, 0.45],
        "y_offset_range": [-0.10, 0.10],
        "speed_range": [0.25, 1.6],
        "lateral_speed_range": [-0.08, 0.08],
        "vertical_speed_range": [-0.03, 0.08],
        "restitution_range": [0.65, 0.95],
        "friction_range": [0.03, 0.22],
        "camera_target": [0.0, 0.0, 0.8],
        "camera_focal_length": 28.0,
        "camera_distance_range": [3.8, 5.8],
        "camera_height_range": [0.8, 5.0],
        "camera_azimuth_range": [0.0, 360.0],
    },
    "rolling_occluder": {
        "radius_range": [0.20, 0.32],
        "mass_range": [0.7, 1.6],
        "height_range": [0.0, 0.05],
        "x_extent_range": [2.2, 2.8],
        "y_offset_range": [-0.10, 0.10],
        "speed_range": [3.4, 4.8],
        "lateral_speed_range": [-0.18, 0.18],
        "vertical_speed_range": [0.0, 0.08],
        "restitution_range": [0.25, 0.65],
        "friction_range": [0.35, 0.85],
        "occluder_y_range": [-1.35, -1.05],
        "occluder_size_range": [0.40, 0.58],
        "camera_target": [0.0, -0.4, 0.5],
        "camera_focal_length": 35.0,
        "camera_distance_range": [4.0, 6.2],
        "camera_height_range": [0.25, 4.4],
        "camera_azimuth_range": [-25.0, 25.0],
    },
    "wall_impact": {
        "radius_range": [0.20, 0.32],
        "mass_range": [0.7, 1.6],
        "height_range": [0.0, 0.06],
        "x_extent_range": [0.8, 1.6],
        "y_offset_range": [-0.18, 0.18],
        "speed_range": [2.6, 5.0],
        "lateral_speed_range": [-0.65, 0.65],
        "vertical_speed_range": [0.0, 0.08],
        "restitution_range": [0.55, 0.95],
        "friction_range": [0.03, 0.25],
        "wall_distance_range": [1.2, 2.8],
        "camera_target": [0.8, 0.0, 0.5],
        "camera_focal_length": 30.0,
        "camera_distance_range": [4.0, 7.0],
        "camera_height_range": [0.8, 5.2],
        "camera_azimuth_range": [-180.0, 0.0],
    },
}



GENERIC_SAMPLING_DEFAULTS = {
    "drop_height_range": [1.25, 1.85],
    "wall_distance_range": [2.1, 2.9],
    "occluder_y_range": [-1.35, -1.05],
    "occluder_size_range": [0.40, 0.58],
}


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
    parser.add_argument(
        "--scenarios",
        default="two_ball_collision",
        help="Comma-separated physics tests to cycle through: two_ball_collision,ball_drop,rolling_occluder,wall_impact.",
    )
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

    parser.add_argument("--radius-range", type=float, nargs=2, default=None)
    parser.add_argument("--mass-range", type=float, nargs=2, default=None)
    parser.add_argument("--height-range", type=float, nargs=2, default=None)
    parser.add_argument("--drop-height-range", type=float, nargs=2, default=None)
    parser.add_argument("--wall-distance-range", type=float, nargs=2, default=None)
    parser.add_argument("--occluder-y-range", type=float, nargs=2, default=None)
    parser.add_argument("--occluder-size-range", type=float, nargs=2, default=None)
    parser.add_argument("--x-extent-range", type=float, nargs=2, default=None)
    parser.add_argument("--y-offset-range", type=float, nargs=2, default=None)
    parser.add_argument("--speed-range", type=float, nargs=2, default=None)
    parser.add_argument("--lateral-speed-range", type=float, nargs=2, default=None)
    parser.add_argument("--vertical-speed-range", type=float, nargs=2, default=None)
    parser.add_argument("--angular-speed-range", type=float, nargs=2, default=[-6.0, 6.0])
    parser.add_argument("--restitution-range", type=float, nargs=2, default=None)
    parser.add_argument("--friction-range", type=float, nargs=2, default=None)

    parser.add_argument("--camera-distance-range", type=float, nargs=2, default=None)
    parser.add_argument("--camera-height-range", type=float, nargs=2, default=None)
    parser.add_argument("--camera-azimuth-range", type=float, nargs=2, default=None)
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


def scenario_list(raw: str) -> list[str]:
    scenarios = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"two_ball_collision", "ball_drop", "rolling_occluder", "wall_impact"}
    invalid = [item for item in scenarios if item not in valid]
    if invalid:
        raise ValueError(f"Unknown scenario(s): {invalid}. Valid scenarios: {sorted(valid)}")
    return scenarios or ["two_ball_collision"]


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


def scenario_default_range(scenario: str, key: str) -> list[float]:
    defaults = SCENARIO_SAMPLING_DEFAULTS.get(scenario, SCENARIO_SAMPLING_DEFAULTS["two_ball_collision"])
    value = defaults.get(key, SCENARIO_SAMPLING_DEFAULTS["two_ball_collision"].get(key, GENERIC_SAMPLING_DEFAULTS.get(key)))
    if value is None:
        raise KeyError(f"No default range for {scenario}.{key}")
    return list(value)


def range_arg(args: argparse.Namespace, scenario: str, attr: str) -> list[float]:
    value = getattr(args, attr)
    if value is not None:
        return list(value)
    return scenario_default_range(scenario, attr)


def scenario_camera_target(scenario: str) -> list[float]:
    defaults = SCENARIO_SAMPLING_DEFAULTS.get(scenario, SCENARIO_SAMPLING_DEFAULTS["two_ball_collision"])
    return list(defaults.get("camera_target", SCENARIO_SAMPLING_DEFAULTS["two_ball_collision"]["camera_target"]))


def scenario_camera_focal_length(scenario: str) -> float:
    defaults = SCENARIO_SAMPLING_DEFAULTS.get(scenario, SCENARIO_SAMPLING_DEFAULTS["two_ball_collision"])
    fallback = {"two_ball_collision": 35.0, "ball_drop": 28.0, "rolling_occluder": 35.0, "wall_impact": 30.0}
    return float(defaults.get("camera_focal_length", fallback.get(scenario, 35.0)))


def sample_params(args: argparse.Namespace, sample_idx: int, scenario: str) -> dict[str, object]:
    rng = random.Random(args.seed + sample_idx)
    radius_range = parse_range(range_arg(args, scenario, "radius_range"), "--radius-range")
    mass_range = parse_range(range_arg(args, scenario, "mass_range"), "--mass-range")
    height_range = parse_range(range_arg(args, scenario, "height_range"), "--height-range")
    drop_height_range = parse_range(range_arg(args, scenario, "drop_height_range"), "--drop-height-range")
    wall_distance_range = parse_range(range_arg(args, scenario, "wall_distance_range"), "--wall-distance-range")
    occluder_y_range = parse_range(range_arg(args, scenario, "occluder_y_range"), "--occluder-y-range")
    occluder_size_range = parse_range(range_arg(args, scenario, "occluder_size_range"), "--occluder-size-range")
    x_extent_range = parse_range(range_arg(args, scenario, "x_extent_range"), "--x-extent-range")
    y_offset_range = parse_range(range_arg(args, scenario, "y_offset_range"), "--y-offset-range")
    speed_range = parse_range(range_arg(args, scenario, "speed_range"), "--speed-range")
    lateral_speed_range = parse_range(range_arg(args, scenario, "lateral_speed_range"), "--lateral-speed-range")
    vertical_speed_range = parse_range(range_arg(args, scenario, "vertical_speed_range"), "--vertical-speed-range")
    angular_speed_range = parse_range(args.angular_speed_range, "--angular-speed-range")
    restitution_range = parse_range(range_arg(args, scenario, "restitution_range"), "--restitution-range")
    friction_range = parse_range(range_arg(args, scenario, "friction_range"), "--friction-range")
    camera_distance_range = parse_range(range_arg(args, scenario, "camera_distance_range"), "--camera-distance-range")
    camera_height_range = parse_range(range_arg(args, scenario, "camera_height_range"), "--camera-height-range")
    camera_azimuth_range = parse_range(range_arg(args, scenario, "camera_azimuth_range"), "--camera-azimuth-range")
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

    ball_0_xy = [-x0, y0]
    ball_1_xy = [x1, y1]
    ball_0_velocity = [speed0, lateral0, vertical0]
    ball_1_velocity = [-speed1, lateral1, vertical1]
    if scenario == "rolling_occluder" and rng.random() < 0.5:
        ball_0_xy = [x0, y0]
        ball_1_xy = [-x1, y1]
        ball_0_velocity = [-speed0, lateral0, vertical0]
        ball_1_velocity = [speed1, lateral1, vertical1]

    return {
        "seed": args.seed + sample_idx,
        "ball_0_radius": sample_range(rng, radius_range),
        "ball_1_radius": sample_range(rng, radius_range),
        "ball_0_mass": sample_range(rng, mass_range),
        "ball_1_mass": sample_range(rng, mass_range),
        "ball_0_xy": ball_0_xy,
        "ball_1_xy": ball_1_xy,
        "ball_0_height": sample_range(rng, height_range),
        "ball_1_height": sample_range(rng, height_range),
        "drop_height": sample_range(rng, drop_height_range),
        "wall_distance": sample_range(rng, wall_distance_range),
        "occluder_y": sample_range(rng, occluder_y_range),
        "occluder_size": sample_range(rng, occluder_size_range),
        "ball_0_velocity": ball_0_velocity,
        "ball_1_velocity": ball_1_velocity,
        "ball_0_angular_velocity": [sample_range(rng, angular_speed_range) for _ in range(3)],
        "ball_1_angular_velocity": [sample_range(rng, angular_speed_range) for _ in range(3)],
        "restitution": sample_range(rng, restitution_range),
        "lateral_friction": sample_range(rng, friction_range),
        "camera_distance": sample_range(rng, camera_distance_range),
        "camera_height": sample_range(rng, camera_height_range),
        "camera_azimuth": sample_range(rng, camera_azimuth_range),
        "camera_target": scenario_camera_target(scenario),
        "camera_focal_length": scenario_camera_focal_length(scenario),
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


def build_sample_command(args: argparse.Namespace, sample_idx: int, sample_dir: Path, scenario: str) -> list[str]:
    params = sample_params(args, sample_idx, scenario)
    cmd = [
        sys.executable,
        str(RUN_ONE),
        "--output-dir",
        str(sample_dir),
        "--blender-bin",
        args.blender_bin,
        "--scenario",
        scenario,
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
        "--camera-target",
        *(str(value) for value in params["camera_target"]),
        "--camera-distance",
        str(params["camera_distance"]),
        "--camera-height",
        str(params["camera_height"]),
        "--camera-azimuth",
        str(params["camera_azimuth"]),
        "--camera-focal-length",
        str(params["camera_focal_length"]),
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
    cmd.extend(["--drop-height", str(params["drop_height"])])
    extend_vec(cmd, "--wall-position", [params["wall_distance"], 0.0, 1.0])  # type: ignore[list-item]
    extend_vec(cmd, "--wall-half-extents", [0.05, 2.5, 1.0])
    extend_vec(cmd, "--occluder-position", [0.0, params["occluder_y"], params["occluder_size"]])  # type: ignore[list-item]
    extend_vec(cmd, "--occluder-half-extents", [params["occluder_size"], params["occluder_size"], params["occluder_size"]])  # type: ignore[list-item]
    extend_vec(cmd, "--ball-0-xy", params["ball_0_xy"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-1-xy", params["ball_1_xy"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-0-velocity", params["ball_0_velocity"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-1-velocity", params["ball_1_velocity"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-0-angular-velocity", params["ball_0_angular_velocity"])  # type: ignore[arg-type]
    extend_vec(cmd, "--ball-1-angular-velocity", params["ball_1_angular_velocity"])  # type: ignore[arg-type]
    return cmd


def run_sample(args: argparse.Namespace, sample_number: int, gpu_ids: list[str]) -> str:
    sample_idx = args.start_index + sample_number
    scenarios = scenario_list(args.scenarios)
    scenario = scenarios[sample_number % len(scenarios)]
    sample_name = f"{scenario}_{sample_idx:06d}"
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

    cmd = build_sample_command(args, sample_idx, sample_dir, scenario)
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
        "--include-parts",
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
        for sample_number in tqdm(range(args.num_samples), desc="Generating physics samples"):
            run_sample(args, sample_number, gpu_ids)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(run_sample, args, sample_number, gpu_ids)
                for sample_number in range(args.num_samples)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating physics samples"):
                future.result()

    run_preprocess(args)

    if not args.keep_raw:
        shutil.rmtree(args.raw_root)

    print(f"Processed dataset: {args.processed_root}")
    print(f"Dataset JSON: {args.json_output}")


if __name__ == "__main__":
    main()

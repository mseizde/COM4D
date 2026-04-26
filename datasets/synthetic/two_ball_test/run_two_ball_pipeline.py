#!/usr/bin/env python3

"""Run two-ball physics generation and Blender rendering in one command.

Example:
  micromamba run -n com4d python datasets/synthetic/two_ball_test/run_two_ball_pipeline.py \
    --output-dir outputs/two_ball_test/seed_0001 \
    --num-frames 96 \
    --seed 1 \
    --position-jitter 0.08 \
    --height-jitter 0.15 \
    --velocity-jitter 0.25 \
    --mass-jitter 0.2 \
    --random-view \
    --view-seed 1 \
    --resolution 512 \
    --samples 32 \
    --device CPU
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "two_ball_test"
DEFAULT_BLEND_FILE = SCRIPT_DIR / "two_ball_scene.blend"
GENERATOR = SCRIPT_DIR / "generate_physics_metadata.py"
RENDERER = SCRIPT_DIR / "render_blender_outputs.py"


def add_vec3_arg(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(name, type=float, nargs=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--blend-file", type=Path, default=DEFAULT_BLEND_FILE)
    parser.add_argument("--skip-render", action="store_true")

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--ball-radius", type=float, default=0.25)
    parser.add_argument("--ball-0-radius", type=float)
    parser.add_argument("--ball-1-radius", type=float)
    parser.add_argument("--ball-mass", type=float, default=1.0)
    parser.add_argument("--ball-0-mass", type=float)
    parser.add_argument("--ball-1-mass", type=float)
    parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -9.81])
    parser.add_argument("--ball-0-xy", type=float, nargs=2, default=[-1.0, 0.0])
    parser.add_argument("--ball-1-xy", type=float, nargs=2, default=[1.0, 0.0])
    parser.add_argument("--ball-0-height", type=float, default=0.0)
    parser.add_argument("--ball-1-height", type=float, default=0.0)
    add_vec3_arg(parser, "--ball-0-position")
    add_vec3_arg(parser, "--ball-1-position")
    parser.add_argument("--ball-0-velocity", type=float, nargs=3, default=[2.0, 0.0, 0.0])
    parser.add_argument("--ball-1-velocity", type=float, nargs=3, default=[-2.0, 0.0, 0.0])
    parser.add_argument("--ball-0-angular-velocity", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--ball-1-angular-velocity", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--restitution", type=float, default=0.9)
    parser.add_argument("--lateral-friction", type=float, default=0.2)
    parser.add_argument("--rolling-friction", type=float, default=0.01)
    parser.add_argument("--spinning-friction", type=float, default=0.01)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--position-jitter", type=float, default=0.0)
    parser.add_argument("--height-jitter", type=float, default=0.0)
    parser.add_argument("--velocity-jitter", type=float, default=0.0)
    parser.add_argument("--mass-jitter", type=float, default=0.0)

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", choices=("AUTO", "CPU", "GPU"), default="AUTO")
    parser.add_argument(
        "--blender-threads",
        type=int,
        default=0,
        help="Threads passed to Blender with --threads. Use 0 to keep Blender's default.",
    )
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE"), default="CYCLES")
    parser.add_argument("--view-seed", type=int)
    add_vec3_arg(parser, "--camera-location")
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.0, 0.0, 0.35])
    parser.add_argument("--camera-distance", type=float, default=5.0)
    parser.add_argument("--camera-height", type=float, default=2.0)
    parser.add_argument("--camera-azimuth", type=float)
    parser.add_argument("--camera-elevation-jitter", type=float, default=0.0)
    parser.add_argument("--random-view", action="store_true")
    parser.add_argument(
        "--ball-0-look",
        choices=("basketball", "black", "blue", "football", "green", "orange", "red", "white"),
        default="red",
    )
    parser.add_argument(
        "--ball-1-look",
        choices=("basketball", "black", "blue", "football", "green", "orange", "red", "white"),
        default="blue",
    )
    parser.add_argument("--ball-0-color", type=float, nargs=4)
    parser.add_argument("--ball-1-color", type=float, nargs=4)
    parser.add_argument(
        "--floor-look",
        choices=("blue_gray", "dark_gray", "gray", "green_gray", "light_gray"),
        default="gray",
    )
    parser.add_argument("--floor-color", type=float, nargs=4)
    parser.add_argument("--material-roughness", type=float, default=0.45)
    parser.add_argument("--light-seed", type=int)
    add_vec3_arg(parser, "--light-location")
    parser.add_argument("--light-energy", type=float, default=500.0)
    parser.add_argument("--light-size", type=float, default=4.0)
    parser.add_argument("--random-light", action="store_true")
    parser.add_argument("--light-distance", type=float, default=4.0)
    parser.add_argument("--light-height", type=float, default=5.0)
    parser.add_argument("--light-energy-jitter", type=float, default=0.0)
    parser.add_argument("--light-size-jitter", type=float, default=0.0)
    parser.add_argument("--skip-masks", action="store_true")
    parser.add_argument("--skip-transforms", action="store_true")
    parser.add_argument("--skip-canonical-meshes", action="store_true")
    return parser.parse_args()


def append_optional(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def append_vec(cmd: list[str], flag: str, values: list[float] | None) -> None:
    if values is not None:
        cmd.append(flag)
        cmd.extend(str(value) for value in values)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_generator_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--output-dir",
        str(args.output_dir),
        "--fps",
        str(args.fps),
        "--num-frames",
        str(args.num_frames),
        "--ball-radius",
        str(args.ball_radius),
        "--ball-mass",
        str(args.ball_mass),
        "--gravity",
        *(str(value) for value in args.gravity),
        "--ball-0-xy",
        *(str(value) for value in args.ball_0_xy),
        "--ball-1-xy",
        *(str(value) for value in args.ball_1_xy),
        "--ball-0-height",
        str(args.ball_0_height),
        "--ball-1-height",
        str(args.ball_1_height),
        "--ball-0-velocity",
        *(str(value) for value in args.ball_0_velocity),
        "--ball-1-velocity",
        *(str(value) for value in args.ball_1_velocity),
        "--ball-0-angular-velocity",
        *(str(value) for value in args.ball_0_angular_velocity),
        "--ball-1-angular-velocity",
        *(str(value) for value in args.ball_1_angular_velocity),
        "--restitution",
        str(args.restitution),
        "--lateral-friction",
        str(args.lateral_friction),
        "--rolling-friction",
        str(args.rolling_friction),
        "--spinning-friction",
        str(args.spinning_friction),
        "--position-jitter",
        str(args.position_jitter),
        "--height-jitter",
        str(args.height_jitter),
        "--velocity-jitter",
        str(args.velocity_jitter),
        "--mass-jitter",
        str(args.mass_jitter),
    ]
    append_optional(cmd, "--ball-0-mass", args.ball_0_mass)
    append_optional(cmd, "--ball-1-mass", args.ball_1_mass)
    append_optional(cmd, "--ball-0-radius", args.ball_0_radius)
    append_optional(cmd, "--ball-1-radius", args.ball_1_radius)
    append_optional(cmd, "--seed", args.seed)
    append_vec(cmd, "--ball-0-position", args.ball_0_position)
    append_vec(cmd, "--ball-1-position", args.ball_1_position)
    return cmd


def build_renderer_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.blender_bin,
        "--background",
        str(args.blend_file),
        "--python",
        str(RENDERER),
        "--",
        "--base-dir",
        str(args.output_dir),
        "--blend-file",
        str(args.blend_file),
        "--resolution",
        str(args.resolution),
        "--samples",
        str(args.samples),
        "--device",
        args.device,
        "--engine",
        args.engine,
        "--camera-target",
        *(str(value) for value in args.camera_target),
        "--camera-distance",
        str(args.camera_distance),
        "--camera-height",
        str(args.camera_height),
        "--camera-elevation-jitter",
        str(args.camera_elevation_jitter),
        "--ball-0-look",
        args.ball_0_look,
        "--ball-1-look",
        args.ball_1_look,
        "--floor-look",
        args.floor_look,
        "--material-roughness",
        str(args.material_roughness),
        "--light-energy",
        str(args.light_energy),
        "--light-size",
        str(args.light_size),
        "--light-distance",
        str(args.light_distance),
        "--light-height",
        str(args.light_height),
        "--light-energy-jitter",
        str(args.light_energy_jitter),
        "--light-size-jitter",
        str(args.light_size_jitter),
    ]
    if args.blender_threads > 0:
        cmd[1:1] = ["--threads", str(args.blender_threads)]
    append_optional(cmd, "--view-seed", args.view_seed)
    append_optional(cmd, "--camera-azimuth", args.camera_azimuth)
    append_optional(cmd, "--light-seed", args.light_seed)
    append_vec(cmd, "--camera-location", args.camera_location)
    append_vec(cmd, "--light-location", args.light_location)
    append_vec(cmd, "--ball-0-color", args.ball_0_color)
    append_vec(cmd, "--ball-1-color", args.ball_1_color)
    append_vec(cmd, "--floor-color", args.floor_color)
    if args.random_view:
        cmd.append("--random-view")
    if args.random_light:
        cmd.append("--random-light")
    if args.skip_masks:
        cmd.append("--skip-masks")
    if args.skip_transforms:
        cmd.append("--skip-transforms")
    if args.skip_canonical_meshes:
        cmd.append("--skip-canonical-meshes")
    return cmd


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.blend_file = args.blend_file.expanduser().resolve()

    run(build_generator_cmd(args))
    if not args.skip_render:
        run(build_renderer_cmd(args))


if __name__ == "__main__":
    main()

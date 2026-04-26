#!/usr/bin/env python3

"""Generate two-ball PyBullet trajectory metadata for Blender rendering.

Cluster/headless example:
  micromamba run -n com4d python datasets/synthetic/two_ball_test/generate_physics_metadata.py \
    --output-dir outputs/two_ball_test \
    --num-frames 32
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pybullet as p
import pybullet_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "two_ball_test"


def initial_position(
    full_position: list[float] | None,
    xy: list[float],
    height: float,
    radius: float,
) -> list[float]:
    if full_position is not None:
        return list(full_position)
    return [xy[0], xy[1], radius + height]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for physics_metadata.json. Defaults to COM4D/outputs/two_ball_test.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--ball-radius", type=float, default=0.25)
    parser.add_argument("--ball-0-radius", type=float)
    parser.add_argument("--ball-1-radius", type=float)
    parser.add_argument("--ball-mass", type=float, default=1.0)
    parser.add_argument("--ball-0-mass", type=float)
    parser.add_argument("--ball-1-mass", type=float)
    parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -9.81])
    parser.add_argument(
        "--ball-0-xy",
        type=float,
        nargs=2,
        default=[-1.0, 0.0],
        help="Recommended: initial ball_0 planar position as X Y.",
    )
    parser.add_argument(
        "--ball-1-xy",
        type=float,
        nargs=2,
        default=[1.0, 0.0],
        help="Recommended: initial ball_1 planar position as X Y.",
    )
    parser.add_argument(
        "--ball-0-height",
        type=float,
        default=0.0,
        help="Recommended: initial ball_0 clearance above ground. Center Z becomes ball_radius + height.",
    )
    parser.add_argument(
        "--ball-1-height",
        type=float,
        default=0.0,
        help="Recommended: initial ball_1 clearance above ground. Center Z becomes ball_radius + height.",
    )
    parser.add_argument(
        "--ball-0-position",
        type=float,
        nargs=3,
        help="Advanced override: full initial ball_0 center position as X Y Z.",
    )
    parser.add_argument(
        "--ball-1-position",
        type=float,
        nargs=3,
        help="Advanced override: full initial ball_1 center position as X Y Z.",
    )
    parser.add_argument(
        "--height-jitter",
        type=float,
        default=0.0,
        help="Uniform jitter applied to height above ground, then converted to center Z and clamped.",
    )
    parser.add_argument("--ball-0-velocity", type=float, nargs=3, default=[2.0, 0.0, 0.0])
    parser.add_argument("--ball-1-velocity", type=float, nargs=3, default=[-2.0, 0.0, 0.0])
    parser.add_argument("--ball-0-angular-velocity", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--ball-1-angular-velocity", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--restitution", type=float, default=0.9)
    parser.add_argument("--lateral-friction", type=float, default=0.2)
    parser.add_argument("--rolling-friction", type=float, default=0.01)
    parser.add_argument("--spinning-friction", type=float, default=0.01)
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for deterministic random perturbations.",
    )
    parser.add_argument(
        "--position-jitter",
        type=float,
        default=0.0,
        help="Uniform XY jitter applied to each initial position. Z is unchanged.",
    )
    parser.add_argument(
        "--velocity-jitter",
        type=float,
        default=0.0,
        help="Uniform jitter applied to each component of each initial linear velocity.",
    )
    parser.add_argument(
        "--mass-jitter",
        type=float,
        default=0.0,
        help="Uniform mass jitter applied independently to each ball.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use PyBullet GUI for local debugging. Default is DIRECT for cluster/headless runs.",
    )
    parser.add_argument(
        "--realtime-sleep",
        action="store_true",
        help="Sleep between frames when using --gui.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    ball_0_radius = args.ball_radius if args.ball_0_radius is None else args.ball_0_radius
    ball_1_radius = args.ball_radius if args.ball_1_radius is None else args.ball_1_radius
    ball_0_mass = args.ball_mass if args.ball_0_mass is None else args.ball_0_mass
    ball_1_mass = args.ball_mass if args.ball_1_mass is None else args.ball_1_mass
    ball_0_height = args.ball_0_height
    ball_1_height = args.ball_1_height
    ball_0_velocity = list(args.ball_0_velocity)
    ball_1_velocity = list(args.ball_1_velocity)

    if args.mass_jitter:
        ball_0_mass = max(0.001, ball_0_mass + rng.uniform(-args.mass_jitter, args.mass_jitter))
        ball_1_mass = max(0.001, ball_1_mass + rng.uniform(-args.mass_jitter, args.mass_jitter))

    if args.height_jitter:
        ball_0_height = max(0.0, ball_0_height + rng.uniform(-args.height_jitter, args.height_jitter))
        ball_1_height = max(0.0, ball_1_height + rng.uniform(-args.height_jitter, args.height_jitter))

    ball_0_position = initial_position(args.ball_0_position, args.ball_0_xy, ball_0_height, ball_0_radius)
    ball_1_position = initial_position(args.ball_1_position, args.ball_1_xy, ball_1_height, ball_1_radius)

    if args.position_jitter:
        for position in (ball_0_position, ball_1_position):
            position[0] += rng.uniform(-args.position_jitter, args.position_jitter)
            position[1] += rng.uniform(-args.position_jitter, args.position_jitter)

    ball_0_position[2] = max(ball_0_radius, ball_0_position[2])
    ball_1_position[2] = max(ball_1_radius, ball_1_position[2])

    if args.velocity_jitter:
        for velocity in (ball_0_velocity, ball_1_velocity):
            for axis in range(3):
                velocity[axis] += rng.uniform(-args.velocity_jitter, args.velocity_jitter)

    dt = 1.0 / args.fps
    mode = p.GUI if args.gui else p.DIRECT
    physics_client = p.connect(mode)

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*args.gravity)
        p.setTimeStep(dt)

        p.loadURDF("plane.urdf")

        collision_shape_0 = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=ball_0_radius,
        )
        collision_shape_1 = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=ball_1_radius,
        )
        visual_red = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=ball_0_radius,
            rgbaColor=[1, 0.1, 0.1, 1],
        )
        visual_blue = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=ball_1_radius,
            rgbaColor=[0.1, 0.1, 1, 1],
        )

        ball_0 = p.createMultiBody(
            baseMass=ball_0_mass,
            baseCollisionShapeIndex=collision_shape_0,
            baseVisualShapeIndex=visual_red,
            basePosition=ball_0_position,
        )
        ball_1 = p.createMultiBody(
            baseMass=ball_1_mass,
            baseCollisionShapeIndex=collision_shape_1,
            baseVisualShapeIndex=visual_blue,
            basePosition=ball_1_position,
        )

        for ball in [ball_0, ball_1]:
            p.changeDynamics(
                ball,
                -1,
                restitution=args.restitution,
                lateralFriction=args.lateral_friction,
                rollingFriction=args.rolling_friction,
                spinningFriction=args.spinning_friction,
            )

        p.resetBaseVelocity(
            ball_0,
            linearVelocity=ball_0_velocity,
            angularVelocity=args.ball_0_angular_velocity,
        )
        p.resetBaseVelocity(
            ball_1,
            linearVelocity=ball_1_velocity,
            angularVelocity=args.ball_1_angular_velocity,
        )

        frames = []
        collision_frame = None

        for frame_idx in range(args.num_frames):
            p.stepSimulation()

            pos_0, quat_0 = p.getBasePositionAndOrientation(ball_0)
            pos_1, quat_1 = p.getBasePositionAndOrientation(ball_1)
            vel_0, ang_0 = p.getBaseVelocity(ball_0)
            vel_1, ang_1 = p.getBaseVelocity(ball_1)
            contacts = p.getContactPoints(ball_0, ball_1)

            if contacts and collision_frame is None:
                collision_frame = frame_idx

            frames.append(
                {
                    "frame": frame_idx,
                    "ball_0": {
                        "position": list(pos_0),
                        "quaternion": list(quat_0),
                        "linear_velocity": list(vel_0),
                        "angular_velocity": list(ang_0),
                    },
                    "ball_1": {
                        "position": list(pos_1),
                        "quaternion": list(quat_1),
                        "linear_velocity": list(vel_1),
                        "angular_velocity": list(ang_1),
                    },
                    "contact": len(contacts) > 0,
                }
            )

            if args.gui and args.realtime_sleep:
                time.sleep(dt)

        metadata = {
            "fps": args.fps,
            "num_frames": args.num_frames,
            "ball_radius": args.ball_radius,
            "ball_radii": {
                "ball_0": ball_0_radius,
                "ball_1": ball_1_radius,
            },
            "ball_mass": args.ball_mass,
            "ball_masses": {
                "ball_0": ball_0_mass,
                "ball_1": ball_1_mass,
            },
            "gravity": args.gravity,
            "initial_conditions": {
                "seed": args.seed,
                "ball_0_xy": args.ball_0_xy,
                "ball_1_xy": args.ball_1_xy,
                "ball_0_height": ball_0_height,
                "ball_1_height": ball_1_height,
                "height_semantics": "center_z = ball_radius + height_above_ground unless full --ball-*-position override is used",
                "position_jitter": args.position_jitter,
                "height_jitter": args.height_jitter,
                "velocity_jitter": args.velocity_jitter,
                "mass_jitter": args.mass_jitter,
                "ball_0": {
                    "radius": ball_0_radius,
                    "mass": ball_0_mass,
                    "position": ball_0_position,
                    "linear_velocity": ball_0_velocity,
                    "angular_velocity": args.ball_0_angular_velocity,
                },
                "ball_1": {
                    "radius": ball_1_radius,
                    "mass": ball_1_mass,
                    "position": ball_1_position,
                    "linear_velocity": ball_1_velocity,
                    "angular_velocity": args.ball_1_angular_velocity,
                },
                "dynamics": {
                    "restitution": args.restitution,
                    "lateral_friction": args.lateral_friction,
                    "rolling_friction": args.rolling_friction,
                    "spinning_friction": args.spinning_friction,
                },
            },
            "collision_frame": collision_frame,
            "frames": frames,
        }

        out_path = output_dir / "physics_metadata.json"
        with out_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to: {out_path}")
        print(f"Collision frame: {collision_frame}")
    finally:
        p.disconnect(physics_client)


if __name__ == "__main__":
    main()

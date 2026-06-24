import argparse
import json
import random
import time
from pathlib import Path

import pybullet as p
import pybullet_data


SCENARIO_DEFAULTS = {
    "two_ball_collision": {
        "fps": 30,
        "num_frames": 48,
        "active_objects": ["ball_0", "ball_1", "floor"],
        "render_objects": ["ball_0", "ball_1", "floor"],
        "physics_objects": ["ball_0", "ball_1", "floor"],
        "ball_radius": 0.25,
        "ball_mass": 1.0,
        "restitution": 0.7,
        "friction": 0.2,
        "ball_0_position": [-1.0, 0.0, 0.25],
        "ball_1_position": [1.0, 0.0, 0.25],
        "ball_0_velocity": [2.0, 0.0, 0.0],
        "ball_1_velocity": [-2.0, 0.0, 0.0],
    },

    "ball_drop": {
        "fps": 30,
        "num_frames": 64,
        "active_objects": ["ball_0", "floor"],
        "render_objects": ["ball_0", "floor"],
        "physics_objects": ["ball_0", "floor"],
        "ball_radius": 0.25,
        "ball_mass": 1.0,
        "restitution": 0.8,
        "friction": 0.1,
        "height": 1.5,
        "ball_0_velocity": [1.5, 0.0, 0.0],
    },

    "rolling_occluder": {
        "fps": 30,
        "num_frames": 64,
        "active_objects": ["ball_0", "floor", "occluder_box"],
        "render_objects": ["ball_0", "floor", "occluder_box"],
        "physics_objects": ["ball_0", "floor", "occluder_box"],
        "ball_radius": 0.25,
        "ball_mass": 1.0,
        "restitution": 0.4,
        "friction": 0.6,
        "ball_0_position": [-2.0, 0.0, 0.25],
        "ball_0_velocity": [3.0, 0.0, 0.0],
        "occluder_position": [0.0, -1.2, 0.6],
        "occluder_half_extents": [0.6, 0.6, 0.6],
    },

    "wall_impact": {
        "fps": 30,
        "num_frames": 64,
        "active_objects": ["ball_0", "floor", "wall"],
        "render_objects": ["ball_0", "floor", "wall"],
        "physics_objects": ["ball_0", "floor", "wall"],
        "ball_radius": 0.25,
        "ball_mass": 1.0,
        "restitution": 0.8,
        "friction": 0.1,
        "ball_0_position": [-1.5, 0.0, 0.25],
        "ball_0_velocity": [4.0, 0.0, 0.0],
        "wall_position": [2.5, 0.0, 1.0],
        "wall_half_extents": [0.05, 2.5, 1.0],
    },
}


CONTACT_PAIRS = {
    "two_ball_collision": [("ball_0", "ball_1")],
    "ball_drop": [("ball_0", "floor")],
    "rolling_occluder": [("ball_0", "occluder_box")],
    "wall_impact": [("ball_0", "wall")],
}


def deep_copy_dict(d):
    return json.loads(json.dumps(d))


def object_specs_from_config(config):
    specs = {}
    radius = float(config.get("ball_radius", 0.25))
    mass = float(config.get("ball_mass", 1.0))
    for name in config.get("render_objects", []):
        if name.startswith("ball_"):
            specs[name] = {
                "type": "sphere",
                "dynamic": True,
                "radius": radius,
                "mass": mass,
            }
        elif name == "floor":
            specs[name] = {"type": "plane", "dynamic": False}
        elif name == "wall":
            half_extents = config.get("wall_half_extents", [0.05, 2.5, 1.0])
            specs[name] = {
                "type": "box",
                "dynamic": False,
                "half_extents": half_extents,
                "size": [2.0 * float(value) for value in half_extents],
                "position": config.get("wall_position", [0.0, 0.0, 0.0]),
            }
        elif name == "occluder_box":
            half_extents = config.get("occluder_half_extents", [0.6, 0.6, 0.6])
            specs[name] = {
                "type": "box",
                "dynamic": False,
                "half_extents": half_extents,
                "size": [2.0 * float(value) for value in half_extents],
                "position": config.get("occluder_position", [0.0, 0.0, 0.0]),
            }
    return specs


def initial_position(full_position, xy, height, radius):
    if full_position is not None:
        return list(full_position)
    return [float(xy[0]), float(xy[1]), float(radius) + float(height)]


def apply_overrides(config, args):
    if args.ball_radius is not None:
        config["ball_radius"] = args.ball_radius
    if args.ball_mass is not None:
        config["ball_mass"] = args.ball_mass
    if args.restitution is not None:
        config["restitution"] = args.restitution
    if args.friction is not None:
        config["friction"] = args.friction
    if args.ball_0_velocity is not None:
        config["ball_0_velocity"] = list(args.ball_0_velocity)
    if args.ball_1_velocity is not None:
        config["ball_1_velocity"] = list(args.ball_1_velocity)

    radius = float(config.get("ball_radius", 0.25))
    if args.ball_0_position is not None or args.ball_0_xy is not None or args.ball_0_height is not None:
        xy = args.ball_0_xy if args.ball_0_xy is not None else config.get("ball_0_position", [0.0, 0.0, radius])[:2]
        height = args.ball_0_height if args.ball_0_height is not None else 0.0
        config["ball_0_position"] = initial_position(args.ball_0_position, xy, height, radius)
    if args.ball_1_position is not None or args.ball_1_xy is not None or args.ball_1_height is not None:
        xy = args.ball_1_xy if args.ball_1_xy is not None else config.get("ball_1_position", [1.0, 0.0, radius])[:2]
        height = args.ball_1_height if args.ball_1_height is not None else 0.0
        config["ball_1_position"] = initial_position(args.ball_1_position, xy, height, radius)

    if args.drop_height is not None:
        config["height"] = max(radius, args.drop_height)
    elif config.get("scenario") == "ball_drop" and args.ball_0_height is not None:
        config["height"] = max(radius, radius + args.ball_0_height)

    if args.wall_position is not None:
        config["wall_position"] = list(args.wall_position)
    if args.wall_half_extents is not None:
        config["wall_half_extents"] = list(args.wall_half_extents)
    if args.occluder_position is not None:
        config["occluder_position"] = list(args.occluder_position)
    if args.occluder_half_extents is not None:
        config["occluder_half_extents"] = list(args.occluder_half_extents)


def create_ball(radius, mass, position, color):
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def create_wall(position, half_extents):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.5, 0.5, 0.5, 1],
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def create_box(position, half_extents, color=[0.4, 0.4, 0.4, 1.0]):
    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
    )
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def dynamic_bounds_ok(frames, dynamic_names, max_abs_xy=None, max_z=None):
    if (max_abs_xy is None or max_abs_xy <= 0) and (max_z is None or max_z <= 0):
        return True, None
    for frame in frames:
        frame_idx = frame.get("frame")
        objects = frame.get("objects", {})
        for name in dynamic_names:
            state = objects.get(name, {})
            pos = state.get("position")
            if pos is None:
                continue
            if max_abs_xy is not None and max_abs_xy > 0 and (abs(float(pos[0])) > max_abs_xy or abs(float(pos[1])) > max_abs_xy):
                return False, {"frame": frame_idx, "object": name, "position": pos, "reason": "xy_bound"}
            if max_z is not None and max_z > 0 and float(pos[2]) > max_z:
                return False, {"frame": frame_idx, "object": name, "position": pos, "reason": "z_bound"}
    return True, None


def simulate(config, out_dir, use_gui=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = config["fps"]
    num_frames = config["num_frames"]
    dt = 1.0 / fps

    mode = p.GUI if use_gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(dt)
    p.setPhysicsEngineParameter(numSubSteps=4)

    object_ids = {}

    # Floor
    if "floor" in config["physics_objects"]:
        object_ids["floor"] = p.loadURDF("plane.urdf")

    radius = config["ball_radius"]
    mass = config["ball_mass"]

    # Ball 0
    if "ball_0" in config["physics_objects"]:
        if config["scenario"] == "ball_drop":
            pos = [0.0, 0.0, config["height"]]
        else:
            pos = config.get("ball_0_position", [0.0, 0.0, radius])

        object_ids["ball_0"] = create_ball(
            radius=radius,
            mass=mass,
            position=pos,
            color=[1.0, 0.1, 0.1, 1.0],
        )
        p.resetBaseVelocity(
            object_ids["ball_0"],
            linearVelocity=config.get("ball_0_velocity", [0, 0, 0]),
        )

    # Ball 1
    if "ball_1" in config["physics_objects"]:
        object_ids["ball_1"] = create_ball(
            radius=radius,
            mass=mass,
            position=config["ball_1_position"],
            color=[0.1, 0.1, 1.0, 1.0],
        )
        p.resetBaseVelocity(
            object_ids["ball_1"],
            linearVelocity=config.get("ball_1_velocity", [0, 0, 0]),
        )

    # Wall
    if "wall" in config["physics_objects"]:
        object_ids["wall"] = create_wall(
            position=config["wall_position"],
            half_extents=config["wall_half_extents"],
        )

    # Occluder Box
    if "occluder_box" in config["physics_objects"]:
        object_ids["occluder_box"] = create_box(
            position=config["occluder_position"],
            half_extents=config["occluder_half_extents"],
            color=[0.4, 0.4, 0.4, 1.0],
        )

    # Dynamics
    for name, body_id in object_ids.items():
        if name.startswith("ball"):
            p.changeDynamics(
                body_id,
                -1,
                restitution=config["restitution"],
                lateralFriction=config["friction"],
                rollingFriction=0.01,
                spinningFriction=0.01,
            )
        else:
            p.changeDynamics(
                body_id,
                -1,
                restitution=config["restitution"],
                lateralFriction=config["friction"],
                rollingFriction=0.0,
                spinningFriction=0.0,
            )

    # Camera position adjustment
    if use_gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=0,
            cameraPitch=-35,
            cameraTargetPosition=[0.5, 0.0, 0.5],
            )

    frames = []
    first_contact_frame = None

    for frame_idx in range(num_frames):
        p.stepSimulation()

        frame_data = {"frame": frame_idx, "objects": {}, "contacts": []}

        for name, body_id in object_ids.items():
            pos, quat = p.getBasePositionAndOrientation(body_id)
            vel, ang = p.getBaseVelocity(body_id)

            frame_data["objects"][name] = {
                "position": list(pos),
                "quaternion": list(quat),  # PyBullet xyzw
                "linear_velocity": list(vel),
                "angular_velocity": list(ang),
            }

        # Contact logging
        contact_pairs = CONTACT_PAIRS.get(config["scenario"], [])

        for a_name, b_name in contact_pairs:
            if a_name not in object_ids or b_name not in object_ids:
                continue

            contacts = p.getContactPoints(object_ids[a_name], object_ids[b_name])

            if contacts:
                frame_data["contacts"].append([a_name, b_name])

                if first_contact_frame is None:
                    first_contact_frame = frame_idx

        frames.append(frame_data)

        if use_gui:
            time.sleep(dt)

    object_properties = {
        "ball_radius": radius,
        "ball_mass": mass,
        "restitution": config["restitution"],
        "friction": config["friction"],
    }
    if "wall_half_extents" in config:
        object_properties["wall_half_extents"] = config["wall_half_extents"]
    if "occluder_half_extents" in config:
        object_properties["occluder_box_half_extents"] = config["occluder_half_extents"]

    dynamic_names = [name for name, spec in object_specs_from_config(config).items() if bool(spec.get("dynamic", name.startswith("ball_")))]
    bounds_ok, bounds_failure = dynamic_bounds_ok(
        frames,
        dynamic_names,
        config.get("max_dynamic_abs_xy"),
        config.get("max_dynamic_z"),
    )
    if not bounds_ok:
        p.disconnect()
        raise RuntimeError(f"Dynamic object left training bounds: {bounds_failure}")

    metadata = {
        "scenario": config["scenario"],
        "fps": fps,
        "num_frames": num_frames,
        "active_objects": config["active_objects"],
        "render_objects": config["render_objects"],
        "physics_objects": config["physics_objects"],
        "objects": object_specs_from_config(config),
        "ball_radius": radius,
        "ball_radii": {name: radius for name in config["physics_objects"] if name.startswith("ball_")},
        "ball_mass": mass,
        "object_properties": object_properties,
        "first_contact_frame": first_contact_frame,
        "collision_frame": first_contact_frame,
        "training_bounds": {
            "max_dynamic_abs_xy": config.get("max_dynamic_abs_xy"),
            "max_dynamic_z": config.get("max_dynamic_z"),
            "ok": bounds_ok,
        },
        "frames": frames,
    }

    with open(out_dir / "physics_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {out_dir / 'physics_metadata.json'}")
    print(f"Scenario: {config['scenario']}")
    print(f"First contact frame: {first_contact_frame}")

    p.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=list(SCENARIO_DEFAULTS.keys()))
    parser.add_argument("--out", default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--ball-radius", type=float)
    parser.add_argument("--ball-mass", type=float)
    parser.add_argument("--restitution", type=float)
    parser.add_argument("--friction", type=float)
    parser.add_argument("--ball-0-xy", type=float, nargs=2)
    parser.add_argument("--ball-1-xy", type=float, nargs=2)
    parser.add_argument("--ball-0-height", type=float)
    parser.add_argument("--ball-1-height", type=float)
    parser.add_argument("--ball-0-position", type=float, nargs=3)
    parser.add_argument("--ball-1-position", type=float, nargs=3)
    parser.add_argument("--ball-0-velocity", type=float, nargs=3)
    parser.add_argument("--ball-1-velocity", type=float, nargs=3)
    parser.add_argument("--drop-height", type=float)
    parser.add_argument("--wall-position", type=float, nargs=3)
    parser.add_argument("--wall-half-extents", type=float, nargs=3)
    parser.add_argument("--occluder-position", type=float, nargs=3)
    parser.add_argument("--occluder-half-extents", type=float, nargs=3)
    parser.add_argument("--max-dynamic-abs-xy", type=float)
    parser.add_argument("--max-dynamic-z", type=float)
    args = parser.parse_args()

    random.seed(args.seed)

    config = deep_copy_dict(SCENARIO_DEFAULTS[args.scenario])
    config["scenario"] = args.scenario
    if args.fps is not None:
        config["fps"] = args.fps
    if args.num_frames is not None:
        config["num_frames"] = args.num_frames
    apply_overrides(config, args)
    if args.max_dynamic_abs_xy is not None:
        config["max_dynamic_abs_xy"] = args.max_dynamic_abs_xy
    if args.max_dynamic_z is not None:
        config["max_dynamic_z"] = args.max_dynamic_z

    out_dir = args.out or f"outputs/{args.scenario}_test"
    simulate(config, out_dir, use_gui=args.gui)


if __name__ == "__main__":
    main()
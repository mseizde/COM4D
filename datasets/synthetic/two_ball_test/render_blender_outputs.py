#!/usr/bin/env python3

"""Render RGB frames, object masks, transforms, and GLB meshes in Blender.

Cluster/headless example with the included prepared scene:
  micromamba run -n com4d blender --background datasets/synthetic/two_ball_test/two_ball_scene.blend \
    --python datasets/synthetic/two_ball_test/render_blender_outputs.py -- \
    --base-dir outputs/two_ball_test \
    --device CPU

Cluster/headless example using the script's default scene lookup:
  micromamba run -n com4d blender --background \
    --python datasets/synthetic/two_ball_test/render_blender_outputs.py -- \
    --base-dir outputs/two_ball_test \
    --device CPU

When Blender is launched without an explicit .blend, this script opens
two_ball_scene.blend from this folder by default. If no prepared scene is
available, it can still create a simple fallback scene with ball_0 and ball_1.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import mathutils


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BLEND_FILE = SCRIPT_DIR / "two_ball_scene.blend"
DEFAULT_BASE_DIR = PROJECT_ROOT / "outputs" / "two_ball_test"
LOOK_COLORS = {
    "red": (1.0, 0.1, 0.1, 1.0),
    "blue": (0.1, 0.1, 1.0, 1.0),
    "green": (0.1, 0.75, 0.25, 1.0),
    "orange": (1.0, 0.45, 0.05, 1.0),
    "white": (0.95, 0.95, 0.9, 1.0),
    "black": (0.02, 0.02, 0.02, 1.0),
    "basketball": (0.95, 0.35, 0.08, 1.0),
    "football": (0.38, 0.16, 0.06, 1.0),
}
FLOOR_COLORS = {
    "gray": (0.75, 0.75, 0.75, 1.0),
    "dark_gray": (0.35, 0.35, 0.35, 1.0),
    "light_gray": (0.86, 0.86, 0.82, 1.0),
    "blue_gray": (0.48, 0.56, 0.62, 1.0),
    "green_gray": (0.48, 0.58, 0.48, 1.0),
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing physics_metadata.json and receiving render outputs.",
    )
    parser.add_argument(
        "--blend-file",
        type=Path,
        default=DEFAULT_BLEND_FILE,
        help="Prepared .blend to open when Blender was launched without one.",
    )
    parser.add_argument(
        "--no-open-default-blend",
        action="store_true",
        help="Do not auto-open two_ball_scene.blend when Blender was launched without a .blend.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--view-seed",
        type=int,
        help="Seed for deterministic random camera viewpoint selection.",
    )
    parser.add_argument(
        "--camera-location",
        type=float,
        nargs=3,
        help="Override camera location as X Y Z.",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.35],
        help="Point the camera looks at when setting or randomizing the camera.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=5.0,
        help="Camera distance for randomized viewpoints.",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=2.0,
        help="Camera height for randomized viewpoints.",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        help="Camera azimuth in degrees. If omitted with --random-view, sampled uniformly.",
    )
    parser.add_argument(
        "--camera-elevation-jitter",
        type=float,
        default=0.0,
        help="Uniform random jitter added to camera Z when --random-view is used.",
    )
    parser.add_argument(
        "--random-view",
        action="store_true",
        help="Randomize camera azimuth around the interaction using --view-seed.",
    )
    parser.add_argument("--ball-0-look", choices=sorted(LOOK_COLORS), default="red")
    parser.add_argument("--ball-1-look", choices=sorted(LOOK_COLORS), default="blue")
    parser.add_argument("--ball-0-color", type=float, nargs=4)
    parser.add_argument("--ball-1-color", type=float, nargs=4)
    parser.add_argument("--floor-look", choices=sorted(FLOOR_COLORS), default="gray")
    parser.add_argument("--floor-color", type=float, nargs=4)
    parser.add_argument("--material-roughness", type=float, default=0.45)
    parser.add_argument("--light-seed", type=int)
    parser.add_argument("--light-location", type=float, nargs=3)
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
    parser.add_argument(
        "--device",
        choices=("AUTO", "CPU", "GPU"),
        default="AUTO",
        help="Cycles device. AUTO tries GPU and falls back to CPU.",
    )
    parser.add_argument(
        "--engine",
        choices=("CYCLES", "BLENDER_EEVEE"),
        default="CYCLES",
        help="Render engine to use.",
    )
    parser.add_argument(
        "--no-create-missing-balls",
        action="store_true",
        help="Fail if ball_0 or ball_1 is missing instead of creating a simple demo scene.",
    )
    return parser.parse_args(script_args)


def scene_has_required_balls() -> bool:
    return bpy.data.objects.get("ball_0") is not None and bpy.data.objects.get("ball_1") is not None


def open_default_blend_if_needed(blend_file: Path, enabled: bool) -> None:
    if not enabled or bpy.data.filepath:
        return

    blend_file = blend_file.expanduser().resolve()
    if not blend_file.exists():
        print(f"Default blend file not found, using current scene: {blend_file}")
        return

    print(f"Opening default blend file: {blend_file}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))


def look_at(obj: bpy.types.Object, target: list[float]) -> None:
    direction = mathutils.Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def configure_camera_view(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    if scene.camera is None:
        ensure_camera_and_light()

    camera = scene.camera
    if camera is None:
        raise RuntimeError("No camera is available after scene setup.")

    target = list(args.camera_target)
    should_update_camera = args.camera_location is not None or args.random_view or args.camera_azimuth is not None
    if not should_update_camera:
        return

    if args.camera_location is not None:
        camera.location = args.camera_location
    else:
        rng = random.Random(args.view_seed)
        azimuth = args.camera_azimuth
        if azimuth is None:
            azimuth = rng.uniform(0.0, 360.0)
        azimuth_rad = math.radians(azimuth)
        height = args.camera_height
        if args.camera_elevation_jitter:
            height += rng.uniform(-args.camera_elevation_jitter, args.camera_elevation_jitter)

        camera.location = (
            target[0] + args.camera_distance * math.sin(azimuth_rad),
            target[1] - args.camera_distance * math.cos(azimuth_rad),
            target[2] + height,
        )

    look_at(camera, target)
    print(
        "Camera view:",
        {
            "location": [round(v, 6) for v in camera.location],
            "target": target,
        },
    )


def configure_cycles(device: str, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True

    if device == "CPU":
        scene.cycles.device = "CPU"
        return

    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        scene.cycles.device = "CPU"
        return

    cycles_prefs = prefs.preferences
    enabled_gpu = False
    for compute_type in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL", "NONE"):
        try:
            cycles_prefs.compute_device_type = compute_type
            cycles_prefs.get_devices()
        except Exception:
            continue

        for cycles_device in cycles_prefs.devices:
            if cycles_device.type != "CPU":
                cycles_device.use = True
                enabled_gpu = True
            elif device == "GPU":
                cycles_device.use = False

        if enabled_gpu:
            scene.cycles.device = "GPU"
            print(f"Using Cycles GPU backend: {compute_type}")
            return

    if device == "GPU":
        raise RuntimeError("Requested --device GPU, but Blender did not expose a Cycles GPU device.")

    scene.cycles.device = "CPU"
    print("No Cycles GPU device found; using CPU.")


def export_selected_glb(obj: bpy.types.Object, out_path: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )


def get_or_create_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.45,
) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    return material


def material_color(look: str, explicit_color: list[float] | None) -> tuple[float, float, float, float]:
    if explicit_color is not None:
        return tuple(explicit_color)  # type: ignore[return-value]
    return LOOK_COLORS[look]


def floor_color(look: str, explicit_color: list[float] | None) -> tuple[float, float, float, float]:
    if explicit_color is not None:
        return tuple(explicit_color)  # type: ignore[return-value]
    return FLOOR_COLORS[look]


def ensure_camera_and_light() -> None:
    if bpy.context.scene.camera is None:
        bpy.ops.object.camera_add(location=(0.0, -4.0, 1.8), rotation=(1.2, 0.0, 0.0))
        bpy.context.scene.camera = bpy.context.object

    if not any(obj.type == "LIGHT" for obj in bpy.context.scene.objects):
        bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 4.0))
        light = bpy.context.object
        light.name = "key_light"
        light.data.energy = 500.0
        light.data.size = 4.0


def ensure_floor() -> bpy.types.Object:
    existing = bpy.data.objects.get("floor")
    if existing is not None:
        return existing
    bpy.ops.mesh.primitive_plane_add(size=5.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "floor"
    floor.data.materials.append(get_or_create_material("floor_gray", FLOOR_COLORS["gray"]))
    return floor


def configure_floor_material(args: argparse.Namespace) -> None:
    floor = ensure_floor()
    floor.data.materials.clear()
    floor.data.materials.append(
        get_or_create_material(
            "floor_material",
            floor_color(args.floor_look, args.floor_color),
            args.material_roughness,
        )
    )


def remove_default_cube() -> None:
    cube = bpy.data.objects.get("Cube")
    if cube is not None and cube.type == "MESH":
        bpy.data.objects.remove(cube, do_unlink=True)


def object_radius(obj: bpy.types.Object) -> float:
    return max(obj.dimensions) / 2.0


def set_object_radius(obj: bpy.types.Object, radius: float) -> None:
    current_radius = object_radius(obj)
    if current_radius <= 0:
        return
    scale_factor = radius / current_radius
    obj.scale = tuple(value * scale_factor for value in obj.scale)


def ball_radius(data: dict, name: str) -> float:
    return float(data.get("ball_radii", {}).get(name, data.get("ball_radius", 0.25)))


def ensure_ball_objects(
    data: dict,
    args: argparse.Namespace,
    create_missing: bool,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    ball_radii = {
        "ball_0": ball_radius(data, "ball_0"),
        "ball_1": ball_radius(data, "ball_1"),
    }
    materials = {
        "ball_0": get_or_create_material(
            "ball_0_material",
            material_color(args.ball_0_look, args.ball_0_color),
            args.material_roughness,
        ),
        "ball_1": get_or_create_material(
            "ball_1_material",
            material_color(args.ball_1_look, args.ball_1_color),
            args.material_roughness,
        ),
    }

    if create_missing and (bpy.data.objects.get("ball_0") is None or bpy.data.objects.get("ball_1") is None):
        remove_default_cube()

    for name, x in (("ball_0", -1.0), ("ball_1", 1.0)):
        obj = bpy.data.objects.get(name)
        if obj is None:
            if not create_missing:
                raise KeyError(f"Missing required Blender object: {name}")
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=64,
                ring_count=32,
                radius=ball_radii[name],
                location=(x, 0.0, ball_radii[name]),
            )
            obj = bpy.context.object
            obj.name = name
        else:
            set_object_radius(obj, ball_radii[name])
        obj.data.materials.clear()
        obj.data.materials.append(materials[name])

    if create_missing:
        ensure_floor()
        ensure_camera_and_light()

    return bpy.data.objects["ball_0"], bpy.data.objects["ball_1"]


def configure_lighting(args: argparse.Namespace) -> None:
    ensure_camera_and_light()
    lights = [obj for obj in bpy.context.scene.objects if obj.type == "LIGHT"]
    if not lights:
        return

    light = lights[0]
    rng = random.Random(args.light_seed)
    energy = args.light_energy
    size = args.light_size
    if args.light_energy_jitter:
        energy = max(0.0, energy + rng.uniform(-args.light_energy_jitter, args.light_energy_jitter))
    if args.light_size_jitter:
        size = max(0.01, size + rng.uniform(-args.light_size_jitter, args.light_size_jitter))

    if args.light_location is not None:
        light.location = args.light_location
    elif args.random_light:
        azimuth = math.radians(rng.uniform(0.0, 360.0))
        light.location = (
            args.light_distance * math.sin(azimuth),
            -args.light_distance * math.cos(azimuth),
            args.light_height,
        )

    light.data.energy = energy
    if hasattr(light.data, "size"):
        light.data.size = size
    print(
        "Light setup:",
        {
            "name": light.name,
            "location": [round(v, 6) for v in light.location],
            "energy": round(energy, 6),
            "size": round(size, 6),
        },
    )


def configure_mask_outputs(mask0_dir: Path, mask1_dir: Path) -> None:
    scene = bpy.context.scene
    scene.use_nodes = True

    view_layer = bpy.context.view_layer
    view_layer.use_pass_object_index = True

    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new(type="CompositorNodeRLayers")

    id_mask_0 = tree.nodes.new(type="CompositorNodeIDMask")
    id_mask_0.index = 1
    id_mask_1 = tree.nodes.new(type="CompositorNodeIDMask")
    id_mask_1.index = 2

    mask0_output = tree.nodes.new(type="CompositorNodeOutputFile")
    mask0_output.base_path = str(mask0_dir)
    mask0_output.file_slots[0].path = "frame_####"
    mask0_output.format.file_format = "PNG"
    mask0_output.format.color_mode = "BW"

    mask1_output = tree.nodes.new(type="CompositorNodeOutputFile")
    mask1_output.base_path = str(mask1_dir)
    mask1_output.file_slots[0].path = "frame_####"
    mask1_output.format.file_format = "PNG"
    mask1_output.format.color_mode = "BW"

    tree.links.new(render_layers.outputs["IndexOB"], id_mask_0.inputs["ID value"])
    tree.links.new(id_mask_0.outputs["Alpha"], mask0_output.inputs[0])
    tree.links.new(render_layers.outputs["IndexOB"], id_mask_1.inputs["ID value"])
    tree.links.new(id_mask_1.outputs["Alpha"], mask1_output.inputs[0])


def disable_compositor_outputs() -> None:
    scene = bpy.context.scene
    scene.use_nodes = False
    if scene.node_tree is not None:
        scene.node_tree.nodes.clear()


def main() -> None:
    args = parse_args()
    open_default_blend_if_needed(args.blend_file, enabled=not args.no_open_default_blend)

    base_dir = args.base_dir.expanduser().resolve()
    json_path = base_dir / "physics_metadata.json"
    rgb_dir = base_dir / "render_rgb"
    mask0_dir = base_dir / "masks" / "ball_0"
    mask1_dir = base_dir / "masks" / "ball_1"
    transform_dir = base_dir / "transforms"
    mesh_dir = base_dir / "meshes"

    output_dirs = [rgb_dir]
    if not args.skip_masks:
        output_dirs.extend([mask0_dir, mask1_dir])
    if not args.skip_transforms:
        output_dirs.append(transform_dir)
    if not args.skip_canonical_meshes:
        output_dirs.append(mesh_dir)

    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    with json_path.open("r") as f:
        data = json.load(f)

    if not scene_has_required_balls():
        print("Scene does not contain both ball_0 and ball_1.")

    ball_0, ball_1 = ensure_ball_objects(data, args, create_missing=not args.no_create_missing_balls)
    configure_floor_material(args)
    configure_camera_view(args)
    configure_lighting(args)

    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.fps = data["fps"]

    if args.engine == "CYCLES":
        configure_cycles(args.device, args.samples)
    else:
        scene.render.engine = "BLENDER_EEVEE"

    ball_0.pass_index = 1
    ball_1.pass_index = 2

    if not args.skip_canonical_meshes:
        export_selected_glb(ball_0, mesh_dir / "ball_0.glb")
        export_selected_glb(ball_1, mesh_dir / "ball_1.glb")
    if not args.skip_masks:
        configure_mask_outputs(mask0_dir, mask1_dir)
    else:
        disable_compositor_outputs()

    for frame in data["frames"]:
        idx = frame["frame"]
        p0 = frame["ball_0"]["position"]
        q0 = frame["ball_0"]["quaternion"]
        p1 = frame["ball_1"]["position"]
        q1 = frame["ball_1"]["quaternion"]

        ball_0.location = p0
        ball_0.rotation_mode = "QUATERNION"
        ball_0.rotation_quaternion = [q0[3], q0[0], q0[1], q0[2]]

        ball_1.location = p1
        ball_1.rotation_mode = "QUATERNION"
        ball_1.rotation_quaternion = [q1[3], q1[0], q1[1], q1[2]]

        scene.frame_set(idx)

        if not args.skip_transforms:
            transform_data = {
                "frame": idx,
                "ball_0": {
                    "location": list(ball_0.location),
                    "quaternion_blender_wxyz": list(ball_0.rotation_quaternion),
                    "pass_index": 1,
                },
                "ball_1": {
                    "location": list(ball_1.location),
                    "quaternion_blender_wxyz": list(ball_1.rotation_quaternion),
                    "pass_index": 2,
                },
            }

            with (transform_dir / f"frame_{idx:04d}.json").open("w") as f:
                json.dump(transform_data, f, indent=2)

        scene.render.filepath = str(rgb_dir / f"frame_{idx:04d}.png")
        bpy.ops.render.render(write_still=True)

    print(f"Finished rendering outputs under: {base_dir}")


if __name__ == "__main__":
    main()

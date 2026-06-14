#!/usr/bin/env python3

"""Render RGB frames, object masks, transforms, and GLB meshes in Blender.

Cluster/headless example with the included prepared scene:
  micromamba run -n com4d blender --background datasets/synthetic/two_ball_test/two_ball_scene.blend \
    --python datasets/synthetic/two_ball_test/render_physics_outputs.py -- \
    --base-dir outputs/two_ball_test \
    --device CPU

Cluster/headless example using the script's default scene lookup:
  micromamba run -n com4d blender --background \
    --python datasets/synthetic/two_ball_test/render_physics_outputs.py -- \
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
DEFAULT_BLEND_FILE = SCRIPT_DIR / "physics_scene.blend" if (SCRIPT_DIR / "physics_scene.blend").exists() else SCRIPT_DIR / "two_ball_scene.blend"
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
    parser.add_argument("--camera-focal-length", type=float)
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
    parser.add_argument(
        "--mask-mode",
        choices=("material", "compositor"),
        default="material",
        help="Object-mask writer. material avoids Blender 3.6 compositor OutputFile crashes on some clusters.",
    )
    parser.add_argument("--save-depth", action="store_true", help="Save ground-truth Z depth pass as OpenEXR files.")
    parser.add_argument("--save-normals", action="store_true", help="Save ground-truth normal pass as OpenEXR files.")
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
    if args.camera_focal_length is not None:
        camera.data.lens = args.camera_focal_length
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


def get_or_create_emission_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new(type="ShaderNodeOutputMaterial")
    emission = tree.nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def replace_materials(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if not hasattr(obj.data, "materials"):
        return
    obj.data.materials.clear()
    obj.data.materials.append(material)


def render_material_masks(
    scene: bpy.types.Scene,
    objects: dict[str, bpy.types.Object],
    render_names: list[str],
    dynamic_names: list[str],
    mask_dirs: dict[str, Path],
    frame_idx: int,
) -> None:
    if not dynamic_names:
        return

    original_engine = scene.render.engine
    original_filepath = scene.render.filepath
    original_film_transparent = scene.render.film_transparent
    original_world_color = tuple(scene.world.color) if scene.world is not None else None
    original_materials = {
        name: list(obj.data.materials)
        for name, obj in objects.items()
        if hasattr(obj.data, "materials")
    }
    original_hide_render = {name: obj.hide_render for name, obj in objects.items()}

    black = get_or_create_emission_material("mask_black_emission", (0.0, 0.0, 0.0, 1.0))
    white = get_or_create_emission_material("mask_white_emission", (1.0, 1.0, 1.0, 1.0))

    try:
        scene.render.engine = "BLENDER_EEVEE"
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = 1
        scene.render.film_transparent = False
        if scene.world is not None:
            scene.world.color = (0.0, 0.0, 0.0)

        for name in render_names:
            obj = objects.get(name)
            if obj is not None:
                obj.hide_render = False
                replace_materials(obj, black)

        for target_name in dynamic_names:
            target = objects.get(target_name)
            if target is None:
                continue
            replace_materials(target, white)
            scene.render.filepath = str(mask_dirs[target_name] / f"frame_{frame_idx:04d}.png")
            bpy.ops.render.render(write_still=True)
            replace_materials(target, black)
    finally:
        for name, materials in original_materials.items():
            obj = objects.get(name)
            if obj is None:
                continue
            obj.data.materials.clear()
            for material in materials:
                obj.data.materials.append(material)
        for name, hidden in original_hide_render.items():
            if name in objects:
                objects[name].hide_render = hidden
        if scene.world is not None and original_world_color is not None:
            scene.world.color = original_world_color
        scene.render.film_transparent = original_film_transparent
        scene.render.engine = original_engine
        scene.render.filepath = original_filepath


def material_color(look: str, explicit_color: list[float] | None) -> tuple[float, float, float, float]:
    if explicit_color is not None:
        return tuple(explicit_color)  # type: ignore[return-value]
    return LOOK_COLORS[look]


def floor_color(look: str, explicit_color: list[float] | None) -> tuple[float, float, float, float]:
    if explicit_color is not None:
        return tuple(explicit_color)  # type: ignore[return-value]
    return FLOOR_COLORS[look]


def metadata_object_specs(data: dict) -> dict[str, dict]:
    objects = data.get("objects")
    if isinstance(objects, dict) and objects:
        return objects
    specs = {}
    for name in data.get("render_objects", ["ball_0", "ball_1", "floor"]):
        if name.startswith("ball_"):
            specs[name] = {"type": "sphere", "dynamic": True, "radius": ball_radius(data, name)}
        elif name == "floor":
            specs[name] = {"type": "plane", "dynamic": False}
        elif name in {"wall", "occluder_box"}:
            specs[name] = {"type": "box", "dynamic": False, "size": [1.0, 1.0, 1.0]}
    return specs


def render_object_names(data: dict) -> list[str]:
    names = data.get("render_objects")
    if isinstance(names, list) and names:
        return [name for name in names if isinstance(name, str)]
    return list(metadata_object_specs(data))


def dynamic_object_names(data: dict) -> list[str]:
    specs = metadata_object_specs(data)
    names = [name for name in render_object_names(data) if bool(specs.get(name, {}).get("dynamic", name.startswith("ball_")))]
    return names or [name for name in ("ball_0", "ball_1") if name in specs]


def static_mask_object_names(data: dict) -> list[str]:
    specs = metadata_object_specs(data)
    return [
        name
        for name in render_object_names(data)
        if name != "floor" and not bool(specs.get(name, {}).get("dynamic", name.startswith("ball_")))
    ]


def frame_state_map(frame: dict) -> dict:
    objects = frame.get("objects")
    return objects if isinstance(objects, dict) else frame


def object_material(name: str, args: argparse.Namespace) -> bpy.types.Material:
    if name == "ball_0":
        color = material_color(args.ball_0_look, args.ball_0_color)
    elif name == "ball_1":
        color = material_color(args.ball_1_look, args.ball_1_color)
    elif name == "wall":
        color = (0.48, 0.48, 0.48, 1.0)
    elif name == "occluder_box":
        color = (0.35, 0.35, 0.35, 1.0)
    else:
        color = LOOK_COLORS.get("green", (0.1, 0.75, 0.25, 1.0))
    return get_or_create_material(f"{name}_material", color, args.material_roughness)


def ensure_box_object(name: str, spec: dict) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    size = spec.get("size")
    if size is None and "half_extents" in spec:
        size = [2.0 * float(value) for value in spec["half_extents"]]
    if size is None:
        size = [1.0, 1.0, 1.0]
    if obj is None:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=spec.get("position", [0.0, 0.0, 0.5]))
        obj = bpy.context.object
        obj.name = name
    obj.dimensions = tuple(float(value) for value in size)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)
    return obj


def ensure_render_objects(data: dict, args: argparse.Namespace, create_missing: bool) -> dict[str, bpy.types.Object]:
    specs = metadata_object_specs(data)
    objects = {}
    if create_missing:
        remove_default_cube()
    for name in render_object_names(data):
        spec = specs.get(name, {})
        object_type = str(spec.get("type", "sphere" if name.startswith("ball_") else "box")).lower()
        if name == "floor" or object_type == "plane":
            obj = ensure_floor()
        elif object_type in {"sphere", "ball"}:
            obj = bpy.data.objects.get(name)
            radius = float(spec.get("radius", ball_radius(data, name)))
            if obj is None:
                if not create_missing:
                    raise KeyError(f"Missing required Blender object: {name}")
                bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=radius, location=(0.0, 0.0, radius))
                obj = bpy.context.object
                obj.name = name
            else:
                set_object_radius(obj, radius)
            obj.data.materials.clear()
            obj.data.materials.append(object_material(name, args))
        elif object_type in {"box", "cube", "cuboid"}:
            obj = ensure_box_object(name, spec)
            obj.data.materials.clear()
            obj.data.materials.append(object_material(name, args))
        else:
            obj = bpy.data.objects.get(name)
            if obj is None:
                if not create_missing:
                    raise KeyError(f"Missing required Blender object: {name}")
                continue
        objects[name] = obj
    if create_missing:
        ensure_camera_and_light()
    return objects


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


def configure_compositor_outputs(
    mask_dirs: dict[str, Path],
    depth_dir: Path | None,
    normal_dir: Path | None,
) -> None:
    scene = bpy.context.scene
    scene.use_nodes = True

    view_layer = bpy.context.view_layer
    view_layer.use_pass_object_index = bool(mask_dirs)
    view_layer.use_pass_z = depth_dir is not None
    view_layer.use_pass_normal = normal_dir is not None

    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new(type="CompositorNodeRLayers")

    for name, mask_dir in sorted(mask_dirs.items()):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.pass_index <= 0:
            continue
        id_mask = tree.nodes.new(type="CompositorNodeIDMask")
        id_mask.index = obj.pass_index
        mask_output = tree.nodes.new(type="CompositorNodeOutputFile")
        mask_output.base_path = str(mask_dir)
        mask_output.file_slots[0].path = "frame_####"
        mask_output.format.file_format = "PNG"
        mask_output.format.color_mode = "BW"
        tree.links.new(render_layers.outputs["IndexOB"], id_mask.inputs["ID value"])
        tree.links.new(id_mask.outputs["Alpha"], mask_output.inputs[0])

    if depth_dir is not None:
        depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_output.base_path = str(depth_dir)
        depth_output.file_slots[0].path = "frame_####"
        depth_output.format.file_format = "OPEN_EXR"
        depth_output.format.color_mode = "RGB"
        depth_output.format.color_depth = "32"
        tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])

    if normal_dir is not None:
        normal_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_output.base_path = str(normal_dir)
        normal_output.file_slots[0].path = "frame_####"
        normal_output.format.file_format = "OPEN_EXR"
        normal_output.format.color_mode = "RGB"
        normal_output.format.color_depth = "32"
        tree.links.new(render_layers.outputs["Normal"], normal_output.inputs[0])


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
    depth_dir = base_dir / "depth"
    normal_dir = base_dir / "normals"
    transform_dir = base_dir / "transforms"
    mesh_dir = base_dir / "meshes"

    with json_path.open("r") as f:
        data = json.load(f)

    render_names = render_object_names(data)
    dynamic_names = dynamic_object_names(data)
    mask_names = dynamic_names + static_mask_object_names(data)
    mask_dirs = {name: base_dir / "masks" / name for name in mask_names}

    output_dirs = [rgb_dir]
    if not args.skip_masks:
        output_dirs.extend(mask_dirs.values())
    if args.save_depth:
        output_dirs.append(depth_dir)
    if args.save_normals:
        output_dirs.append(normal_dir)
    if not args.skip_transforms:
        output_dirs.append(transform_dir)
    if not args.skip_canonical_meshes:
        output_dirs.append(mesh_dir)

    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    missing = [name for name in render_names if bpy.data.objects.get(name) is None and name != "floor"]
    if missing:
        print(f"Scene is missing render objects that will be created if allowed: {missing}")

    objects = ensure_render_objects(data, args, create_missing=not args.no_create_missing_balls)
    keep_names = set(render_names)
    for obj in bpy.data.objects:
        keep = obj.name in keep_names or obj.type in {"CAMERA", "LIGHT"}
        obj.hide_render = not keep
        obj.hide_viewport = not keep
        if obj.name not in mask_names:
            obj.pass_index = 0
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

    for pass_index, name in enumerate(mask_names, start=1):
        if name in objects:
            objects[name].pass_index = pass_index

    if not args.skip_canonical_meshes:
        for name in render_names:
            obj = objects.get(name)
            if obj is not None and name != "floor":
                export_selected_glb(obj, mesh_dir / f"{name}.glb")
    compositor_mask_dirs = mask_dirs if (not args.skip_masks and args.mask_mode == "compositor") else {}
    if compositor_mask_dirs or args.save_depth or args.save_normals:
        configure_compositor_outputs(
            mask_dirs=compositor_mask_dirs,
            depth_dir=depth_dir if args.save_depth else None,
            normal_dir=normal_dir if args.save_normals else None,
        )
    else:
        disable_compositor_outputs()

    for frame in data["frames"]:
        idx = frame["frame"]
        states = frame_state_map(frame)
        for name, state in states.items():
            obj = objects.get(name)
            if obj is None or not isinstance(state, dict):
                continue
            pos = state.get("position")
            quat = state.get("quaternion")
            if pos is None or quat is None:
                continue
            obj.location = pos
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = [quat[3], quat[0], quat[1], quat[2]]

        scene.frame_set(idx)

        if not args.skip_transforms:
            transform_data = {"frame": idx, "objects": {}}
            for name in render_names:
                obj = objects.get(name)
                if obj is None:
                    continue
                item = {
                    "location": list(obj.location),
                    "quaternion_blender_wxyz": list(obj.rotation_quaternion),
                    "pass_index": obj.pass_index,
                }
                transform_data["objects"][name] = item
                transform_data[name] = item

            with (transform_dir / f"frame_{idx:04d}.json").open("w") as f:
                json.dump(transform_data, f, indent=2)

        scene.render.filepath = str(rgb_dir / f"frame_{idx:04d}.png")
        bpy.ops.render.render(write_still=True)

        if not args.skip_masks and args.mask_mode == "material":
            render_material_masks(scene, objects, render_names, mask_names, mask_dirs, idx)

    print(f"Finished rendering outputs under: {base_dir}")


if __name__ == "__main__":
    main()

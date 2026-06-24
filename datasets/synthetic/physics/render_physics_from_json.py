import bpy
import json
from pathlib import Path
import sys
import mathutils

# =====================
# Required CLI argument
# =====================
if "--" not in sys.argv:
    raise RuntimeError(
        "No BASE_DIR provided. Run Blender like:\n"
        "blender -b physics_scene.blend -P render_physics_from_json.py -- outputs/ball_drop_test"
    )

args = sys.argv[sys.argv.index("--") + 1:]

if len(args) < 1:
    raise RuntimeError(
        "Missing BASE_DIR after '--'. Example:\n"
        "blender -b physics_scene.blend -P render_physics_from_json.py -- outputs/ball_drop_test"
    )

BASE_DIR = Path(args[0])

if not BASE_DIR.is_absolute():
    BASE_DIR = Path(bpy.path.abspath(f"//{BASE_DIR.as_posix()}"))

BASE_DIR = BASE_DIR.resolve()

if not BASE_DIR.exists():
    raise FileNotFoundError(f"BASE_DIR does not exist: {BASE_DIR}")

JSON_PATH = BASE_DIR / "physics_metadata.json"
RGB_DIR = BASE_DIR / "render_rgb"
MASK_DIR = BASE_DIR / "masks"
TRANSFORM_DIR = BASE_DIR / "transforms"
MESH_DIR = BASE_DIR / "meshes"

for d in [RGB_DIR, MASK_DIR, TRANSFORM_DIR, MESH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

# =====================
# Scenario-specific camera presets
# =====================
CAMERA_PRESETS = {
    "two_ball_collision": {
        "location": [0.0, -5.0, 2.0],
        "target": [0.0, 0.0, 0.35],
        "focal_length": 35,
    },
    "ball_drop": {
        "location": [0.8, -6.0, 2.6],
        "target": [0.6, 0.0, 1.1],
        "focal_length": 28,
    },
    "rolling_occluder": {
        "location": [0.0, -5.5, 1.6],
        "target": [0.0, -0.4, 0.5],
        "focal_length": 35,
    },
    "wall_impact": {
        "location": [0.5, -6.0, 1.8],
        "target": [0.8, 0.0, 0.5],
        "focal_length": 30,
    },
}


def look_at(obj, target):
    loc = mathutils.Vector(obj.location)
    target = mathutils.Vector(target)
    direction = target - loc
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def apply_camera_preset(scenario):
    if "Camera" not in bpy.data.objects:
        raise RuntimeError("No object named 'Camera' found in the Blender scene.")

    cam = bpy.data.objects["Camera"]
    preset = CAMERA_PRESETS.get(scenario)

    if preset is None:
        raise RuntimeError(
            f"No camera preset defined for scenario '{scenario}'. "
            f"Available presets: {list(CAMERA_PRESETS.keys())}"
        )

    cam.location = preset["location"]
    look_at(cam, preset["target"])
    cam.data.lens = preset["focal_length"]
    cam.data.sensor_width = 32

    bpy.context.scene.camera = cam

    print(f"Applied camera preset for scenario: {scenario}")


apply_camera_preset(data["scenario"])

scene = bpy.context.scene
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.fps = data["fps"]

scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.cycles.device = "GPU"

render_objects = data["render_objects"]

# Hide everything except render_objects, camera, lights
always_keep = {"Camera", "Area"}
for obj in bpy.data.objects:
    should_render = obj.name in render_objects or obj.name in always_keep
    obj.hide_render = not should_render
    obj.hide_viewport = not should_render

# Assign pass indices dynamically
view_layer = bpy.context.view_layer
view_layer.use_pass_object_index = True

object_pass_indices = {}
next_index = 1
for name in render_objects:
    if name in bpy.data.objects and name not in {"floor", "wall", "occluder_box"}:
        bpy.data.objects[name].pass_index = next_index
        object_pass_indices[name] = next_index
        next_index += 1

# Export GLBs for render objects
def export_selected_glb(obj, out_path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )

for name in render_objects:
    if name in bpy.data.objects:
        export_selected_glb(bpy.data.objects[name], MESH_DIR / f"{name}.glb")

# Compositor mask outputs for dynamic objects
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()

render_layers = tree.nodes.new(type="CompositorNodeRLayers")

for name, pass_idx in object_pass_indices.items():
    obj_mask_dir = MASK_DIR / name
    obj_mask_dir.mkdir(parents=True, exist_ok=True)

    id_mask = tree.nodes.new(type="CompositorNodeIDMask")
    id_mask.index = pass_idx

    out_node = tree.nodes.new(type="CompositorNodeOutputFile")
    out_node.base_path = str(obj_mask_dir)
    out_node.file_slots[0].path = "frame_####"
    out_node.format.file_format = "PNG"
    out_node.format.color_mode = "BW"

    tree.links.new(render_layers.outputs["IndexOB"], id_mask.inputs["ID value"])
    tree.links.new(id_mask.outputs["Alpha"], out_node.inputs[0])

# Render loop
for frame in data["frames"]:
    idx = frame["frame"]

    frame_objects = frame["objects"]

    for name, state in frame_objects.items():
        if name not in bpy.data.objects:
            continue

        obj = bpy.data.objects[name]
        pos = state["position"]
        quat = state["quaternion"]  # PyBullet xyzw

        obj.location = pos
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = [quat[3], quat[0], quat[1], quat[2]]

    scene.frame_set(idx)

    transform_data = {
        "frame": idx,
        "objects": {},
    }

    for name in render_objects:
        if name not in bpy.data.objects:
            continue

        obj = bpy.data.objects[name]
        transform_data["objects"][name] = {
            "location": list(obj.location),
            "quaternion_blender_wxyz": list(obj.rotation_quaternion),
            "pass_index": object_pass_indices.get(name, 0),
        }

    with open(TRANSFORM_DIR / f"frame_{idx:04d}.json", "w") as f:
        json.dump(transform_data, f, indent=2)

    scene.render.filepath = str(RGB_DIR / f"frame_{idx:04d}.png")
    bpy.ops.render.render(write_still=True)

print("Finished rendering scenario.")
#!/usr/bin/env python3
"""Render shaded latent-refinement meshes with pyrender's default backend."""

import argparse
import json
import shlex
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh

# Import this before pyrender so COM4D selects OSMesa/EGL/GLX consistently.
from src.utils.render_utils import export_renderings, render_sequence_fixed_camera
import pyrender


COLORS = {
    "reference": (215, 91, 78, 255),
    "shared_optimized": (82, 170, 220, 255),
    "memory_fitted": (45, 136, 117, 255),
    "ball_0": (245, 180, 55, 255),
    "floor": (145, 145, 145, 255),
    "occluder_box": (170, 80, 190, 255),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("dataset_json/physics_train_100_static.json"))
    p.add_argument("--render-size", type=int, default=384)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--azimuth", type=float, default=-20.0)
    p.add_argument("--elevation", type=float, default=15.0)
    p.add_argument("--fit-scale", type=float, default=2.2)
    p.add_argument("--rotation-step-degrees", type=float, default=5.0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--prediction-overlay-alpha", type=float, default=0.48)
    p.add_argument("--pipeline-log", type=Path, default=None,
                   help="Original raw sequence pipeline.log; inferred for processed physics data.")
    return p.parse_args()


def rotation_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    x, y, z, w = q
    return np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def pose_transform(frame, index):
    value = np.eye(4)
    value[:3, :3] = rotation_matrix(frame["object_quaternion_xyzw"][index])
    value[:3, 3] = np.asarray(frame["object_translation"][index], dtype=np.float64)
    return value


def colored(mesh, rgba):
    output = mesh.copy()
    output.visual.vertex_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(output.vertices), 1))
    return output


def load_mesh(path, rgba):
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return colored(mesh, rgba)


def local_gt_mesh(name, part, frame, index, rgba):
    points = np.asarray(part["surface_points"], dtype=np.float64)
    transform = pose_transform(frame, index)
    local = (points - transform[:3, 3][None]) @ transform[:3, :3]
    lo, hi = local.min(0), local.max(0)
    center = 0.5 * (lo + hi)
    extents = hi - lo

    if name == "floor":
        # Exactly two triangles; their shared diagonal remains visible in wireframe mode.
        x0, y0, z = lo[0], lo[1], center[2]
        x1, y1 = hi[0], hi[1]
        mesh = trimesh.Trimesh(
            vertices=[[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]],
            faces=[[0, 1, 2], [0, 2, 3]], process=False,
        )
    elif "box" in name:
        # Minimal triangulated cuboid: six sides, two triangles per side.
        mesh = trimesh.creation.box(extents=np.maximum(extents, 1e-4))
        mesh.apply_translation(center)
    elif name.startswith("ball"):
        # Low-complexity sphere (80 triangular faces / 120 unique edges).
        radius = float(np.median(np.linalg.norm(local - center[None], axis=1)))
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        mesh.apply_translation(center)
    else:
        mesh = trimesh.points.PointCloud(local).convex_hull
    return colored(mesh, rgba)


def scene_for_frame(local_meshes, frame, names):
    scene = trimesh.Scene()
    for name, mesh in local_meshes.items():
        item = mesh.copy()
        item.apply_transform(pose_transform(frame, names.index(name)))
        scene.add_geometry(item, geom_name=name)
    return scene


def rotate_scene(scene, center, degrees):
    output = scene.copy()
    output.apply_transform(trimesh.transformations.rotation_matrix(
        np.deg2rad(degrees), [0, 0, 1], point=center
    ))
    return output


def panel(image, label, size):
    image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 30), "white")
    canvas.paste(image, (0, 30))
    ImageDraw.Draw(canvas).text((8, 8), label, fill="black")
    return canvas


def compose(images, labels, size):
    items = [panel(image, label, size) for image, label in zip(images, labels)]
    output = Image.new("RGB", (size * len(items), size + 30), "white")
    for index, item in enumerate(items):
        output.paste(item, (index * size, 0))
    return output


def fade_prediction(image, alpha):
    white = Image.new("RGB", image.size, "white")
    return Image.blend(white, image.convert("RGB"), max(0.0, min(1.0, alpha)))


def overlay_wireframe(prediction, wireframe, alpha):
    faded = fade_prediction(prediction, alpha)
    wire = np.asarray(wireframe.convert("RGB"), dtype=np.uint8)
    mask = Image.fromarray((255 - wire.min(axis=2)).astype(np.uint8))
    return Image.composite(wireframe.convert("RGB"), faded, mask)


def parse_pipeline_camera(path):
    lines = path.read_text().splitlines()
    command = next((line for line in lines if "run_physics_pipeline.py " in line), None)
    if command is None:
        raise ValueError(f"No run_physics_pipeline.py command in {path}")
    tokens = shlex.split(command)
    def values(flag, count=1):
        index = tokens.index(flag)
        raw = tokens[index + 1:index + 1 + count]
        return [float(value) for value in raw]
    target = np.asarray(values("--camera-target", 3))
    distance = values("--camera-distance")[0]
    height = values("--camera-height")[0]
    azimuth = np.deg2rad(values("--camera-azimuth")[0])
    focal = values("--camera-focal-length")[0]
    location = np.asarray([
        target[0] + distance * np.sin(azimuth),
        target[1] - distance * np.cos(azimuth),
        target[2] + height,
    ])
    backward = location - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray([0.0, 0.0, 1.0]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, backward, location
    resolution = 512
    fx = focal * resolution / 36.0
    return {
        "camera_to_world": pose.tolist(),
        "intrinsics": [[fx, 0.0, resolution / 2], [0.0, fx, resolution / 2], [0.0, 0.0, 1.0]],
        "resolution": [resolution, resolution],
        "clip_start": 0.1,
        "clip_end": 1000.0,
    }


def render(scenes, cfg, flags=pyrender.constants.RenderFlags.NONE, camera_metadata=None):
    return render_sequence_fixed_camera(
        scenes, azimuth=cfg.azimuth, elevation=cfg.elevation,
        fit_scale=cfg.fit_scale, image_size=(cfg.render_size, cfg.render_size),
        light_intensity=5.0, return_type="pil", bg_color=(255, 255, 255, 255),
        flags=flags,
        camera_metadata=camera_metadata,
        pred_to_gt_transform=np.eye(4) if camera_metadata is not None else None,
    )


def main():
    cfg = parse_args()
    metrics = json.loads((cfg.experiment_dir / "metrics.json").read_text())
    manifest = json.loads(cfg.manifest.read_text())
    frames = manifest[metrics["sequence"]]
    names = frames[0]["object_names"]
    target_name = metrics["object"]
    target_index = names.index(target_name)
    indices = list(range(0, len(frames), max(cfg.frame_stride, 1)))
    if cfg.max_frames > 0:
        indices = indices[:cfg.max_frames]

    pipeline_log = cfg.pipeline_log
    if pipeline_log is None:
        pipeline_log = Path(
            f"/mnt/mocap_b/work/com4d/datasets/processed/physics/raw_train_100/{metrics['sequence']}/pipeline.log"
        )
    camera_metadata = parse_pipeline_camera(pipeline_log)

    predictions = {
        name: load_mesh(cfg.experiment_dir / f"{name}.glb", COLORS[name])
        for name in ("reference", "shared_optimized", "memory_fitted")
    }
    first_data = np.load(frames[0]["surface_path"], allow_pickle=True).item()
    gt_meshes = {
        name: local_gt_mesh(name, first_data["parts"][index], frames[0], index, (0, 210, 235, 255))
        for index, name in enumerate(names)
    }

    method_scenes = {
        method: [
            scene_for_frame({target_name: prediction}, frames[frame_index], names)
            for frame_index in indices
        ]
        for method, prediction in predictions.items()
    }
    gt_scenes = [
        scene_for_frame(gt_meshes, frames[frame_index], names) for frame_index in indices
    ]

    rendered = {
        name: render(scenes, cfg, camera_metadata=camera_metadata)
        for name, scenes in method_scenes.items()
    }
    fixed_wireframes = render(
        gt_scenes, cfg, flags=pyrender.constants.RenderFlags.ALL_WIREFRAME,
        camera_metadata=camera_metadata,
    )
    rendered_with_wireframes = {
        name: [
            overlay_wireframe(frame, fixed_wireframes[i], 1.0)
            for i, frame in enumerate(method_frames)
        ]
        for name, method_frames in rendered.items()
    }
    source = [Image.open(frames[index]["image_path"]).convert("RGB") for index in indices]
    diagnostic = [
        compose(
            [source[i], rendered_with_wireframes["reference"][i],
             rendered_with_wireframes["shared_optimized"][i],
             rendered_with_wireframes["memory_fitted"][i]],
            ["Input RGB", "Reference latent + GT wireframes",
             "Shared optimized + GT wireframes", "Memory fitted + GT wireframes"],
            cfg.render_size,
        )
        for i in range(len(indices))
    ]

    # Overlay only the shaded memory ball; all scene geometry comes from cyan GT wireframes.
    memory_only_scenes = [
        scene_for_frame({target_name: predictions["memory_fitted"]}, frames[frame_index], names)
        for frame_index in indices
    ]

    all_bounds = np.asarray([scene.bounds for scene in gt_scenes])
    center = 0.5 * (all_bounds[:, 0].min(0) + all_bounds[:, 1].max(0))
    rotating_prediction_scenes = [
        rotate_scene(scene, center, i * cfg.rotation_step_degrees)
        for i, scene in enumerate(memory_only_scenes)
    ]
    rotating_gt_scenes = [
        rotate_scene(scene, center, i * cfg.rotation_step_degrees)
        for i, scene in enumerate(gt_scenes)
    ]
    rotating_predictions = render(
        rotating_prediction_scenes, cfg, camera_metadata=camera_metadata
    )
    rotating_wireframes = render(
        rotating_gt_scenes, cfg, flags=pyrender.constants.RenderFlags.ALL_WIREFRAME,
        camera_metadata=camera_metadata,
    )
    rotating_overlay = [
        overlay_wireframe(rotating_predictions[i], rotating_wireframes[i], cfg.prediction_overlay_alpha)
        for i in range(len(indices))
    ]

    outputs = {
        "animation_diagnostic.gif": diagnostic,
        "animation_gt_overlay.gif": rotating_overlay,
    }

    for filename, images in outputs.items():
        path = cfg.experiment_dir / filename
        export_renderings(images, str(path), fps=cfg.fps)
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render software-only GIF diagnostics for latent-refinement outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from src.utils.render_utils import export_renderings


COLORS = {
    "reference": (215, 91, 78),
    "shared_optimized": (82, 170, 220),
    "memory_fitted": (45, 136, 117),
    "ball_0": (245, 180, 55),
    "floor": (110, 110, 110),
    "occluder_box": (170, 80, 190),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("dataset_json/physics_train_100_static.json"))
    p.add_argument("--render-size", type=int, default=384)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--azimuth", type=float, default=-20.0)
    p.add_argument("--elevation", type=float, default=15.0)
    p.add_argument("--rotation-step-degrees", type=float, default=5.0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--surface-points", type=int, default=5000)
    p.add_argument("--max-wire-edges", type=int, default=1800)
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


def load_points(path, count, seed):
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=np.random.default_rng(seed))
    return np.asarray(points, dtype=np.float64)


def world_points(local, frame, object_index):
    rotation = rotation_matrix(frame["object_quaternion_xyzw"][object_index])
    translation = np.asarray(frame["object_translation"][object_index], dtype=np.float64)
    return local @ rotation.T + translation[None]


def view_rotation(azimuth, elevation):
    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    rz = np.asarray([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    rx = np.asarray([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return rz @ rx


def projected(points, rotation):
    value = points @ rotation
    return value[:, [0, 2, 1]]


def square_bounds(clouds, margin_fraction=0.08):
    xy = np.concatenate([cloud[:, :2] for cloud in clouds], axis=0)
    lo, hi = xy.min(0), xy.max(0)
    center = 0.5 * (lo + hi)
    side = max(float((hi - lo).max()), 1e-6)
    side *= 1 + 2 * margin_fraction
    return center - side / 2, center + side / 2


def image_xy(cloud, bounds, size):
    lo, hi = bounds
    scale = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    x = (cloud[:, 0] - lo[0]) / scale * (size - 24) + 12
    y = size - ((cloud[:, 1] - lo[1]) / scale * (size - 24) + 12)
    return x, y


def draw_cloud(clouds, colors, bounds, size, wireframes=(), label=None):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for vertices, edges, color in wireframes:
        x, y = image_xy(vertices, bounds, size)
        for left, right in edges:
            draw.line((x[left], y[left], x[right], y[right]), fill=color, width=1)
    entries = []
    for cloud, color in zip(clouds, colors):
        x, y = image_xy(cloud, bounds, size)
        entries.extend((float(depth), float(px), float(py), color)
                       for px, py, depth in zip(x, y, cloud[:, 2]))
    for _, x, y, color in sorted(entries, reverse=True):
        draw.ellipse((x - 1.1, y - 1.1, x + 1.1, y + 1.1), fill=color)
    if label:
        draw.text((8, 8), label, fill="black")
    return image


def panel(image, label, size):
    image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 30), "white")
    canvas.paste(image, (0, 30))
    ImageDraw.Draw(canvas).text((8, 8), label, fill="black")
    return canvas


def compose(images, labels, size):
    panels = [panel(image, label, size) for image, label in zip(images, labels)]
    canvas = Image.new("RGB", (size * len(panels), size + 30), "white")
    for index, item in enumerate(panels):
        canvas.paste(item, (index * size, 0))
    return canvas


def local_gt_mesh(part, translation, quaternion):
    points = np.asarray(part["surface_points"], dtype=np.float64)
    rotation = rotation_matrix(quaternion)
    local = (points - np.asarray(translation)[None]) @ rotation
    extents = np.ptp(local, axis=0)
    if extents.min() < max(extents.max() * 1e-3, 1e-5):
        center = 0.5 * (local.min(0) + local.max(0))
        extents = np.maximum(extents, max(extents.max() * 0.005, 0.01))
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        return mesh
    return trimesh.points.PointCloud(local).convex_hull


def sampled_edges(mesh, maximum, seed):
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) > maximum:
        rng = np.random.default_rng(seed)
        edges = edges[rng.choice(len(edges), maximum, replace=False)]
    return edges


def main():
    cfg = parse_args()
    metrics = json.loads((cfg.experiment_dir / "metrics.json").read_text())
    manifest = json.loads(cfg.manifest.read_text())
    frames = manifest[metrics["sequence"]]
    names = frames[0]["object_names"]
    object_index = names.index(metrics["object"])
    indices = list(range(0, len(frames), max(cfg.frame_stride, 1)))

    local_predictions = {
        name: load_points(cfg.experiment_dir / f"{name}.glb", cfg.surface_points, index)
        for index, name in enumerate(("reference", "shared_optimized", "memory_fitted"))
    }

    first_data = np.load(frames[0]["surface_path"], allow_pickle=True).item()
    local_gt = {}
    gt_edges = {}
    for index, name in enumerate(names):
        mesh = local_gt_mesh(
            first_data["parts"][index],
            frames[0]["object_translation"][index],
            frames[0]["object_quaternion_xyzw"][index],
        )
        local_gt[name] = np.asarray(mesh.vertices, dtype=np.float64)
        gt_edges[name] = sampled_edges(mesh, cfg.max_wire_edges, index)

    fixed_rotation = view_rotation(cfg.azimuth, cfg.elevation)
    prediction_sequences = {
        name: [projected(world_points(points, frames[i], object_index), fixed_rotation) for i in indices]
        for name, points in local_predictions.items()
    }
    fixed_gt = {
        name: [projected(world_points(vertices, frames[i], names.index(name)), fixed_rotation) for i in indices]
        for name, vertices in local_gt.items()
    }
    fixed_bounds = square_bounds([
        cloud for sequence in fixed_gt.values() for cloud in sequence
    ])

    diagnostic = []
    fixed_overlays = []
    rotating_overlays = []
    rotating_clouds_for_bounds = []
    rotating_data = []
    for position, frame_index in enumerate(indices):
        angle = cfg.azimuth + position * cfg.rotation_step_degrees
        rotation = view_rotation(angle, cfg.elevation)
        pred = projected(
            world_points(local_predictions["memory_fitted"], frames[frame_index], object_index),
            rotation,
        )
        wires = {
            name: projected(world_points(vertices, frames[frame_index], names.index(name)), rotation)
            for name, vertices in local_gt.items()
        }
        rotating_data.append((pred, wires))
        rotating_clouds_for_bounds.extend(wires.values())
    rotating_bounds = square_bounds(rotating_clouds_for_bounds)

    for position, frame_index in enumerate(indices):
        source = Image.open(frames[frame_index]["image_path"]).convert("RGB")
        method_images = [
            draw_cloud([prediction_sequences[name][position]], [COLORS[name]],
                       fixed_bounds, cfg.render_size)
            for name in ("reference", "shared_optimized", "memory_fitted")
        ]
        diagnostic.append(compose(
            [source, *method_images],
            ["Input RGB", "Reference latent", "Shared optimized", "Memory fitted"],
            cfg.render_size,
        ))

        fixed_wireframes = [
            (fixed_gt[name][position], gt_edges[name], COLORS.get(name, (40, 40, 40)))
            for name in names
        ]
        fixed_image = draw_cloud(
            [prediction_sequences["memory_fitted"][position]],
            [COLORS["memory_fitted"]], fixed_bounds, cfg.render_size,
            wireframes=fixed_wireframes,
            label="GT wireframes: ball=yellow, floor=gray, occluder=purple",
        )
        fixed_overlays.append(compose(
            [source, fixed_image], ["Input RGB", "Memory + full-scene GT wireframes"],
            cfg.render_size,
        ))

        rotating_pred, rotating_wires = rotating_data[position]
        rotating_wireframes = [
            (rotating_wires[name], gt_edges[name], COLORS.get(name, (40, 40, 40)))
            for name in names
        ]
        rotating_image = draw_cloud(
            [rotating_pred], [COLORS["memory_fitted"]], rotating_bounds, cfg.render_size,
            wireframes=rotating_wireframes,
            label=f"Rotating view: {cfg.azimuth + position * cfg.rotation_step_degrees:.1f} deg",
        )
        rotating_overlays.append(compose(
            [source, rotating_image], ["Input RGB", "Rotating memory + GT wireframes"],
            cfg.render_size,
        ))

    outputs = {
        "animation_diagnostic.gif": diagnostic,
        "animation_gt_overlay.gif": fixed_overlays,
        "animation_rotating_overlay.gif": rotating_overlays,
    }
    for filename, images in outputs.items():
        path = cfg.experiment_dir / filename
        export_renderings(images, str(path), fps=cfg.fps)
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render animation.gif files from already-exported COM4D dynamic GLBs.

This is a visualization backfill utility. It does not run inference and does not
need checkpoint paths. It scans prediction export directories containing
``dynamic/dynamic_scene_frame_*.glb`` and renders those frame GLBs with the same
fixed-camera helper used by inference_com4d.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils.render_utils import (
    export_renderings,
    load_camera_metadata,
    load_pred_to_gt_transform,
    render_sequence_fixed_camera,
)
import pyrender


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill animation.gif from exported dynamic GLB frames.")
    ap.add_argument(
        "--export-dir",
        type=Path,
        action="append",
        default=None,
        help="Render one explicit prediction export directory. Can be repeated and bypasses root sample/model discovery.",
    )
    ap.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("outputs/evaluation/physics_compare/predictions"),
        help="Root containing <sample>/<model>/ prediction folders.",
    )
    ap.add_argument("--sample", action="append", default=None, help="Sample name to include. Can be repeated.")
    ap.add_argument("--model", action="append", default=None, help="Model tag to include. Can be repeated.")
    ap.add_argument("--exclude-sample", action="append", default=[], help="Sample name to skip. Can be repeated.")
    ap.add_argument("--output-name", default="animation.gif", help="GIF filename written inside each prediction export dir.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite GIFs that already exist.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned renders without loading GLBs or writing GIFs.")
    ap.add_argument("--render-size", type=int, default=512, help="Square render size in pixels.")
    ap.add_argument("--fps", type=int, default=None, help="GIF frames per second. Defaults to GT metadata fps when available, otherwise 18.")
    ap.add_argument(
        "--source-frames-dir",
        type=Path,
        default=None,
        help="Optional source video frames directory to show on the left side of each GIF.",
    )
    ap.add_argument(
        "--no-side-by-side",
        action="store_true",
        help="Write rendered animation only, without source frames on the left.",
    )
    ap.add_argument("--azimuth", type=float, default=35.0, help="Fixed camera azimuth in degrees.")
    ap.add_argument("--elevation", type=float, default=20.0, help="Fixed camera elevation in degrees.")
    ap.add_argument("--fit-scale", type=float, default=2.2, help="Camera distance multiplier for global sequence bounds.")
    ap.add_argument("--camera-metadata", type=Path, default=None, help="Single physics_metadata.json (or camera JSON) used for every selected export.")
    ap.add_argument("--camera-metadata-root", type=Path, default=None, help="Root containing <sample>/physics_metadata.json files.")
    ap.add_argument("--alignment-metadata", type=Path, default=None, help="Single reconstruction metrics JSON containing pred_to_gt_transform.")
    ap.add_argument("--alignment-metrics-root", type=Path, default=None, help="Root containing <sample>/<model>/reconstruction/metrics.json files.")
    ap.add_argument("--gt-geometry-root", type=Path, default=None, help="GT case directory containing meshes/, transforms/, and physics_metadata.json. Defaults to the camera metadata parent.")
    ap.add_argument("--diagnostic-output-name", default="animation_diagnostic.gif")
    ap.add_argument("--gt-overlay-output-name", default="animation_gt_overlay.gif")
    ap.add_argument("--rotation-step-degrees", type=float, default=5.0, help="Viewpoint rotation per frame for rotating diagnostic and GT-overlay views.")
    ap.add_argument("--prediction-overlay-alpha", type=float, default=0.48, help="Prediction opacity in the GT-wireframe overlay GIF.")
    ap.add_argument("--no-diagnostic-artifacts", action="store_true", help="Only write --output-name, without diagnostic or GT-overlay artifacts.")
    ap.add_argument("--max-frames", type=int, default=0, help="Optional cap on rendered frames per GIF; 0 renders all frames.")
    ap.add_argument("--frame-stride", type=int, default=1, help="Render every Nth dynamic frame.")
    return ap.parse_args()


def has_dynamic_frames(export_dir: Path) -> bool:
    return any((export_dir / "dynamic").glob("dynamic_scene_frame_*.glb"))


def iter_model_dirs(predictions_root: Path, samples: set[str] | None, models: set[str] | None, excluded: set[str]) -> Iterable[Path]:
    sample_dirs = sorted(p for p in predictions_root.iterdir() if p.is_dir())
    for sample_dir in sample_dirs:
        if sample_dir.name in excluded:
            continue
        if samples is not None and sample_dir.name not in samples:
            continue
        for model_dir in sorted(p for p in sample_dir.iterdir() if p.is_dir()):
            if models is not None and model_dir.name not in models:
                continue
            yield model_dir


def iter_export_dirs(model_dir: Path) -> Iterable[Path]:
    if has_dynamic_frames(model_dir):
        yield model_dir
    for child in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        if has_dynamic_frames(child):
            yield child


def load_scene(path: Path) -> trimesh.Scene:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    if isinstance(loaded, trimesh.Trimesh):
        return trimesh.Scene(loaded)
    raise TypeError(f"Unsupported GLB load result for {path}: {type(loaded)!r}")


def parse_frame_index(path: Path) -> int | None:
    stem = path.stem
    digits = ""
    for char in reversed(stem):
        if not char.isdigit():
            break
        digits = char + digits
    return int(digits) if digits else None


def discover_source_frames_dir(export_dir: Path, args: argparse.Namespace, gt_root: Path | None = None) -> Path | None:
    if args.source_frames_dir is not None:
        return args.source_frames_dir.expanduser().resolve()

    if gt_root is not None:
        gt_rgb_dir = gt_root / "render_rgb"
        if not gt_rgb_dir.is_dir():
            raise FileNotFoundError(f"GT video frames not found: {gt_rgb_dir}")
        return gt_rgb_dir

    args_path = export_dir / "args.json"
    if not args_path.is_file():
        return None
    try:
        metadata = json.loads(args_path.read_text())
    except Exception as exc:
        print(f"Warning: failed to read {args_path}: {exc}")
        return None

    for key in ("frames_original_dir", "frames_dir"):
        value = metadata.get(key)
        if value:
            path = Path(value).expanduser()
            if path.is_dir():
                return path.resolve()
    return None


def load_source_frames(source_dir: Path) -> dict[int, Image.Image]:
    frames: dict[int, Image.Image] = {}
    for path in sorted(source_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        frame_index = parse_frame_index(path)
        if frame_index is None:
            continue
        try:
            frames[frame_index] = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"Warning: failed to load source frame {path}: {exc}")
    return frames


def source_images_for_paths(
    frame_paths: list[Path],
    source_dir: Path | None,
    *,
    expected_num_frames: int | None = None,
    required: bool = False,
) -> list[Image.Image | None]:
    if source_dir is None:
        if required:
            raise FileNotFoundError("A source video directory is required for GT diagnostic rendering")
        return [None] * len(frame_paths)
    source_frames = load_source_frames(source_dir)
    if not source_frames:
        if required:
            raise FileNotFoundError(f"No source video frames found in {source_dir}")
        print(f"Warning: no source frames found in {source_dir}.")
        return [None] * len(frame_paths)
    if expected_num_frames is not None:
        missing = sorted(set(range(expected_num_frames)).difference(source_frames))
        if missing:
            raise ValueError(f"GT video {source_dir} is missing frame indices: {missing}")
    output: list[Image.Image | None] = []
    for frame_path in frame_paths:
        frame_index = parse_frame_index(frame_path)
        if frame_index is None or frame_index not in source_frames:
            if required:
                raise ValueError(f"No exact source-video frame for prediction frame {frame_path.name} in {source_dir}")
            output.append(None)
        else:
            output.append(source_frames[frame_index])
    return output


def labeled_panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    panel = image.convert("RGB")
    if panel.size != size:
        panel = panel.resize(size, Image.LANCZOS)
    panel = panel.copy()
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((0, 0, panel.width, 24), fill=(0, 0, 0, 155))
    draw.text((7, 6), label, fill=(255, 255, 255, 255))
    return panel


def compose_panels(images: list[Image.Image], labels: list[str], size: tuple[int, int]) -> Image.Image:
    panels = [labeled_panel(image, label, size) for image, label in zip(images, labels)]
    canvas = Image.new("RGB", (size[0] * len(panels), size[1]), (255, 255, 255))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * size[0], 0))
    return canvas


def compose_side_by_side(
    rendered_frames: list[Image.Image],
    frame_paths: list[Path],
    source_dir: Path | None,
    *,
    expected_num_frames: int | None = None,
    required: bool = False,
) -> list[Image.Image]:
    sources = source_images_for_paths(
        frame_paths, source_dir, expected_num_frames=expected_num_frames, required=required,
    )
    if not any(source is not None for source in sources):
        return rendered_frames
    output = []
    for rendered, source in zip(rendered_frames, sources):
        rendered = rendered.convert("RGB")
        if source is None:
            if required:
                raise ValueError("Required source-video frame unexpectedly resolved to None")
            source = rendered
        source = source.convert("RGB")
        if source.size != rendered.size:
            source = source.resize(rendered.size, Image.LANCZOS)
        canvas = Image.new("RGB", (rendered.width * 2, rendered.height))
        canvas.paste(source, (0, 0))
        canvas.paste(rendered, (rendered.width, 0))
        output.append(canvas)
    return output


def scene_to_mesh(scene_or_mesh: trimesh.Scene | trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh.copy()
    meshes = [mesh.copy() for mesh in scene_or_mesh.dump(concatenate=False) if isinstance(mesh, trimesh.Trimesh)]
    if not meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(meshes)


def scene_to_local_mesh(scene_or_mesh: trimesh.Scene | trimesh.Trimesh) -> trimesh.Trimesh:
    """Return canonical GLB geometry without applying its exported scene-node pose."""
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh.copy()
    else:
        meshes = [item.copy() for item in scene_or_mesh.geometry.values() if isinstance(item, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
    gltf_to_blender = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    mesh.apply_transform(gltf_to_blender)
    return mesh


def gt_geometry_names(metadata: dict) -> list[str]:
    render_objects = metadata.get("render_objects")
    if isinstance(render_objects, list):
        return [name for name in render_objects if isinstance(name, str)]
    objects = metadata.get("objects")
    if isinstance(objects, dict):
        return [name for name in objects if isinstance(name, str)]
    return ["ball_0", "ball_1", "floor"]

def load_gt_scenes(gt_root: Path, frame_paths: list[Path]) -> list[trimesh.Scene]:
    metadata_path = gt_root / "physics_metadata.json"
    mesh_dir = gt_root / "meshes"
    transform_dir = gt_root / "transforms"
    if not metadata_path.is_file() or not mesh_dir.is_dir() or not transform_dir.is_dir():
        raise FileNotFoundError(f"GT geometry requires physics_metadata.json, meshes/, and transforms/ under {gt_root}")
    metadata = json.loads(metadata_path.read_text())
    names = [name for name in gt_geometry_names(metadata) if name == "floor" or (mesh_dir / f"{name}.glb").is_file()]
    if not names:
        raise FileNotFoundError(f"No canonical GT meshes found under {mesh_dir}")
    canonical = {
        name: scene_to_local_mesh(load_scene(mesh_dir / f"{name}.glb"))
        for name in names
        if name != "floor"
    }
    if "floor" in names:
        canonical["floor"] = trimesh.Trimesh(
            vertices=[[-2.5, -2.5, 0.0], [2.5, -2.5, 0.0], [2.5, 2.5, 0.0], [-2.5, 2.5, 0.0]],
            faces=[[0, 1, 2], [0, 2, 3]], process=False,
        )
    scenes = []
    for frame_path in frame_paths:
        frame_index = parse_frame_index(frame_path)
        transform_path = transform_dir / f"frame_{int(frame_index or 0):04d}.json"
        if not transform_path.is_file():
            raise FileNotFoundError(f"GT transform not found: {transform_path}")
        transform_data = json.loads(transform_path.read_text())
        objects = transform_data.get("objects", transform_data)
        scene = trimesh.Scene()
        for name in names:
            item = objects.get(name)
            if name == "floor" and not isinstance(item, dict):
                item = {"location": [0.0, 0.0, 0.0], "quaternion_blender_wxyz": [1.0, 0.0, 0.0, 0.0]}
            elif not isinstance(item, dict):
                continue
            mesh = canonical[name].copy()
            matrix = trimesh.transformations.quaternion_matrix(item["quaternion_blender_wxyz"])
            matrix[:3, 3] = np.asarray(item["location"], dtype=np.float64)
            mesh.apply_transform(matrix)
            color = np.tile(np.asarray([0, 210, 235, 255], dtype=np.uint8), (len(mesh.vertices), 1))
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=color)
            scene.add_geometry(mesh, geom_name=f"gt_{name}")
        scenes.append(scene)
    return scenes


def fade_prediction(image: Image.Image, alpha: float) -> Image.Image:
    image = image.convert("RGB")
    white = Image.new("RGB", image.size, (255, 255, 255))
    return Image.blend(white, image, max(0.0, min(1.0, float(alpha))))


def overlay_wireframe(prediction: Image.Image, wireframe: Image.Image, alpha: float) -> Image.Image:
    faded = fade_prediction(prediction, alpha)
    wire = np.asarray(wireframe.convert("RGB"), dtype=np.uint8)
    mask_array = (255 - wire.min(axis=2)).astype(np.uint8)
    mask = Image.fromarray(mask_array)
    return Image.composite(wireframe.convert("RGB"), faded, mask)


def transformed_scene(scene: trimesh.Scene, transform: np.ndarray) -> trimesh.Scene:
    output = scene.copy()
    output.apply_transform(np.asarray(transform, dtype=np.float64))
    return output


def rotate_scene_about(scene: trimesh.Scene, center: np.ndarray, angle_degrees: float) -> trimesh.Scene:
    rotation = trimesh.transformations.rotation_matrix(
        np.deg2rad(float(angle_degrees)), np.asarray([0.0, 0.0, 1.0]), point=np.asarray(center, dtype=np.float64),
    )
    return transformed_scene(scene, rotation)


def sequence_bounds_center(scenes: list[trimesh.Scene]) -> np.ndarray:
    bounds = np.asarray([scene.bounds for scene in scenes], dtype=np.float64)
    return (bounds[:, 0].min(axis=0) + bounds[:, 1].max(axis=0)) * 0.5


def resolve_gt_render_inputs(export_dir: Path, args: argparse.Namespace) -> tuple[dict | None, object | None, Path | None]:
    has_camera = args.camera_metadata is not None or args.camera_metadata_root is not None
    has_alignment = args.alignment_metadata is not None or args.alignment_metrics_root is not None
    if not has_camera and not has_alignment:
        return None, None, None
    if has_camera != has_alignment:
        raise ValueError("GT-view rendering requires both camera metadata and reconstruction alignment metadata")

    camera_path = args.camera_metadata.expanduser().resolve() if args.camera_metadata is not None else None
    alignment_path = args.alignment_metadata.expanduser().resolve() if args.alignment_metadata is not None else None
    if camera_path is None or alignment_path is None:
        relative = export_dir.resolve().relative_to(args.predictions_root_resolved)
        if len(relative.parts) < 2:
            raise ValueError(f"Cannot infer sample/model from export path: {export_dir}; use explicit metadata paths")
        sample, model = relative.parts[:2]
        if camera_path is None:
            camera_path = args.camera_metadata_root.expanduser().resolve() / sample / "physics_metadata.json"
        if alignment_path is None:
            alignment_path = args.alignment_metrics_root.expanduser().resolve() / sample / model / "reconstruction" / "metrics.json"
    if not camera_path.is_file():
        raise FileNotFoundError(f"Camera metadata not found: {camera_path}")
    if not alignment_path.is_file():
        raise FileNotFoundError(f"Alignment metadata not found: {alignment_path}")
    gt_root = args.gt_geometry_root.expanduser().resolve() if args.gt_geometry_root is not None else camera_path.parent
    return load_camera_metadata(camera_path), load_pred_to_gt_transform(alignment_path), gt_root

def render_export_dir(export_dir: Path, args: argparse.Namespace) -> str:
    frame_paths = sorted((export_dir / "dynamic").glob("dynamic_scene_frame_*.glb"))
    if args.frame_stride > 1:
        frame_paths = frame_paths[:: args.frame_stride]
    if args.max_frames and args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
    output_path = export_dir / args.output_name
    if not frame_paths:
        return "skip:no_frames"

    camera_metadata, pred_to_gt_transform, gt_root = resolve_gt_render_inputs(export_dir, args)
    gt_mode = camera_metadata is not None and pred_to_gt_transform is not None
    artifact_paths = [output_path]
    if gt_mode and not args.no_diagnostic_artifacts:
        artifact_paths += [export_dir / args.diagnostic_output_name, export_dir / args.gt_overlay_output_name]
    pending_paths = [path for path in artifact_paths if args.overwrite or not path.exists()]
    if not pending_paths:
        return "skip:exists"
    viewpoint = "gt-camera" if gt_mode else "fixed-camera"
    if args.dry_run:
        return f"dry-run:{viewpoint}:{len(frame_paths)}frames->{','.join(str(path) for path in pending_paths)}"

    gt_num_frames = None
    output_fps = int(args.fps) if args.fps is not None else 18
    if gt_mode:
        assert gt_root is not None
        gt_metadata = json.loads((gt_root / "physics_metadata.json").read_text())
        gt_num_frames = int(gt_metadata["num_frames"])
        gt_fps = float(gt_metadata["fps"])
        if gt_num_frames <= 0 or not np.isfinite(gt_fps) or gt_fps <= 0:
            raise ValueError(f"Invalid fps/num_frames in {gt_root / 'physics_metadata.json'}")
        if args.fps is None:
            output_fps = max(1, int(round(gt_fps)))

    size = (int(args.render_size), int(args.render_size))
    scenes = [load_scene(path) for path in frame_paths]
    fitted_frames = render_sequence_fixed_camera(
        scenes, azimuth=float(args.azimuth), elevation=float(args.elevation), fit_scale=float(args.fit_scale),
        image_size=size, light_intensity=5.0, return_type="pil", bg_color=(255, 255, 255, 255),
    )
    primary_frames = fitted_frames
    gt_view_frames = None
    if gt_mode:
        gt_view_frames = render_sequence_fixed_camera(
            scenes, image_size=size, light_intensity=5.0, return_type="pil", bg_color=(255, 255, 255, 255),
            camera_metadata=camera_metadata, pred_to_gt_transform=pred_to_gt_transform,
        )
        primary_frames = gt_view_frames
    source_dir = discover_source_frames_dir(export_dir, args, gt_root if gt_mode else None)
    if not args.no_side_by_side:
        primary_frames = compose_side_by_side(
            primary_frames, frame_paths, source_dir,
            expected_num_frames=gt_num_frames, required=gt_mode,
        )
    if output_path in pending_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_renderings(primary_frames, str(output_path), fps=output_fps)

    if gt_mode and not args.no_diagnostic_artifacts:
        source_images = source_images_for_paths(
            frame_paths, source_dir, expected_num_frames=gt_num_frames, required=True,
        )
        diagnostic_frames = []
        assert gt_view_frames is not None
        for source, gt_view, fitted in zip(source_images, gt_view_frames, fitted_frames):
            diagnostic_frames.append(compose_panels(
                [source, gt_view, fitted],  # type: ignore[list-item]
                ["Video", "Ground-truth viewpoint alignment prediction", "Default viewpoint prediction"], size,
            ))

        aligned_last = transformed_scene(scenes[-1], pred_to_gt_transform)
        rotation_count = max(1, int(np.ceil(360.0 / float(args.rotation_step_degrees))))
        rotation_center = np.asarray(aligned_last.bounds, dtype=np.float64).mean(axis=0)
        rotating_scenes = [
            rotate_scene_about(aligned_last, rotation_center, index * args.rotation_step_degrees)
            for index in range(rotation_count)
        ]
        rotating_frames = render_sequence_fixed_camera(
            rotating_scenes, image_size=size, light_intensity=5.0, return_type="pil", bg_color=(255, 255, 255, 255),
            camera_metadata=camera_metadata, pred_to_gt_transform=np.eye(4, dtype=np.float64),
        )
        frozen_source = source_images[-1]
        frozen_default = fitted_frames[-1]
        diagnostic_frames.extend(
            compose_panels(
                [frozen_source, rotating, frozen_default],  # type: ignore[list-item]
                ["Video - frozen", "Ground-truth viewpoint alignment prediction - frozen", "Default viewpoint prediction - frozen"], size,
            )
            for rotating in rotating_frames
        )

        diagnostic_path = export_dir / args.diagnostic_output_name
        overlay_path = export_dir / args.gt_overlay_output_name
        if diagnostic_path in pending_paths:
            export_renderings(diagnostic_frames, str(diagnostic_path), fps=output_fps)
        if overlay_path in pending_paths:
            gt_scenes = load_gt_scenes(gt_root, frame_paths)
            aligned_scenes = [transformed_scene(scene, pred_to_gt_transform) for scene in scenes]
            rotation_center = sequence_bounds_center(gt_scenes)
            angles = [index * args.rotation_step_degrees for index in range(len(frame_paths))]
            rotating_predictions = [
                rotate_scene_about(scene, rotation_center, angle) for scene, angle in zip(aligned_scenes, angles)
            ]
            rotating_gt_scenes = [
                rotate_scene_about(scene, rotation_center, angle) for scene, angle in zip(gt_scenes, angles)
            ]
            rotating_prediction_frames = render_sequence_fixed_camera(
                rotating_predictions, image_size=size, light_intensity=5.0, return_type="pil", bg_color=(255, 255, 255, 255),
                camera_metadata=camera_metadata, pred_to_gt_transform=np.eye(4, dtype=np.float64),
            )
            gt_wireframes = render_sequence_fixed_camera(
                rotating_gt_scenes, image_size=size, light_intensity=5.0, return_type="pil", bg_color=(255, 255, 255, 255),
                flags=pyrender.constants.RenderFlags.ALL_WIREFRAME,
                camera_metadata=camera_metadata, pred_to_gt_transform=np.eye(4, dtype=np.float64),
            )
            overlays = [
                labeled_panel(overlay_wireframe(pred, wire, args.prediction_overlay_alpha), "Transparent alignment prediction with ground-truth rendering", size)
                for pred, wire in zip(rotating_prediction_frames, gt_wireframes)
            ]
            export_renderings(overlays, str(overlay_path), fps=output_fps)
    return f"wrote:{len(frame_paths)}frames->{','.join(str(path) for path in pending_paths)}"

def main() -> None:
    args = parse_args()
    if args.camera_metadata is not None and args.camera_metadata_root is not None:
        raise SystemExit("Use only one of --camera-metadata and --camera-metadata-root")
    if args.alignment_metadata is not None and args.alignment_metrics_root is not None:
        raise SystemExit("Use only one of --alignment-metadata and --alignment-metrics-root")
    predictions_root = args.predictions_root.expanduser().resolve()
    args.predictions_root_resolved = predictions_root
    if not args.export_dir and not predictions_root.is_dir():
        raise SystemExit(f"predictions root does not exist: {predictions_root}")
    if not 0.0 <= args.prediction_overlay_alpha <= 1.0:
        raise SystemExit("--prediction-overlay-alpha must be in [0, 1]")
    if not 0.0 < args.rotation_step_degrees <= 360.0:
        raise SystemExit("--rotation-step-degrees must be in (0, 360]")
    samples = set(args.sample) if args.sample else None
    models = set(args.model) if args.model else None
    excluded = set(args.exclude_sample or [])

    export_dirs: list[Path] = []
    if args.export_dir:
        export_dirs.extend(path.expanduser().resolve() for path in args.export_dir)
    else:
        for model_dir in iter_model_dirs(predictions_root, samples, models, excluded):
            export_dirs.extend(iter_export_dirs(model_dir))

    if not export_dirs:
        print("No prediction export dirs with dynamic_scene_frame_*.glb found.")
        return

    wrote = skipped = failed = 0
    for export_dir in export_dirs:
        try:
            status = render_export_dir(export_dir, args)
            if status.startswith("wrote"):
                wrote += 1
            elif status.startswith("dry-run"):
                skipped += 1
            else:
                skipped += 1
            print(f"[{status}] {export_dir}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[error:{type(exc).__name__}: {exc}] {export_dir}", flush=True)

    print(f"Done. wrote={wrote} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

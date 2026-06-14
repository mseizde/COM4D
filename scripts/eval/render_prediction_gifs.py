#!/usr/bin/env python3
"""Render animation.gif files from already-exported COM4D dynamic GLBs.

This is a visualization backfill utility. It does not run inference and does not
need checkpoint paths. It scans prediction export directories containing
``dynamic/dynamic_scene_frame_*.glb`` and renders those frame GLBs with the same
fixed-camera helper used by inference_com4d.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils.render_utils import export_renderings, render_sequence_fixed_camera


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill animation.gif from exported dynamic GLB frames.")
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
    ap.add_argument("--fps", type=int, default=18, help="GIF frames per second.")
    ap.add_argument("--azimuth", type=float, default=35.0, help="Fixed camera azimuth in degrees.")
    ap.add_argument("--elevation", type=float, default=20.0, help="Fixed camera elevation in degrees.")
    ap.add_argument("--fit-scale", type=float, default=2.2, help="Camera distance multiplier for global sequence bounds.")
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


def render_export_dir(export_dir: Path, args: argparse.Namespace) -> str:
    frame_paths = sorted((export_dir / "dynamic").glob("dynamic_scene_frame_*.glb"))
    if args.frame_stride > 1:
        frame_paths = frame_paths[:: args.frame_stride]
    if args.max_frames and args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
    output_path = export_dir / args.output_name

    if not frame_paths:
        return "skip:no_frames"
    if output_path.exists() and not args.overwrite:
        return "skip:exists"
    if args.dry_run:
        return f"dry-run:{len(frame_paths)}frames->{output_path}"

    scenes = [load_scene(path) for path in frame_paths]
    frames = render_sequence_fixed_camera(
        scenes,
        azimuth=float(args.azimuth),
        elevation=float(args.elevation),
        fit_scale=float(args.fit_scale),
        image_size=(int(args.render_size), int(args.render_size)),
        light_intensity=5.0,
        return_type="pil",
        bg_color=(255, 255, 255, 255),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_renderings(frames, str(output_path), fps=int(args.fps))
    return f"wrote:{len(frame_paths)}frames->{output_path}"


def main() -> None:
    args = parse_args()
    predictions_root = args.predictions_root.expanduser().resolve()
    if not predictions_root.is_dir():
        raise SystemExit(f"predictions root does not exist: {predictions_root}")
    samples = set(args.sample) if args.sample else None
    models = set(args.model) if args.model else None
    excluded = set(args.exclude_sample or [])

    export_dirs: list[Path] = []
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

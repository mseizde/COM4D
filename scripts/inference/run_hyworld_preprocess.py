#!/usr/bin/env python3
"""Run HY-World/WorldMirror on COM4D frame folders and save geometry priors.

Run from the dedicated HY-World environment:

    CUDA_VISIBLE_DEVICES=1 micromamba run -n hyworld2 python scripts/inference/run_hyworld_preprocess.py \
        --frames-dir assets/teaser/video_frames_raw \
        --output-dir assets/teaser/hyworld_direct
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HY-World WorldMirror preprocessing for a frame directory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-dir", help="Directory containing RGB frames.")
    source.add_argument("--video-path", help="Video path. Frames are extracted uniformly without square padding.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults next to the input.")
    parser.add_argument("--hyworld-repo", default="/data/mseizde/com4d/HY-World-2.0")
    parser.add_argument("--model-id", default="tencent/HY-World-2.0")
    parser.add_argument("--subfolder", default="HY-WorldMirror-2.0")
    parser.add_argument("--target-size", type=int, default=952)
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame from a frame directory.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames. 0 means no limit.")
    parser.add_argument("--fps", type=int, default=16, help="Uniform extraction FPS for --video-path.")
    parser.add_argument("--save-conf", action="store_true", default=True)
    parser.add_argument("--no-save-conf", dest="save_conf", action="store_false")
    parser.add_argument("--save-colmap", action="store_true")
    parser.add_argument("--save-rendered", action="store_true")
    parser.add_argument("--no-preview-png", action="store_true")
    return parser.parse_args()


def default_output_dir(frames_dir: Path | None, video_path: Path | None) -> Path:
    if video_path is not None:
        return video_path.parent / f"{video_path.stem}_hyworld"
    if frames_dir is None:
        raise ValueError("Either frames_dir or video_path is required")
    return frames_dir.parent / "hyworld"


def list_frames(frames_dir: Path, stride: int, max_frames: int) -> list[Path]:
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    frames = sorted(p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    frames = frames[::stride]
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"No image frames found in {frames_dir}")
    return frames


def extract_video_frames(video_path: Path, output_dir: Path, fps: int) -> Path:
    frames_dir = output_dir / "input_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("*.png"):
        old_frame.unlink()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-vsync",
        "0",
        str(frames_dir / "%06d.png"),
    ]
    subprocess.run(command, check=True)
    return frames_dir


def stage_frames(frames: list[Path], output_dir: Path) -> Path:
    staged = output_dir / "input_frames"
    staged.mkdir(parents=True, exist_ok=True)
    for old in staged.iterdir():
        if old.is_file() and old.suffix.lower() in IMAGE_EXTS:
            old.unlink()
    for idx, src in enumerate(frames, start=1):
        dst = staged / f"{idx:06d}{src.suffix.lower()}"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    return staged


def create_preview_png(output_dir: Path) -> dict[str, list[str]]:
    preview_root = output_dir / "preview_png"
    mapping = {
        "depth": (output_dir / "depth", "depth_{:04d}.png", 0),
        "normals": (output_dir / "normal", "normal_{:04d}.png", 0),
        "depth_conf": (output_dir / "depth_conf", "conf_{:04d}.png", 1),
    }
    previews: dict[str, list[str]] = {}
    for name, (src_dir, pattern, start_idx) in mapping.items():
        if not src_dir.is_dir():
            continue
        dst_dir = preview_root / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        src_files = sorted(src_dir.glob("*.png"))
        for out_idx, _ in enumerate(src_files, start=1):
            src = src_dir / pattern.format(out_idx - 1 + start_idx)
            if not src.exists():
                continue
            dst = dst_dir / f"{out_idx:06d}.png"
            shutil.copy2(src, dst)
            paths.append(str(dst))
        previews[name] = paths
    return previews


def image_sizes(frames: list[Path]) -> list[list[int]]:
    sizes = []
    for frame in frames:
        with Image.open(frame) as image:
            w, h = image.size
        sizes.append([h, w])
    return sizes


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).resolve() if args.video_path else None
    frames_dir_arg = Path(args.frames_dir).resolve() if args.frames_dir else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(frames_dir_arg, video_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if video_path is not None:
        frames_dir = extract_video_frames(video_path, output_dir, args.fps)
        frames = list_frames(frames_dir, 1, args.max_frames)
    else:
        assert frames_dir_arg is not None
        frames = list_frames(frames_dir_arg, args.stride, args.max_frames)
        frames_dir = stage_frames(frames, output_dir)
        frames = list_frames(frames_dir, 1, 0)

    hyworld_repo = Path(args.hyworld_repo).resolve()
    command = [
        sys.executable,
        "-m",
        "hyworld2.worldrecon.pipeline",
        "--input_path",
        str(frames_dir),
        "--strict_output_path",
        str(output_dir),
        "--pretrained_model_name_or_path",
        args.model_id,
        "--subfolder",
        args.subfolder,
        "--target_size",
        str(args.target_size),
        "--no_interactive",
    ]
    if args.save_conf:
        command.append("--save_conf")
    if args.save_colmap:
        command.append("--save_colmap")
    if args.save_rendered:
        command.append("--save_rendered")

    subprocess.run(command, cwd=str(hyworld_repo), check=True)

    previews = {} if args.no_preview_png else create_preview_png(output_dir)
    frame_names = [p.name for p in frames]
    summary = {
        "method": "HY-World-2.0 WorldMirror",
        "model_id": args.model_id,
        "subfolder": args.subfolder,
        "video_path": str(video_path) if video_path else None,
        "frames_dir": str(frames_dir),
        "num_frames": len(frames),
        "frames": frame_names,
        "source_frame_sizes": image_sizes(frames),
        "target_size": args.target_size,
        "output_dir": str(output_dir),
        "cameras": str(output_dir / "camera_params.json"),
        "outputs": {
            "depth": sorted(str(p) for p in (output_dir / "depth").glob("*.npy")) if (output_dir / "depth").is_dir() else [],
            "depth_preview": sorted(str(p) for p in (output_dir / "depth").glob("*.png")) if (output_dir / "depth").is_dir() else [],
            "normals_preview": sorted(str(p) for p in (output_dir / "normal").glob("*.png")) if (output_dir / "normal").is_dir() else [],
            "depth_conf_preview": sorted(str(p) for p in (output_dir / "depth_conf").glob("*.png")) if (output_dir / "depth_conf").is_dir() else [],
            "points": str(output_dir / "points.ply") if (output_dir / "points.ply").exists() else None,
            "gaussians": str(output_dir / "gaussians.ply") if (output_dir / "gaussians.ply").exists() else None,
        },
        "preview_png": previews,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved HY-World preprocessing outputs to {output_dir}")


if __name__ == "__main__":
    main()

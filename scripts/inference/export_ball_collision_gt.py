#!/usr/bin/env python3
"""Export synthetic two-ball GT passes into a COM4D asset folder."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/mnt/mocap_b/work/com4d/datasets/synthetic/two_ball_compare/gt_raw/two_ball_eval_000"),
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("/data/mseizde/com4d/COM4D/assets/ball_collision"),
    )
    parser.add_argument("--fps", type=float, default=None, help="Override FPS. Defaults to physics_metadata.json fps.")
    return parser.parse_args()


def list_frames(frame_dir: Path, suffix: str) -> list[Path]:
    frames = sorted(frame_dir.glob(f"frame_*.{suffix}"))
    if not frames:
        raise FileNotFoundError(f"No frame_*.{suffix} files found under {frame_dir}")
    return frames


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    reset_dir(dst)
    for path in sorted(src.iterdir()):
        if path.is_file():
            shutil.copy2(path, dst / path.name)


def read_exr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read EXR: {path}")
    return image.astype(np.float32, copy=False)


def depth_preview(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[finite], [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    grayscale = (255.0 * (1.0 - normalized)).astype(np.uint8)
    return cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)


def normal_preview(normal_rgb: np.ndarray) -> np.ndarray:
    preview_rgb = np.clip((normal_rgb + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)


def write_video_from_frames(frame_paths: list[Path], out_path: Path, fps: float) -> None:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")

    frame_dir = frame_paths[0].parent
    frame_pattern = frame_dir / "frame_%04d.png"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(frame_pattern),
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def export_depth(raw_dir: Path, out_dir: Path) -> None:
    exr_dir = raw_dir / "depth"
    frames = list_frames(exr_dir, "exr")
    copy_tree(exr_dir, out_dir / "depth_exr")

    npy_dir = out_dir / "depth"
    preview_dir = out_dir / "preview_png" / "depth"
    reset_dir(npy_dir)
    reset_dir(preview_dir)
    for frame_path in frames:
        depth_exr = read_exr(frame_path)
        depth = depth_exr[..., 0] if depth_exr.ndim == 3 else depth_exr
        stem = frame_path.stem
        np.save(npy_dir / f"{stem}.npy", depth)
        cv2.imwrite(str(preview_dir / f"{stem}.png"), depth_preview(depth))


def export_normals(raw_dir: Path, out_dir: Path) -> None:
    exr_dir = raw_dir / "normals"
    frames = list_frames(exr_dir, "exr")
    copy_tree(exr_dir, out_dir / "normals_exr")

    npy_dir = out_dir / "normals"
    preview_dir = out_dir / "preview_png" / "normals"
    reset_dir(npy_dir)
    reset_dir(preview_dir)
    for frame_path in frames:
        normal_bgr = read_exr(frame_path)
        if normal_bgr.ndim != 3 or normal_bgr.shape[2] < 3:
            raise RuntimeError(f"Expected 3-channel normal EXR: {frame_path}")
        normal_rgb = normal_bgr[..., :3][..., ::-1]
        stem = frame_path.stem
        np.save(npy_dir / f"{stem}.npy", normal_rgb)
        cv2.imwrite(str(preview_dir / f"{stem}.png"), normal_preview(normal_rgb))


def load_fps(raw_dir: Path, override: float | None) -> float:
    if override is not None:
        return override
    metadata_path = raw_dir / "physics_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if "fps" in metadata:
            return float(metadata["fps"])
    return 30.0


def main() -> None:
    # OpenCV requires this opt-in before EXR images are decoded.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    args = parse_args()
    raw_dir = args.raw_dir
    asset_dir = args.asset_dir
    out_dir = asset_dir / "gt"
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = load_fps(raw_dir, args.fps)
    rgb_frames = list_frames(raw_dir / "render_rgb", "png")

    export_depth(raw_dir, out_dir)
    export_normals(raw_dir, out_dir)
    copy_tree(raw_dir / "render_rgb", out_dir / "render_rgb")
    if (raw_dir / "transforms").exists():
        copy_tree(raw_dir / "transforms", out_dir / "transforms")
    if (raw_dir / "physics_metadata.json").exists():
        shutil.copy2(raw_dir / "physics_metadata.json", out_dir / "physics_metadata.json")

    write_video_from_frames(rgb_frames, out_dir / "rgb.mp4", fps)
    write_video_from_frames(list_frames(out_dir / "preview_png" / "depth", "png"), out_dir / "depth_preview.mp4", fps)
    write_video_from_frames(list_frames(out_dir / "preview_png" / "normals", "png"), out_dir / "normals_preview.mp4", fps)

    print(f"Exported GT assets to {out_dir}")
    print(f"Frames: {len(rgb_frames)}")
    print(f"FPS: {fps:g}")


if __name__ == "__main__":
    main()

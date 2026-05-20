#!/usr/bin/env python3

"""Prepare a raw two-ball sequence for COM4D inference.

The raw two-ball renderer writes:
  render_rgb/frame_0000.png
  masks/ball_0/frame_0000.png
  masks/ball_1/frame_0000.png

COM4D inference expects:
  frames/frame_0000.png
  masks/frame_0000_object_000.png
  masks/frame_0000_object_001.png
  masks_static/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True, help="Raw two-ball sequence directory.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Prepared COM4D inference input directory.")
    ap.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError as exc:
        print(f"[warn] Symlink failed ({exc}); copying instead: {dst}")
        shutil.copy2(src, dst)


def frame_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {directory}")
    return paths


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    rgb_dir = raw_dir / "render_rgb"
    ball_dirs = [raw_dir / "masks" / "ball_0", raw_dir / "masks" / "ball_1"]
    metadata_path = raw_dir / "physics_metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing physics metadata: {metadata_path}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")
    for ball_dir in ball_dirs:
        if not ball_dir.is_dir():
            raise FileNotFoundError(f"Missing mask directory: {ball_dir}")

    reset_dir(output_dir, overwrite=args.overwrite)
    (output_dir / "frames").mkdir()
    (output_dir / "masks").mkdir()
    (output_dir / "masks_static").mkdir()

    rgbs = frame_paths(rgb_dir)
    masks_by_ball = [frame_paths(ball_dir) for ball_dir in ball_dirs]
    frame_count = len(rgbs)
    for obj_idx, masks in enumerate(masks_by_ball):
        if len(masks) != frame_count:
            raise ValueError(
                f"Mask count mismatch for object {obj_idx}: {len(masks)} masks vs {frame_count} RGB frames."
            )

    for frame_idx, src in enumerate(rgbs):
        link_or_copy(src, output_dir / "frames" / f"frame_{frame_idx:04d}.png", args.mode)

    for obj_idx, masks in enumerate(masks_by_ball):
        for frame_idx, src in enumerate(masks):
            link_or_copy(
                src,
                output_dir / "masks" / f"frame_{frame_idx:04d}_object_{obj_idx:03d}.png",
                args.mode,
            )

    link_or_copy(metadata_path, output_dir / "physics_metadata.json", args.mode)
    print(f"Prepared {frame_count} frames for COM4D inference: {output_dir}")


if __name__ == "__main__":
    main()

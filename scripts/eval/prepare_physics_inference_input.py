#!/usr/bin/env python3

"""Prepare a raw synthetic physics sequence for COM4D inference.

The raw physics renderer writes:
  render_rgb/frame_0000.png
  masks/ball_0/frame_0000.png
  masks/ball_1/frame_0000.png
  masks/wall/frame_0000.png or masks/occluder_box/frame_0000.png, when present

COM4D inference expects:
  frames/frame_0000.png
  masks/frame_0000_object_000.png
  masks/frame_0000_object_001.png
  masks_static/frame_0000_object_000.png
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dynamic_object_names(metadata: dict) -> list[str]:
    objects = metadata.get("objects")
    if isinstance(objects, dict) and objects:
        names = [
            name
            for name, spec in objects.items()
            if isinstance(spec, dict) and bool(spec.get("dynamic", name.startswith("ball_")))
        ]
        if names:
            render_objects = metadata.get("render_objects")
            if isinstance(render_objects, list):
                order = {name: idx for idx, name in enumerate(render_objects)}
                return sorted(names, key=lambda name: order.get(name, len(order)))
            return sorted(names)
    render_objects = metadata.get("render_objects")
    if isinstance(render_objects, list):
        names = [name for name in render_objects if isinstance(name, str) and name.startswith("ball_")]
        if names:
            return names
    return ["ball_0", "ball_1"]


def static_object_names(metadata: dict) -> list[str]:
    objects = metadata.get("objects")
    render_objects = metadata.get("render_objects")
    if not isinstance(objects, dict) or not isinstance(render_objects, list):
        return []
    return [
        name
        for name in render_objects
        if (
            isinstance(name, str)
            and name != "floor"
            and isinstance(objects.get(name), dict)
            and not bool(objects[name].get("dynamic", name.startswith("ball_")))
        )
    ]


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
    metadata_path = raw_dir / "physics_metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing physics metadata: {metadata_path}")
    metadata = load_metadata(metadata_path)
    object_names = dynamic_object_names(metadata)
    static_names = static_object_names(metadata)
    object_dirs = [raw_dir / "masks" / name for name in object_names]
    static_dirs = [raw_dir / "masks" / name for name in static_names]
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")
    for object_dir in object_dirs + static_dirs:
        if not object_dir.is_dir():
            raise FileNotFoundError(f"Missing mask directory: {object_dir}")

    reset_dir(output_dir, overwrite=args.overwrite)
    (output_dir / "frames").mkdir()
    (output_dir / "masks").mkdir()
    (output_dir / "masks_static").mkdir()

    rgbs = frame_paths(rgb_dir)
    masks_by_object = [frame_paths(object_dir) for object_dir in object_dirs]
    static_masks_by_object = [frame_paths(object_dir) for object_dir in static_dirs]
    frame_count = len(rgbs)
    for obj_idx, masks in enumerate(masks_by_object + static_masks_by_object):
        if len(masks) != frame_count:
            raise ValueError(
                f"Mask count mismatch for object {obj_idx}: {len(masks)} masks vs {frame_count} RGB frames."
            )

    for frame_idx, src in enumerate(rgbs):
        link_or_copy(src, output_dir / "frames" / f"frame_{frame_idx:04d}.png", args.mode)

    for obj_idx, masks in enumerate(masks_by_object):
        for frame_idx, src in enumerate(masks):
            link_or_copy(
                src,
                output_dir / "masks" / f"frame_{frame_idx:04d}_object_{obj_idx:03d}.png",
                args.mode,
            )

    for obj_idx, masks in enumerate(static_masks_by_object):
        for frame_idx, src in enumerate(masks):
            link_or_copy(
                src,
                output_dir / "masks_static" / f"frame_{frame_idx:04d}_object_{obj_idx:03d}.png",
                args.mode,
            )

    link_or_copy(metadata_path, output_dir / "physics_metadata.json", args.mode)
    print(
        f"Prepared {frame_count} frames, {len(object_names)} dynamic objects, "
        f"and {len(static_names)} static objects for COM4D inference: {output_dir}"
    )


if __name__ == "__main__":
    main()

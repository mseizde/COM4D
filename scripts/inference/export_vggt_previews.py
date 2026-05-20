#!/usr/bin/env python3
"""Export human-readable PNG previews from a VGGT preprocessing directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VGGT .npy outputs into preview PNGs.")
    parser.add_argument("vggt_dir", help="Directory containing depth/, normals/, depth_conf/, etc.")
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Optional COM4D frame directory. If provided, previews are resized to each frame's dimensions.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preview PNGs.")
    return parser.parse_args()


def normalize_scalar_image(
    values: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    if value_range is None:
        lo, hi = np.percentile(values[finite], [percentile_low, percentile_high])
    else:
        lo, hi = value_range
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values[finite].min())
        hi = float(values[finite].max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~finite] = 0.0
    return (normalized * 255.0).astype(np.uint8)


def colorize_depth(depth: np.ndarray, value_range: tuple[float, float] | None = None) -> np.ndarray:
    gray = normalize_scalar_image(depth, value_range=value_range)
    x = gray.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def normal_to_rgb(normals: np.ndarray) -> np.ndarray:
    rgb = (np.asarray(normals, dtype=np.float32) * 0.5 + 0.5) * 255.0
    rgb[~np.isfinite(rgb)] = 0.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def frame_sizes_by_stem(frames_dir: Path | None) -> dict[str, tuple[int, int]]:
    if frames_dir is None or not frames_dir.exists():
        return {}
    sizes = {}
    for path in frames_dir.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        with Image.open(path) as image:
            sizes[path.stem] = image.size
    return sizes


def scalar_range(source_dir: Path) -> tuple[float, float] | None:
    values = []
    for npy_path in sorted(source_dir.glob("*.npy")):
        array = np.load(npy_path).astype(np.float32, copy=False)
        finite = array[np.isfinite(array)]
        if finite.size:
            values.append(finite.reshape(-1))
    if not values:
        return None
    merged = np.concatenate(values)
    lo, hi = np.percentile(merged, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return float(lo), float(hi)


def convert_folder(
    vggt_dir: Path,
    source_name: str,
    preview_name: str,
    kind: str,
    overwrite: bool,
    frame_sizes: dict[str, tuple[int, int]],
) -> list[str]:
    source_dir = vggt_dir / source_name
    if not source_dir.exists():
        return []
    out_dir = vggt_dir / "preview_png" / preview_name
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    value_range = scalar_range(source_dir) if kind in {"depth", "scalar"} else None
    for npy_path in sorted(source_dir.glob("*.npy")):
        out_path = out_dir / f"{npy_path.stem}.png"
        if out_path.exists() and not overwrite:
            written.append(str(out_path))
            continue
        array = np.load(npy_path)
        if kind == "depth":
            image = colorize_depth(array, value_range=value_range)
        elif kind == "normal":
            image = normal_to_rgb(array)
        else:
            image = normalize_scalar_image(array, value_range=value_range)
        pil_image = Image.fromarray(image)
        if npy_path.stem in frame_sizes and pil_image.size != frame_sizes[npy_path.stem]:
            pil_image = pil_image.resize(frame_sizes[npy_path.stem], Image.Resampling.BILINEAR)
        pil_image.save(out_path)
        written.append(str(out_path))
    return written


def main() -> None:
    args = parse_args()
    vggt_dir = Path(args.vggt_dir).resolve()
    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir else None
    frame_sizes = frame_sizes_by_stem(frames_dir)
    outputs = {
        "depth": convert_folder(vggt_dir, "depth", "depth", "depth", args.overwrite, frame_sizes),
        "normals": convert_folder(vggt_dir, "normals", "normals", "normal", args.overwrite, frame_sizes),
        "depth_conf": convert_folder(vggt_dir, "depth_conf", "depth_conf", "scalar", args.overwrite, frame_sizes),
        "point_conf": convert_folder(vggt_dir, "point_conf", "point_conf", "scalar", args.overwrite, frame_sizes),
    }
    summary_path = vggt_dir / "preview_png" / "summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2))
    total = sum(len(paths) for paths in outputs.values())
    print(f"Wrote or found {total} preview PNGs under {vggt_dir / 'preview_png'}")


if __name__ == "__main__":
    main()

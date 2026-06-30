#!/usr/bin/env python3

"""Binarize prepared physics inference masks.

The COM4D inference path uses mask grayscale values as alpha. This utility
converts prepared masks to strict 0/255 so gray "undefined" pixels do not act
as partial foreground.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path("/mnt/mocap_b/work/com4d/datasets/synthetic/physics_compare/inference_input")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Prepared inference_input root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Sample to process, e.g. wall_impact_eval_001. Can be repeated. Defaults to all samples.",
    )
    parser.add_argument(
        "--mask-subdir",
        action="append",
        default=None,
        help="Mask subdir to process. Defaults to masks and masks_static. Can be repeated.",
    )
    parser.add_argument(
        "--mode",
        choices=("white-only", "threshold"),
        default="white-only",
        help="white-only keeps only pixels equal to 255; threshold keeps pixels > --threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Threshold used only with --mode threshold.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=None,
        help="Optional suffix for one-time backups before overwrite, e.g. .pre_binarize.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    return parser.parse_args()


def iter_mask_dirs(root: Path, samples: list[str] | None, mask_subdirs: list[str]) -> list[Path]:
    sample_dirs = [root / sample for sample in samples] if samples else sorted(p for p in root.iterdir() if p.is_dir())
    dirs: list[Path] = []
    for sample_dir in sample_dirs:
        for subdir in mask_subdirs:
            mask_dir = sample_dir / subdir
            if mask_dir.is_dir():
                dirs.append(mask_dir)
            else:
                print(f"[skip:missing] {mask_dir}")
    return dirs


def binarize_array(arr: np.ndarray, mode: str, threshold: int) -> np.ndarray:
    if mode == "white-only":
        keep = arr == 255
    else:
        keep = arr > threshold
    return np.where(keep, 255, 0).astype(np.uint8)


def process_mask(path: Path, mode: str, threshold: int, dry_run: bool, backup_suffix: str | None) -> tuple[bool, int, int]:
    image = Image.open(path).convert("L")
    arr = np.asarray(image)
    out = binarize_array(arr, mode, threshold)
    changed = bool(np.any(out != arr))
    non_binary = int(np.count_nonzero((arr != 0) & (arr != 255)))
    foreground = int(np.count_nonzero(out == 255))
    if changed and not dry_run:
        if backup_suffix:
            backup = path.with_name(path.name + backup_suffix)
            if not backup.exists():
                shutil.copy2(path, backup)
        Image.fromarray(out).save(path)
    return changed, non_binary, foreground


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root does not exist or is not a directory: {root}")
    if not 0 <= args.threshold <= 255:
        raise SystemExit("--threshold must be in [0, 255]")

    mask_subdirs = args.mask_subdir or ["masks", "masks_static"]
    mask_dirs = iter_mask_dirs(root, args.sample, mask_subdirs)
    if not mask_dirs:
        raise SystemExit("No mask directories found.")

    total_files = 0
    changed_files = 0
    non_binary_pixels = 0
    foreground_pixels = 0

    for mask_dir in mask_dirs:
        paths = sorted(mask_dir.glob("*.png"))
        print(f"[dir] {mask_dir} ({len(paths)} pngs)")
        for path in paths:
            changed, non_binary, foreground = process_mask(
                path,
                mode=args.mode,
                threshold=args.threshold,
                dry_run=args.dry_run,
                backup_suffix=args.backup_suffix,
            )
            total_files += 1
            changed_files += int(changed)
            non_binary_pixels += non_binary
            foreground_pixels += foreground

    action = "would change" if args.dry_run else "changed"
    print(
        f"Done: {action} {changed_files}/{total_files} files; "
        f"non-binary input pixels={non_binary_pixels}; foreground output pixels={foreground_pixels}."
    )


if __name__ == "__main__":
    main()

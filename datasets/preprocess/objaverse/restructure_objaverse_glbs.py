#!/usr/bin/env python3
"""Restructure Objaverse GLBs into per-object folders.

This converts the default Objaverse cache layout:

    <input-root>/<shard>/<uid>.glb

into a preprocessing-friendly layout:

    <output-root>/<uid>/mesh.glb

Optionally, a category prefix can be added to the folder name when a manifest
or metadata lookup provides one:

    <output-root>/chair__<uid>/mesh.glb

The GLB filename stays `mesh.glb`; the UID remains embedded in the folder name
so downstream code can still recover a stable identifier.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restructure Objaverse GLBs into per-object folders.")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root containing downloaded Objaverse GLBs, typically .../hf-objaverse-v1/glbs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root that will contain one folder per object with mesh.glb inside.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional UID-to-local-path manifest JSON written by objaverse_filter.py.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="Optional Objaverse metadata directory for looking up human-readable categories.",
    )
    parser.add_argument(
        "--prefix-category",
        action="store_true",
        help="Prefix folder names with an inferred category, e.g. chair__<uid>.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize mesh.glb in the output tree.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output mesh.glb if present.",
    )
    return parser.parse_args()


def _sanitize_token(value: str) -> str:
    token = value.strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _load_manifest(manifest_path: Optional[Path]) -> Dict[str, Path]:
    if manifest_path is None:
        return {}
    with manifest_path.open("r") as f:
        data = json.load(f)
    return {uid: Path(path) for uid, path in data.items() if path}


def _iter_input_glbs(input_root: Path, manifest_paths: Dict[str, Path]) -> Iterable[Path]:
    if manifest_paths:
        for path in manifest_paths.values():
            if path.suffix.lower() == ".glb" and path.exists():
                yield path
        return
    yield from sorted(input_root.rglob("*.glb"))


def _build_metadata_index(metadata_root: Optional[Path]) -> Dict[str, Path]:
    if metadata_root is None:
        return {}
    index: Dict[str, Path] = {}
    for path in sorted(metadata_root.glob("*.json.gz")):
        shard_name = path.name.removesuffix(".json.gz")
        index[shard_name] = path
    return index


def _infer_category(uid: str, glb_path: Path, metadata_index: Dict[str, Path]) -> Optional[str]:
    shard_name = glb_path.parent.name
    metadata_path = metadata_index.get(shard_name)
    if metadata_path is None or not metadata_path.exists():
        return None
    try:
        with gzip.open(metadata_path, "rt") as f:
            data = json.load(f)
    except Exception:
        return None

    entry = data.get(uid) or {}

    categories = entry.get("categories") or []
    if categories:
        for category in categories:
            name = category.get("name", "") if isinstance(category, dict) else str(category)
            token = _sanitize_token(name)
            if token:
                return token

    name = entry.get("name")
    if isinstance(name, str):
        token = _sanitize_token(name)
        if token:
            return token

    tags = entry.get("tags") or []
    for tag in tags:
        name = tag.get("name", "") if isinstance(tag, dict) else str(tag)
        token = _sanitize_token(name)
        if token:
            return token
    return None


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    os.symlink(src.resolve(), dst)


def main() -> None:
    args = _parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_paths = _load_manifest(args.manifest.resolve() if args.manifest else None)
    metadata_index = _build_metadata_index(args.metadata_root.resolve() if args.metadata_root else None)

    written = 0
    skipped = 0

    for glb_path in _iter_input_glbs(input_root, manifest_paths):
        uid = glb_path.stem
        folder_name = uid
        if args.prefix_category:
            category = _infer_category(uid, glb_path, metadata_index)
            if category:
                folder_name = f"{category}__{uid}"

        out_dir = output_root / folder_name
        out_file = out_dir / "mesh.glb"

        if out_file.exists() or out_file.is_symlink():
            if not args.overwrite:
                skipped += 1
                continue
            out_file.unlink()

        out_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(glb_path, out_file, args.mode)
        written += 1

    print(f"Wrote {written} objects to {output_root}")
    if skipped:
        print(f"Skipped {skipped} existing objects")


if __name__ == "__main__":
    main()

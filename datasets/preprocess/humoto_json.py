#!/usr/bin/env python3
"""
Build a HUMOTO 4D dataset JSON.

The output follows the DeformingThings frame-sequence format:

{
    "<action-id>": [
        {
            "surface_path": ".../points.npy",
            "image_path": ".../frame_0001.png",
            "iou_mean": 0.0,
            "iou_max": 0.0,
            "objects": ["..."],
            "short_script": "...",
            "long_script": [...],
            "start_frame": 1,
            "end_frame": 181,
            "scene": "standalone"
        },
        ...
    ],
    ...
}

Example:
python datasets/preprocess/humoto_json.py \
    --preprocessed-root /mnt/mocap_b/work/com4d/datasets/processed/HUMOTO/points \
    --render-root /mnt/mocap_b/work/com4d/datasets/processed/HUMOTO/render \
    --yaml-root /mnt/mocap_b/work/com4d/datasets/raw/HUMOTO/humoto_0805 \
    --output ./dataset_json/humoto.json \
    --pretty
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "PyYAML is required for HUMOTO metadata. Install it with `pip install pyyaml`."
    ) from exc


SUBDIR_PATTERN = re.compile(r"^(?P<base>.+)_frame_(?P<frame>\d+)(?:_rendering)?$")
RENDER_FILE_PATTERN = re.compile(r"^frame_(?P<frame>\d+)\.(?P<ext>png|jpg|jpeg)$", re.IGNORECASE)


def parse_base_frame(dirname: str) -> Optional[Tuple[str, str]]:
    match = SUBDIR_PATTERN.match(dirname)
    if match is None:
        return None
    return match.group("base"), match.group("frame").zfill(4)


def collect_preproc(preprocessed_root: Path, verbose: bool = True) -> Dict[Tuple[str, str], Path]:
    results: Dict[Tuple[str, str], Path] = {}
    if not preprocessed_root.exists():
        if verbose:
            print(f"[WARN] Preprocessed root does not exist: {preprocessed_root}", file=sys.stderr)
        return results

    for points in preprocessed_root.rglob("points.npy"):
        parsed = parse_base_frame(points.parent.name)
        if parsed is None:
            continue
        previous = results.get(parsed)
        if previous is None or len(str(points)) < len(str(previous)):
            results[parsed] = points.resolve()
    return results


def collect_render(render_root: Path, verbose: bool = True) -> Dict[Tuple[str, str], Path]:
    results: Dict[Tuple[str, str], Path] = {}
    if not render_root.exists():
        if verbose:
            print(f"[WARN] Render root does not exist: {render_root}", file=sys.stderr)
        return results

    for image in render_root.rglob("*"):
        if not image.is_file():
            continue

        key = None
        if image.name.lower() == "rendering.png":
            parsed = parse_base_frame(image.parent.name)
            if parsed is not None:
                key = parsed
        else:
            match = RENDER_FILE_PATTERN.match(image.name)
            if match is not None:
                key = (image.parent.name, match.group("frame").zfill(4))

        if key is None:
            continue
        previous = results.get(key)
        if previous is None or len(str(image)) < len(str(previous)):
            results[key] = image.resolve()
    return results


def find_yaml_files(yaml_root: Path) -> Dict[str, Path]:
    yaml_files: Dict[str, Path] = {}
    for path in yaml_root.rglob("*.yaml"):
        action_id = path.stem
        previous = yaml_files.get(action_id)
        if previous is None or len(str(path)) < len(str(previous)):
            yaml_files[action_id] = path.resolve()
    return yaml_files


def load_yaml_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")

    metadata = {
        "objects": data.get("objects", []),
        "short_script": data.get("short_script", ""),
        "long_script": data.get("long_script", []),
        "start_frame": data.get("start_frame"),
        "end_frame": data.get("end_frame"),
        "scene": data.get("scene", ""),
    }
    if metadata["objects"] is None:
        metadata["objects"] = []
    if metadata["long_script"] is None:
        metadata["long_script"] = []
    return metadata


def _count_explicit_parts(points_path: Path) -> int:
    try:
        data = np.load(points_path, allow_pickle=True).item()
    except Exception:
        return 0
    parts = data.get("parts", None) if isinstance(data, dict) else None
    return len(parts) if isinstance(parts, list) else 0


def build_index(
    preprocessed_root: Path,
    render_root: Path,
    yaml_root: Path,
    strict: bool = False,
    verbose: bool = True,
    min_explicit_parts: int = 0,
) -> Dict[str, list[dict[str, Any]]]:
    preproc_map = collect_preproc(preprocessed_root.resolve(), verbose=verbose)
    render_map = collect_render(render_root.resolve(), verbose=verbose)
    yaml_files = find_yaml_files(yaml_root.resolve())

    if verbose:
        print(f"[INFO] Found preprocessed frames: {len(preproc_map)}")
        print(f"[INFO] Found rendered frames: {len(render_map)}")
        print(f"[INFO] Found YAML metadata files: {len(yaml_files)}")

    common_keys = sorted(
        {(name, int(frame)) for name, frame in preproc_map} & {(name, int(frame)) for name, frame in render_map},
        key=lambda item: (item[0], item[1]),
    )

    if strict:
        preproc_keys = set(preproc_map)
        render_keys = set(render_map)
        only_preproc = preproc_keys - render_keys
        only_render = render_keys - preproc_keys
        if only_preproc:
            sample = "\n  ".join(f"{base}_frame_{frame}" for base, frame in sorted(only_preproc)[:25])
            raise FileNotFoundError(f"Missing renderings for preprocessed frames, first 25:\n  {sample}")
        if only_render:
            sample = "\n  ".join(f"{base}_frame_{frame}" for base, frame in sorted(only_render)[:25])
            raise FileNotFoundError(f"Missing points.npy for rendered frames, first 25:\n  {sample}")

    grouped: Dict[str, list[dict[str, Any]]] = {}
    metadata_cache: Dict[str, Dict[str, Any]] = {}
    missing_yaml: set[str] = set()
    skipped_for_parts = 0

    for action_id, frame_int in common_keys:
        frame = f"{frame_int:04d}"
        if min_explicit_parts > 0 and _count_explicit_parts(preproc_map[(action_id, frame)]) < min_explicit_parts:
            skipped_for_parts += 1
            continue

        metadata = metadata_cache.get(action_id)
        if metadata is None:
            yaml_path = yaml_files.get(action_id)
            if yaml_path is None:
                missing_yaml.add(action_id)
                metadata = {
                    "objects": [],
                    "short_script": "",
                    "long_script": [],
                    "start_frame": None,
                    "end_frame": None,
                    "scene": "",
                }
            else:
                metadata = load_yaml_metadata(yaml_path)
            metadata_cache[action_id] = metadata

        grouped.setdefault(action_id, []).append(
            {
                "surface_path": str(preproc_map[(action_id, frame)]),
                "image_path": str(render_map[(action_id, frame)]),
                "iou_mean": 0.0,
                "iou_max": 0.0,
                **metadata,
            }
        )

    if strict and missing_yaml:
        sample = "\n  ".join(sorted(missing_yaml)[:25])
        raise FileNotFoundError(f"Missing YAML metadata for actions, first 25:\n  {sample}")

    if verbose and missing_yaml:
        print(f"[WARN] Missing YAML metadata for {len(missing_yaml)} actions", file=sys.stderr)
    if verbose and min_explicit_parts > 0:
        print(
            f"[INFO] Skipped {skipped_for_parts} frame(s) with fewer than {min_explicit_parts} explicit parts.",
            file=sys.stderr,
        )

    return {key: grouped[key] for key in sorted(grouped) if grouped[key]}


def positive_path(path: str) -> Path:
    return Path(path).expanduser()


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a HUMOTO 4D frame-sequence JSON.")
    parser.add_argument("--preprocessed-root", type=positive_path, required=True)
    parser.add_argument("--render-root", type=positive_path, required=True)
    parser.add_argument("--yaml-root", type=positive_path, required=True)
    parser.add_argument("--output", "-o", type=positive_path, required=True)
    parser.add_argument("--strict", action="store_true", help="Error on missing frame counterparts or YAML metadata.")
    parser.add_argument(
        "--min-explicit-parts",
        type=int,
        default=0,
        help="Only include frames whose points.npy has at least this many entries in data['parts'].",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    args = parser.parse_args(argv)

    index = build_index(
        preprocessed_root=args.preprocessed_root,
        render_root=args.render_root,
        yaml_root=args.yaml_root,
        strict=args.strict,
        verbose=not args.quiet,
        min_explicit_parts=args.min_explicit_parts,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        if args.pretty:
            json.dump(index, handle, indent=4, ensure_ascii=False)
        else:
            json.dump(index, handle, separators=(",", ":"), ensure_ascii=False)

    if not args.quiet:
        total_frames = sum(len(frames) for frames in index.values())
        print(f"[INFO] Wrote {args.output} with {len(index)} actions and {total_frames} frames.")


if __name__ == "__main__":
    main()

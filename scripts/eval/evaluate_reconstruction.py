#!/usr/bin/env python3

"""Evaluate SceneGen-style reconstruction metrics for predicted and GT GLBs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils.metric_utils import (  # noqa: E402
    bbox_iou_3d,
    bbox_overlap_volume,
    bounds_from_mesh_or_scene,
    compute_cd_and_f_score,
    load_mesh_or_scene,
    scene_to_single_mesh,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", type=Path, required=True, help="COM4D inference output directory.")
    ap.add_argument("--gt-dir", type=Path, required=True, help="Ground-truth GLB root.")
    ap.add_argument("--output-dir", type=Path, default=None, help="Defaults to <pred-dir>/eval_reconstruction.")
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--metric", default="l2")
    ap.add_argument(
        "--pred-scene-glob",
        default="dynamic/dynamic_scene_frame_*.glb",
        help="Glob relative to --pred-dir for full-scene frame GLBs.",
    )
    ap.add_argument(
        "--gt-scene-glob",
        default="**/frame_*.glb",
        help="Glob relative to --gt-dir for full-scene GT frame GLBs.",
    )
    ap.add_argument(
        "--skip-object-level",
        action="store_true",
        help="Only compute full-scene metrics.",
    )
    return ap.parse_args()


def frame_index(path: Path) -> int | None:
    patterns = [
        r"dynamic_scene_frame_(\d+)\.glb$",
        r"frame_(\d+)\.glb$",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.name)
        if match is not None:
            return int(match.group(1))
    return None


def dynamic_object_paths(root: Path) -> dict[str, dict[int, Path]]:
    dynamic_dir = root / "dynamic"
    tracks: dict[str, dict[int, Path]] = {}
    if not dynamic_dir.exists():
        return tracks
    for obj_dir in sorted(dynamic_dir.glob("object_*")):
        if not obj_dir.is_dir():
            continue
        frame_map = {}
        for path in sorted(obj_dir.glob("frame_*.glb")):
            idx = frame_index(path)
            if idx is not None:
                frame_map[idx] = path
        if frame_map:
            tracks[obj_dir.name] = frame_map
    return tracks


def frame_paths(root: Path, pattern: str) -> dict[int, Path]:
    paths = {}
    for path in sorted(root.glob(pattern)):
        if any(parent.name.startswith("object_") for parent in path.parents):
            continue
        idx = frame_index(path)
        if idx is not None and idx not in paths:
            paths[idx] = path
    return paths


def mesh_metrics(pred_path: Path, gt_path: Path, num_samples: int, threshold: float, metric: str) -> dict[str, Any]:
    pred_geom = load_mesh_or_scene(pred_path)
    gt_geom = load_mesh_or_scene(gt_path)
    pred_mesh = scene_to_single_mesh(pred_geom)
    gt_mesh = scene_to_single_mesh(gt_geom)
    cd, f_score = compute_cd_and_f_score(pred_mesh, gt_mesh, num_samples=num_samples, threshold=threshold, metric=metric)
    pred_bounds = bounds_from_mesh_or_scene(pred_geom)
    gt_bounds = bounds_from_mesh_or_scene(gt_geom)
    return {
        "chamfer_distance": float(cd),
        "f_score": float(f_score),
        "bbox_iou_3d": bbox_iou_3d(pred_bounds, gt_bounds),
        "bbox_overlap_volume": bbox_overlap_volume(pred_bounds, gt_bounds),
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def add_prefixed_means(summary: dict[str, Any], prefix: str, rows: list[dict[str, Any]]) -> None:
    for key in ("chamfer_distance", "f_score", "bbox_iou_3d", "bbox_overlap_volume"):
        summary[f"{prefix}_{key}_mean"] = mean_or_nan([float(row[key]) for row in rows])


def main() -> None:
    args = parse_args()
    pred_dir = args.pred_dir.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()
    output_dir = (args.output_dir or (pred_dir / "eval_reconstruction")).expanduser().resolve()

    scene_rows = []
    pred_scenes = frame_paths(pred_dir, args.pred_scene_glob)
    gt_scenes = frame_paths(gt_dir, args.gt_scene_glob)
    for frame in sorted(set(pred_scenes).intersection(gt_scenes)):
        row = mesh_metrics(pred_scenes[frame], gt_scenes[frame], args.num_samples, args.threshold, args.metric)
        row.update({"level": "scene", "frame": frame})
        scene_rows.append(row)

    object_rows = []
    if not args.skip_object_level:
        pred_tracks = dynamic_object_paths(pred_dir)
        gt_tracks = dynamic_object_paths(gt_dir)
        for object_id in sorted(set(pred_tracks).intersection(gt_tracks)):
            pred_frames = pred_tracks[object_id]
            gt_frames = gt_tracks[object_id]
            for frame in sorted(set(pred_frames).intersection(gt_frames)):
                row = mesh_metrics(pred_frames[frame], gt_frames[frame], args.num_samples, args.threshold, args.metric)
                row.update({"level": "object", "object_id": object_id, "frame": frame})
                object_rows.append(row)

    summary: dict[str, Any] = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "num_scene_pairs": len(scene_rows),
        "num_object_pairs": len(object_rows),
        "num_samples": args.num_samples,
        "threshold": args.threshold,
    }
    add_prefixed_means(summary, "scene", scene_rows)
    add_prefixed_means(summary, "object", object_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "summary": summary,
                "scene": scene_rows,
                "objects": object_rows,
            },
            f,
            indent=2,
            allow_nan=True,
        )
    write_csv(output_dir / "scene_metrics.csv", scene_rows)
    write_csv(output_dir / "object_metrics.csv", object_rows)
    write_csv(output_dir / "summary.csv", [summary])
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

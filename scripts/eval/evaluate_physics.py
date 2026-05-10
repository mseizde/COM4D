#!/usr/bin/env python3

"""Evaluate simple physical/geometric failure modes for COM4D GLB outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils.metric_utils import (  # noqa: E402
    axis_index,
    bbox_iou_3d,
    bbox_overlap_area_xz,
    bbox_overlap_volume,
    bounds_from_mesh_or_scene,
    center_from_bounds,
    floor_penetration_depth,
    floor_support_error,
    is_floating,
    load_mesh_or_scene,
    size_from_bounds,
    trajectory_acceleration_stats,
    trajectory_speed_stats,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inference-dir", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=None, help="Optional PyBullet physics_metadata.json.")
    ap.add_argument("--output-dir", type=Path, default=None, help="Defaults to <inference-dir>/eval_physics.")
    ap.add_argument("--floor-height", type=float, default=0.0)
    ap.add_argument("--up-axis", choices=("x", "y", "z"), default="y")
    ap.add_argument(
        "--metadata-up-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Up axis used in physics_metadata.json. PyBullet/Blender two-ball metadata is z-up.",
    )
    ap.add_argument("--support-tol", type=float, default=0.05)
    ap.add_argument("--overlap-volume-tol", type=float, default=1e-6)
    ap.add_argument("--fps", type=float, default=None, help="Override FPS. Defaults to metadata fps or 30.")
    return ap.parse_args()


def frame_index(path: Path) -> int:
    match = re.search(r"frame_(\d+)\.glb$", path.name)
    if match is None:
        raise ValueError(f"Could not parse frame index from {path}")
    return int(match.group(1))


def dynamic_object_paths(inference_dir: Path) -> dict[str, list[Path]]:
    dynamic_dir = inference_dir / "dynamic"
    tracks: dict[str, list[Path]] = {}
    if not dynamic_dir.exists():
        return tracks
    for obj_dir in sorted(dynamic_dir.glob("object_*")):
        if obj_dir.is_dir():
            frames = sorted(obj_dir.glob("frame_*.glb"), key=frame_index)
            if frames:
                tracks[obj_dir.name] = frames
    return tracks


def load_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.expanduser().open("r") as f:
        return json.load(f)


def ball_keys(metadata: dict[str, Any] | None) -> list[str]:
    if metadata is None:
        return []
    radii = metadata.get("ball_radii")
    if isinstance(radii, dict):
        return sorted(radii.keys())
    frames = metadata.get("frames", [])
    if frames:
        return sorted(key for key in frames[0] if key.startswith("ball_"))
    return []


def object_to_metadata_key(object_ids: list[str], metadata: dict[str, Any] | None) -> dict[str, str]:
    keys = ball_keys(metadata)
    return {object_id: keys[idx] for idx, object_id in enumerate(sorted(object_ids)) if idx < len(keys)}


def metadata_frame_map(metadata: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if metadata is None:
        return {}
    return {int(frame["frame"]): frame for frame in metadata.get("frames", [])}


def expected_radius(metadata: dict[str, Any] | None, key: str) -> float | None:
    if metadata is None:
        return None
    radii = metadata.get("ball_radii")
    if isinstance(radii, dict) and key in radii:
        return float(radii[key])
    if "ball_radius" in metadata:
        return float(metadata["ball_radius"])
    return None


def metadata_expected_supported(
    metadata_frames: dict[int, dict[str, Any]],
    frame: int,
    key: str,
    radius: float | None,
    metadata_up_axis: str,
    support_tol: float,
) -> bool | None:
    if radius is None or frame not in metadata_frames:
        return None
    obj = metadata_frames[frame].get(key)
    if not isinstance(obj, dict) or "position" not in obj:
        return None
    up = axis_index(metadata_up_axis)
    bottom = float(obj["position"][up]) - radius
    return bool(abs(bottom) <= support_tol)


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


def max_or_nan(values: list[float]) -> float:
    return float(np.max(values)) if values else float("nan")


def finite_rate(flags: list[bool]) -> float:
    return float(np.mean(np.asarray(flags, dtype=np.float32))) if flags else float("nan")


def main() -> None:
    args = parse_args()
    inference_dir = args.inference_dir.expanduser().resolve()
    output_dir = (args.output_dir or (inference_dir / "eval_physics")).expanduser().resolve()
    metadata = load_metadata(args.metadata)
    fps = float(args.fps or (metadata or {}).get("fps", 30.0))

    tracks = dynamic_object_paths(inference_dir)
    metadata_keys = object_to_metadata_key(list(tracks), metadata)
    metadata_frames = metadata_frame_map(metadata)

    frame_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    records_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    centers_by_object: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)

    for object_id, paths in tracks.items():
        metadata_key = metadata_keys.get(object_id)
        radius = expected_radius(metadata, metadata_key) if metadata_key else None
        expected_diameter = None if radius is None else 2.0 * radius

        for path in paths:
            frame = frame_index(path)
            geom = load_mesh_or_scene(path)
            bounds = bounds_from_mesh_or_scene(geom)
            center = center_from_bounds(bounds)
            size = size_from_bounds(bounds)
            support_error = floor_support_error(bounds, args.floor_height, args.up_axis)
            penetration = floor_penetration_depth(bounds, args.floor_height, args.up_axis)
            floating = is_floating(bounds, args.floor_height, args.support_tol, args.up_axis)
            expected_supported = metadata_expected_supported(
                metadata_frames,
                frame,
                metadata_key,
                radius,
                args.metadata_up_axis,
                args.support_tol,
            ) if metadata_key else None
            scale_error = float("nan")
            if expected_diameter is not None and expected_diameter > 0.0:
                scale_error = float(abs(float(np.max(size)) - expected_diameter) / expected_diameter)

            row = {
                "object_id": object_id,
                "metadata_key": metadata_key,
                "frame": frame,
                "path": str(path),
                "support_error": support_error,
                "floor_penetration_depth": penetration,
                "is_floating": floating,
                "expected_floor_supported": expected_supported,
                "support_consistent": None if expected_supported is None else (abs(support_error) <= args.support_tol),
                "bbox_min_x": float(bounds[0, 0]),
                "bbox_min_y": float(bounds[0, 1]),
                "bbox_min_z": float(bounds[0, 2]),
                "bbox_max_x": float(bounds[1, 0]),
                "bbox_max_y": float(bounds[1, 1]),
                "bbox_max_z": float(bounds[1, 2]),
                "bbox_size_x": float(size[0]),
                "bbox_size_y": float(size[1]),
                "bbox_size_z": float(size[2]),
                "scale_error": scale_error,
            }
            object_rows.append(row)
            records_by_frame[frame].append({"object_id": object_id, "bounds": bounds})
            centers_by_object[object_id].append((frame, center))

    pair_rows: list[dict[str, Any]] = []
    collision_flags = []
    for frame, records in sorted(records_by_frame.items()):
        for i, rec_a in enumerate(records):
            for rec_b in records[i + 1 :]:
                overlap_volume = bbox_overlap_volume(rec_a["bounds"], rec_b["bounds"])
                overlaps = overlap_volume > args.overlap_volume_tol
                collision_flags.append(overlaps)
                pair_rows.append(
                    {
                        "frame": frame,
                        "object_a": rec_a["object_id"],
                        "object_b": rec_b["object_id"],
                        "bbox_overlap_volume": overlap_volume,
                        "bbox_overlap_area_xz": bbox_overlap_area_xz(rec_a["bounds"], rec_b["bounds"]),
                        "bbox_iou_3d": bbox_iou_3d(rec_a["bounds"], rec_b["bounds"]),
                        "bbox_collision": overlaps,
                    }
                )

    trajectory_rows: list[dict[str, Any]] = []
    for object_id, items in sorted(centers_by_object.items()):
        items = sorted(items, key=lambda item: item[0])
        frames = [frame for frame, _ in items]
        centers = np.stack([center for _, center in items], axis=0)
        speed = trajectory_speed_stats(centers, fps)
        accel = trajectory_acceleration_stats(centers, fps)
        expected_frames = set(range(min(frames), max(frames) + 1)) if frames else set()
        missing = len(expected_frames.difference(frames))
        trajectory_rows.append(
            {
                "object_id": object_id,
                "num_frames": len(frames),
                "first_frame": min(frames) if frames else None,
                "last_frame": max(frames) if frames else None,
                "missing_frame_count": missing,
                "missing_frame_rate": float(missing / len(expected_frames)) if expected_frames else 0.0,
                "speed_mean": speed["mean"],
                "speed_max": speed["max"],
                "speed_std": speed["std"],
                "acceleration_mean": accel["mean"],
                "acceleration_max": accel["max"],
                "acceleration_std": accel["std"],
            }
        )

    penetration_values = [float(row["floor_penetration_depth"]) for row in object_rows]
    floating_flags = [bool(row["is_floating"]) for row in object_rows]
    support_flags = [bool(row["support_consistent"]) for row in object_rows if row["support_consistent"] is not None]
    scale_values = [float(row["scale_error"]) for row in object_rows if np.isfinite(float(row["scale_error"]))]

    summary = {
        "inference_dir": str(inference_dir),
        "metadata": str(args.metadata.expanduser().resolve()) if args.metadata else None,
        "num_objects": len(tracks),
        "num_object_frames": len(object_rows),
        "num_pair_frames": len(pair_rows),
        "floor_penetration_mean": mean_or_nan(penetration_values),
        "floor_penetration_max": max_or_nan(penetration_values),
        "floating_rate": finite_rate(floating_flags),
        "bbox_collision_rate": finite_rate(collision_flags),
        "support_consistency_rate": finite_rate(support_flags),
        "scale_error_mean": mean_or_nan(scale_values),
        "scale_error_max": max_or_nan(scale_values),
        "trajectory_speed_max_mean": mean_or_nan([float(row["speed_max"]) for row in trajectory_rows]),
        "trajectory_acceleration_max_mean": mean_or_nan([float(row["acceleration_max"]) for row in trajectory_rows]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "summary": summary,
                "objects": object_rows,
                "pairs": pair_rows,
                "trajectories": trajectory_rows,
            },
            f,
            indent=2,
            allow_nan=True,
        )
    write_csv(output_dir / "object_metrics.csv", object_rows)
    write_csv(output_dir / "pair_metrics.csv", pair_rows)
    write_csv(output_dir / "trajectory_metrics.csv", trajectory_rows)
    write_csv(output_dir / "summary.csv", [summary])
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""De-reference physics frames where any ball's amodal mask touches the sensor border."""

import argparse
import copy
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--apply", action="store_true", help="Atomically replace --manifest; otherwise dry-run.")
    p.add_argument("--min-frames", type=int, default=4)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument(
        "--drop-fully-occluded", action="store_true",
        help="Also remove visible-area-zero balls that remain in the amodal sensor view. Not recommended for memory training.",
    )
    p.add_argument(
        "--backup-dir", type=Path,
        default=Path("/data/mseizde/com4d/outputs/data_cleanup"),
    )
    p.add_argument("--report", type=Path, default=None)
    return p.parse_args()


def frame_index(frame, fallback):
    match = re.search(r"frame_(\d+)", Path(frame.get("surface_path", "")).as_posix())
    return int(match.group(1)) if match else int(frame.get("frame_index", fallback))


def mask_stats(path):
    if not path:
        return 0, False
    values = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
    area = int(values.sum())
    touches = bool(
        area and (
            values[0].any() or values[-1].any()
            or values[:, 0].any() or values[:, -1].any()
        )
    )
    return area, touches


def enrich_sequence(frames, mask_cache, drop_fully_occluded=False):
    records = []
    maxima = {}
    non_border_maxima = {}
    max_frame_index = max(frame_index(frame, i) for i, frame in enumerate(frames))
    for fallback, frame in enumerate(frames):
        names = frame.get("object_names", [])
        visible_paths = frame.get("visible_mask_paths", [])
        amodal_paths = frame.get("amodal_mask_paths", [])
        if not (len(names) == len(visible_paths) == len(amodal_paths)):
            raise ValueError(f"mask/object length mismatch in {frame.get('surface_path')}")
        visible_area, amodal_area, touches = [], [], []
        visibility = frame.get("visibility", [0.0] * len(names))
        for object_index, (name, _visible_path, amodal_path) in enumerate(
            zip(names, visible_paths, amodal_paths)
        ):
            if name.startswith("ball_"):
                a_area, border = mask_cache[amodal_path]
                v_area = int(round(float(visibility[object_index]) * a_area))
                maxima[name] = max(maxima.get(name, 0), a_area)
                if not border:
                    non_border_maxima[name] = max(non_border_maxima.get(name, 0), a_area)
            else:
                # Non-ball masks do not participate in the removal policy.
                a_area, v_area, border = 0, 0, False
            visible_area.append(v_area)
            amodal_area.append(a_area)
            touches.append(border)
        bad_ball = any(
            (
                border
                or amodal_area[object_index] == 0
                or (drop_fully_occluded and visible_area[object_index] == 0)
            )
            for object_index, (name, border) in enumerate(zip(names, touches))
            if name.startswith("ball_")
        )
        records.append((frame, visible_area, amodal_area, touches, bad_ball, frame_index(frame, fallback)))

    output = []
    removed = []
    for frame, visible_area, amodal_area, touches, bad_ball, index in records:
        if bad_ball:
            removed.append(index)
            continue
        value = copy.deepcopy(frame)
        names = value["object_names"]
        sensor_coverage = []
        quality = []
        visibility = value.get("visibility", [0.0] * len(names))
        for object_index, (name, area) in enumerate(zip(names, amodal_area)):
            expected = non_border_maxima.get(name, 0) or maxima.get(name, 0)
            coverage = (
                min(max(area / max(expected, 1), 0.0), 1.0)
                if name.startswith("ball_") else 1.0
            )
            sensor_coverage.append(coverage)
            quality.append(min(max(float(visibility[object_index]) * coverage, 0.0), 1.0))
        value.update({
            "frame_index": index,
            "frame_time": index / max(max_frame_index, 1),
            "visible_mask_area": visible_area,
            "amodal_mask_area": amodal_area,
            "mask_touches_border": touches,
            "sensor_coverage_proxy": sensor_coverage,
            "observation_quality": quality,
        })
        output.append(value)
    return output, removed


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    ball_mask_paths = sorted({
        path
        for frames in manifest.values()
        for frame in frames
        for name, path in zip(
            frame.get("object_names", []), frame.get("amodal_mask_paths", [])
        )
        if name.startswith("ball_") and path
    })
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        mask_cache = dict(zip(ball_mask_paths, executor.map(mask_stats, ball_mask_paths)))

    cleaned = {}
    report = {
        "source": str(args.manifest.resolve()),
        "policy": (
            "remove a frame when any ball_* amodal mask touches a sensor border "
            "or has zero amodal area"
            + (" or has zero visible area" if args.drop_fully_occluded else "")
        ),
        "sequences_before": len(manifest),
        "frames_before": sum(len(frames) for frames in manifest.values()),
        "sequences": {},
    }
    dropped_sequences = []
    for name, frames in manifest.items():
        retained, removed = enrich_sequence(
            frames, mask_cache, drop_fully_occluded=args.drop_fully_occluded
        )
        if len(retained) < args.min_frames:
            dropped_sequences.append(name)
        else:
            cleaned[name] = retained
        report["sequences"][name] = {
            "before": len(frames), "after": len(retained),
            "removed_frame_indices": removed,
            "dropped": len(retained) < args.min_frames,
        }
    report.update({
        "sequences_after": len(cleaned),
        "frames_after": sum(len(frames) for frames in cleaned.values()),
        "dropped_sequences": dropped_sequences,
    })

    report_path = args.report or (
        args.backup_dir / f"{args.manifest.stem}_border_cleanup_report.json"
    )
    print(json.dumps({
        key: report[key] for key in (
            "sequences_before", "sequences_after", "frames_before", "frames_after",
            "dropped_sequences",
        )
    }, indent=2))

    if not args.apply:
        print("Dry run; pass --apply to replace the manifest.")
        return

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.backup_dir / f"{args.manifest.stem}_before_border_cleanup_{stamp}.json"
    shutil.copy2(args.manifest, backup)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(cleaned, indent=2))
    temporary.replace(args.manifest)
    print(f"Backup: {backup}")
    print(f"Report: {report_path}")
    print(f"Updated: {args.manifest}")


if __name__ == "__main__":
    main()

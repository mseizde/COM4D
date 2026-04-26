#!/usr/bin/env python3
"""
Filter a DeformingThings manifest by rendered-image foreground occupancy.

This is intended for cleaning frame-based manifests like dataset_json/deformingthings.json
after preprocessing produced blank or nearly blank fixed-camera renderings for some sequences.

Example:
    python datasets/preprocess/filter_deformingthings_by_render_quality.py \
        --input /data/mseizde/com4d/COM4D/dataset_json/deformingthings.json \
        --output /data/mseizde/com4d/COM4D/dataset_json/deformingthings.json \
        --report /data/mseizde/com4d/COM4D/dataset_json/deformingthings_render_qc_report.json \
        --image-substring /animals_render/ \
        --min-foreground-ratio 0.005 \
        --drop-sequence-bad-ratio 0.9 \
        --min-sequence-frames 8 \
        --workers 32 \
        --pretty
"""

import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Filter DeformingThings manifest by render occupancy.")
    parser.add_argument("--input", required=True, help="Input DeformingThings JSON manifest.")
    parser.add_argument("--output", required=True, help="Output cleaned JSON manifest.")
    parser.add_argument("--report", required=True, help="Output QC report JSON.")
    parser.add_argument(
        "--image-substring",
        action="append",
        default=[],
        help="Only inspect frames whose image_path contains this substring. Repeatable.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=250,
        help="Pixel values >= this threshold in all RGB channels are treated as background.",
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=0.005,
        help="Frames at or below this occupancy are treated as bad.",
    )
    parser.add_argument(
        "--drop-sequence-bad-ratio",
        type=float,
        default=0.9,
        help="Drop a whole sequence if this fraction of inspected frames are bad.",
    )
    parser.add_argument(
        "--min-sequence-frames",
        type=int,
        default=8,
        help="Drop a sequence if fewer than this many frames remain after frame filtering.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="Worker count for image scanning.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON outputs.")
    return parser.parse_args()


def should_inspect(image_path: str, substrings: list[str]) -> bool:
    if not substrings:
        return True
    return any(token in image_path for token in substrings)


def measure_foreground(task):
    sequence_name, frame_index, image_path, white_threshold = task
    try:
        img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        foreground = np.any(img < white_threshold, axis=2)
        ratio = float(foreground.mean())
        return {
            "sequence_name": sequence_name,
            "frame_index": frame_index,
            "image_path": image_path,
            "foreground_ratio": ratio,
            "error": None,
        }
    except Exception as exc:
        return {
            "sequence_name": sequence_name,
            "frame_index": frame_index,
            "image_path": image_path,
            "foreground_ratio": None,
            "error": repr(exc),
        }


def percentile(sorted_values: list[float], p: float):
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * p))))
    return float(sorted_values[idx])


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    with open(input_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError("Expected DeformingThings manifest root to be a dict of sequence_name -> frames")

    tasks = []
    for sequence_name, frames in manifest.items():
        if not isinstance(frames, list):
            continue
        for frame_index, frame in enumerate(frames):
            image_path = frame.get("image_path")
            if not image_path:
                continue
            if should_inspect(image_path, args.image_substring):
                tasks.append((sequence_name, frame_index, image_path, args.white_threshold))

    with Pool(processes=max(1, args.workers)) as pool:
        results = list(pool.imap_unordered(measure_foreground, tasks, chunksize=16))

    metrics_by_sequence = defaultdict(dict)
    foreground_values = []
    error_count = 0
    for item in results:
        metrics_by_sequence[item["sequence_name"]][item["frame_index"]] = item
        if item["foreground_ratio"] is not None:
            foreground_values.append(item["foreground_ratio"])
        else:
            error_count += 1

    foreground_values.sort()

    cleaned_manifest = {}
    dropped_sequences = []
    trimmed_sequences = []
    untouched_sequences = 0
    total_original_frames = 0
    total_kept_frames = 0
    total_dropped_frames = 0
    total_inspected_frames = len(results)
    total_bad_frames = 0

    for sequence_name, frames in manifest.items():
        total_original_frames += len(frames)
        inspected_metrics = metrics_by_sequence.get(sequence_name, {})
        if not inspected_metrics:
            cleaned_manifest[sequence_name] = frames
            untouched_sequences += 1
            total_kept_frames += len(frames)
            continue

        kept_frames = []
        bad_frame_paths = []
        good_inspected_count = 0
        bad_inspected_count = 0
        sequence_ratios = []

        for frame_index, frame in enumerate(frames):
            metric = inspected_metrics.get(frame_index)
            if metric is None:
                kept_frames.append(frame)
                continue
            ratio = metric["foreground_ratio"]
            sequence_ratios.append(ratio)
            is_bad = metric["error"] is not None or ratio is None or ratio <= args.min_foreground_ratio
            if is_bad:
                bad_inspected_count += 1
                bad_frame_paths.append(metric["image_path"])
            else:
                good_inspected_count += 1
                kept_frames.append(frame)

        total_bad_frames += bad_inspected_count
        inspected_count = good_inspected_count + bad_inspected_count
        bad_ratio = (bad_inspected_count / inspected_count) if inspected_count else 0.0

        if bad_ratio >= args.drop_sequence_bad_ratio or len(kept_frames) < args.min_sequence_frames:
            dropped_sequences.append(
                {
                    "sequence_name": sequence_name,
                    "original_frames": len(frames),
                    "kept_frames": len(kept_frames),
                    "dropped_frames": len(frames),
                    "inspected_frames": inspected_count,
                    "bad_frames": bad_inspected_count,
                    "bad_ratio": bad_ratio,
                    "foreground_ratio_min": min((v for v in sequence_ratios if v is not None), default=None),
                    "foreground_ratio_median": percentile(sorted(v for v in sequence_ratios if v is not None), 0.5),
                    "foreground_ratio_max": max((v for v in sequence_ratios if v is not None), default=None),
                    "sample_bad_frame_paths": bad_frame_paths[:8],
                }
            )
            total_dropped_frames += len(frames)
            continue

        cleaned_manifest[sequence_name] = kept_frames
        total_kept_frames += len(kept_frames)
        dropped_here = len(frames) - len(kept_frames)
        total_dropped_frames += dropped_here
        if dropped_here > 0:
            trimmed_sequences.append(
                {
                    "sequence_name": sequence_name,
                    "original_frames": len(frames),
                    "kept_frames": len(kept_frames),
                    "dropped_frames": dropped_here,
                    "inspected_frames": inspected_count,
                    "bad_frames": bad_inspected_count,
                    "bad_ratio": bad_ratio,
                    "foreground_ratio_min": min((v for v in sequence_ratios if v is not None), default=None),
                    "foreground_ratio_median": percentile(sorted(v for v in sequence_ratios if v is not None), 0.5),
                    "foreground_ratio_max": max((v for v in sequence_ratios if v is not None), default=None),
                    "sample_bad_frame_paths": bad_frame_paths[:8],
                }
            )

    dropped_sequences.sort(key=lambda x: (-x["bad_ratio"], -x["dropped_frames"], x["sequence_name"]))
    trimmed_sequences.sort(key=lambda x: (-x["dropped_frames"], -x["bad_ratio"], x["sequence_name"]))

    report = {
        "input_manifest": str(input_path.resolve()),
        "output_manifest": str(output_path.resolve()),
        "filter": {
            "image_substring": args.image_substring,
            "white_threshold": args.white_threshold,
            "min_foreground_ratio": args.min_foreground_ratio,
            "drop_sequence_bad_ratio": args.drop_sequence_bad_ratio,
            "min_sequence_frames": args.min_sequence_frames,
            "workers": args.workers,
        },
        "summary": {
            "original_sequences": len(manifest),
            "cleaned_sequences": len(cleaned_manifest),
            "dropped_sequences": len(dropped_sequences),
            "trimmed_sequences": len(trimmed_sequences),
            "untouched_sequences": untouched_sequences,
            "original_frames": total_original_frames,
            "cleaned_frames": total_kept_frames,
            "dropped_frames": total_dropped_frames,
            "inspected_frames": total_inspected_frames,
            "bad_frames": total_bad_frames,
            "error_frames": error_count,
        },
        "foreground_ratio_distribution": {
            "min": percentile(foreground_values, 0.0),
            "p01": percentile(foreground_values, 0.01),
            "p05": percentile(foreground_values, 0.05),
            "p10": percentile(foreground_values, 0.10),
            "p25": percentile(foreground_values, 0.25),
            "p50": percentile(foreground_values, 0.50),
            "p75": percentile(foreground_values, 0.75),
            "p90": percentile(foreground_values, 0.90),
            "max": percentile(foreground_values, 1.0),
        },
        "dropped_sequences": dropped_sequences,
        "trimmed_sequences": trimmed_sequences,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(cleaned_manifest, f, indent=4, ensure_ascii=False)
        else:
            json.dump(cleaned_manifest, f, separators=(",", ":"), ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(report, f, indent=4, ensure_ascii=False)
        else:
            json.dump(report, f, separators=(",", ":"), ensure_ascii=False)

    print(
        "[INFO] Wrote cleaned manifest with "
        f"{len(cleaned_manifest)} sequences / {total_kept_frames} frames "
        f"(dropped {len(dropped_sequences)} sequences and {total_dropped_frames} frames total)."
    )
    print(f"[INFO] QC report written to {report_path.resolve()}")


if __name__ == "__main__":
    main()

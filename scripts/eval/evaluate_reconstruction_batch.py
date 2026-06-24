#!/usr/bin/env python3
"""Batch reconstruction evaluation over selected existing prediction exports."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
EVALUATOR = Path(__file__).with_name("evaluate_reconstruction.py")
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "physics_compare"
DEFAULT_GT_ROOT = Path("/mnt/mocap_b/work/com4d/datasets/synthetic/physics_compare/gt_raw")
SAMPLE_PATTERN = re.compile(r"^(?P<case>.+)_eval_(?P<index>\d+)$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--predictions-root", type=Path, default=None)
    ap.add_argument("--metrics-root", type=Path, default=None)
    ap.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    ap.add_argument(
        "--case",
        action="append",
        default=None,
        help="Case prefix such as ball_drop, rolling_occluder, two_ball, or wall_impact. Repeatable; defaults to all.",
    )
    ap.add_argument(
        "--first-n-per-case",
        type=int,
        default=0,
        help="Select the first N numeric *_eval_* samples per case; 0 selects all.",
    )
    ap.add_argument("--sample", action="append", default=None, help="Exact sample name. Repeatable; bypasses case selection.")
    ap.add_argument("--model", action="append", required=True, help="Existing prediction model tag. Repeatable.")
    ap.add_argument(
        "--alignment",
        choices=("none", "translation", "similarity", "first_frame_similarity"),
        default="similarity",
    )
    ap.add_argument("--object-assignment", choices=("fixed", "best"), default="best")
    ap.add_argument("--alignment-samples-per-frame", type=int, default=1024)
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--skip-voxel-iou", action="store_true")
    ap.add_argument(
        "--include-raw-metrics",
        action="store_true",
        help="Also compute unaligned metrics. By default only aligned metrics are computed.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Replace existing reconstruction metrics.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fail-on-missing", action="store_true", help="Treat absent model/sample exports as failures.")
    return ap.parse_args()


def has_dynamic_frames(path: Path) -> bool:
    return any((path / "dynamic").glob("dynamic_scene_frame_*.glb"))


def resolve_export_dir(model_dir: Path) -> Path | None:
    candidates = []
    if has_dynamic_frames(model_dir):
        candidates.append(model_dir)
    candidates.extend(
        child for child in sorted(model_dir.iterdir()) if child.is_dir() and has_dynamic_frames(child)
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        selected = candidates[-1]
        print(
            f"Warning: {model_dir} has {len(candidates)} prediction exports; using lexicographically latest {selected.name}.",
            flush=True,
        )
        return selected
    return candidates[0]


def discover_samples(predictions_root: Path, cases: set[str] | None, first_n: int) -> list[str]:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for path in predictions_root.iterdir():
        if not path.is_dir():
            continue
        match = SAMPLE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        case = match.group("case")
        if cases is not None and case not in cases:
            continue
        grouped.setdefault(case, []).append((int(match.group("index")), path.name))

    selected = []
    for case in sorted(grouped):
        samples = [name for _, name in sorted(grouped[case])]
        selected.extend(samples[:first_n] if first_n > 0 else samples)
    return selected


def build_command(
    args: argparse.Namespace,
    pred_dir: Path,
    gt_dir: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--pred-dir",
        str(pred_dir),
        "--gt-dir",
        str(gt_dir),
        "--output-dir",
        str(output_dir),
        "--alignment",
        args.alignment,
        "--object-assignment",
        args.object_assignment,
        "--alignment-samples-per-frame",
        str(args.alignment_samples_per_frame),
        "--num-samples",
        str(args.num_samples),
        "--threshold",
        str(args.threshold),
    ]
    if not args.include_raw_metrics:
        command.append("--skip-raw-metrics")
    if args.skip_voxel_iou:
        command.append("--skip-voxel-iou")
    return command


def main() -> None:
    args = parse_args()
    if args.first_n_per_case < 0:
        raise SystemExit("--first-n-per-case must be non-negative")
    if args.alignment_samples_per_frame <= 0:
        raise SystemExit("--alignment-samples-per-frame must be positive")

    eval_root = args.eval_root.expanduser().resolve()
    predictions_root = (
        args.predictions_root.expanduser().resolve()
        if args.predictions_root is not None
        else eval_root / "predictions"
    )
    metrics_root = (
        args.metrics_root.expanduser().resolve()
        if args.metrics_root is not None
        else eval_root / "metrics"
    )
    gt_root = args.gt_root.expanduser().resolve()
    if not predictions_root.is_dir():
        raise SystemExit(f"Predictions root does not exist: {predictions_root}")
    if not gt_root.is_dir():
        raise SystemExit(f"GT root does not exist: {gt_root}")

    if args.sample:
        samples = list(dict.fromkeys(args.sample))
    else:
        cases = set(args.case) if args.case else None
        samples = discover_samples(predictions_root, cases, args.first_n_per_case)
    models = list(dict.fromkeys(args.model))
    if not samples:
        raise SystemExit("No samples matched the requested selection")

    print(f"Selected {len(samples)} samples × {len(models)} models ({len(samples) * len(models)} combinations).")
    wrote = skipped = missing = failed = 0
    for sample in samples:
        gt_dir = gt_root / sample
        for model in models:
            model_dir = predictions_root / sample / model
            output_dir = metrics_root / sample / model / "reconstruction"
            if not gt_dir.is_dir() or not model_dir.is_dir():
                missing += 1
                print(f"[missing] {sample}/{model}", flush=True)
                continue
            pred_dir = resolve_export_dir(model_dir)
            if pred_dir is None:
                missing += 1
                print(f"[missing:dynamic] {sample}/{model}", flush=True)
                continue
            metrics_path = output_dir / "metrics.json"
            if metrics_path.exists() and not args.overwrite:
                skipped += 1
                print(f"[skip:exists] {sample}/{model}", flush=True)
                continue

            command = build_command(args, pred_dir, gt_dir, output_dir)
            print(f"[run] {sample}/{model}: {shlex.join(command)}", flush=True)
            if args.dry_run:
                skipped += 1
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if result.returncode == 0:
                wrote += 1
            else:
                failed += 1
                print(f"[failed:{result.returncode}] {sample}/{model}", flush=True)

    print(f"Done. wrote={wrote} skipped={skipped} missing={missing} failed={failed}")
    if failed or (missing and args.fail_on_missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

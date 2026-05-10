#!/usr/bin/env python3

"""Run COM4D inference and/or evaluation scripts for one benchmark entry."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-key", default=None, help="infer.json key to run before evaluation.")
    ap.add_argument("--image-size", default="518")
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--inference-dir", type=Path, default=None, help="Existing inference output directory.")
    ap.add_argument("--metadata", type=Path, default=None, help="Optional physics_metadata.json for physics metrics.")
    ap.add_argument("--gt-dir", type=Path, default=None, help="Optional GT GLB root for reconstruction metrics.")
    ap.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "benchmarks")
    ap.add_argument("--benchmark-csv", type=Path, default=None)
    ap.add_argument("--floor-height", type=float, default=0.0)
    ap.add_argument("--up-axis", choices=("x", "y", "z"), default="y")
    ap.add_argument("--metadata-up-axis", choices=("x", "y", "z"), default="z")
    ap.add_argument("--support-tol", type=float, default=0.05)
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def newest_inference_dir(config_key: str) -> Path:
    root = REPO_ROOT.parent / "outputs" / "inference"
    candidates = [path for path in root.glob(f"{config_key}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No inference directories found for {config_key!r} under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    return dict(data.get("summary", {}))


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="") as f:
            existing_rows = list(csv.DictReader(f))
    keys = sorted(set(row).union(*(set(existing) for existing in existing_rows)))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow(existing)
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.inference_dir is None and (args.skip_inference or args.config_key is None):
        raise ValueError("Provide --inference-dir, or provide --config-key without --skip-inference.")

    if not args.skip_inference:
        if args.config_key is None:
            raise ValueError("--config-key is required when inference is enabled.")
        run(["bash", "sh/infer.sh", args.config_key, str(args.image_size)], cwd=REPO_ROOT, dry_run=args.dry_run)
        inference_dir = newest_inference_dir(args.config_key) if not args.dry_run else REPO_ROOT.parent / "outputs" / "inference" / f"{args.config_key}_DRY_RUN"
    else:
        inference_dir = args.inference_dir.expanduser().resolve()

    run_name = inference_dir.name
    output_root = args.output_root.expanduser().resolve()
    run_output = output_root / run_name
    physics_output = run_output / "physics"
    reconstruction_output = run_output / "reconstruction"

    physics_cmd = [
        sys.executable,
        str(EVAL_DIR / "evaluate_physics.py"),
        "--inference-dir",
        str(inference_dir),
        "--output-dir",
        str(physics_output),
        "--floor-height",
        str(args.floor_height),
        "--up-axis",
        args.up_axis,
        "--metadata-up-axis",
        args.metadata_up_axis,
        "--support-tol",
        str(args.support_tol),
    ]
    if args.metadata is not None:
        physics_cmd.extend(["--metadata", str(args.metadata.expanduser().resolve())])
    run(physics_cmd, cwd=REPO_ROOT, dry_run=args.dry_run)

    if args.gt_dir is not None:
        reconstruction_cmd = [
            sys.executable,
            str(EVAL_DIR / "evaluate_reconstruction.py"),
            "--pred-dir",
            str(inference_dir),
            "--gt-dir",
            str(args.gt_dir.expanduser().resolve()),
            "--output-dir",
            str(reconstruction_output),
            "--num-samples",
            str(args.num_samples),
            "--threshold",
            str(args.threshold),
        ]
        run(reconstruction_cmd, cwd=REPO_ROOT, dry_run=args.dry_run)

    if args.dry_run:
        return

    physics_summary = load_summary(physics_output / "metrics.json")
    reconstruction_summary = load_summary(reconstruction_output / "metrics.json") if args.gt_dir is not None else {}
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_key": args.config_key,
        "run_name": run_name,
        "inference_dir": str(inference_dir),
    }
    row.update({f"physics_{key}": value for key, value in physics_summary.items()})
    row.update({f"reconstruction_{key}": value for key, value in reconstruction_summary.items()})

    benchmark_csv = args.benchmark_csv or (output_root / "benchmark.csv")
    append_row(benchmark_csv.expanduser().resolve(), row)
    print(f"Wrote benchmark row to {benchmark_csv}")


if __name__ == "__main__":
    main()

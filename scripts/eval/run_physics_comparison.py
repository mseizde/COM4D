#!/usr/bin/env python3

"""Run a base-vs-physics COM4D comparison on one synthetic physics sequence."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
TWO_BALL_PIPELINE = REPO_ROOT / "datasets" / "synthetic" / "two_ball_test" / "run_physics_pipeline.py"
PREPARE_INPUT = SCRIPT_DIR / "prepare_physics_inference_input.py"
EVALUATE_PHYSICS = SCRIPT_DIR / "evaluate_physics.py"
EVALUATE_RECONSTRUCTION = SCRIPT_DIR / "evaluate_reconstruction.py"

DEFAULT_DATASET_ROOT = Path("/mnt/mocap_b/work/com4d/datasets/synthetic/two_ball_compare")
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "two_ball_compare"
DEFAULT_BLENDER = PROJECT_ROOT / "tools" / "blender-3.6.5-linux-x64" / "blender"
DEFAULT_BASE_TRANSFORMER = REPO_ROOT / "pretrained_weights" / "COM4D" / "transformer_ema"
DEFAULT_PHYSICS_TRANSFORMER = PROJECT_ROOT / "outputs" / "ckpts" / "com4d_sdemb_mf8_mp8_nt512_30400" / "checkpoints" / "003000"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--sample-name", default="two_ball_eval_000")
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument(
        "--scenario",
        choices=("two_ball_collision", "ball_drop", "rolling_occluder", "wall_impact"),
        default="two_ball_collision",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resolution", type=int, default=518)
    ap.add_argument("--render-samples", type=int, default=32)
    ap.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    ap.add_argument("--blender-device", choices=("AUTO", "CPU", "GPU"), default="CPU")
    ap.add_argument("--save-depth", action="store_true", help="Render GT depth EXR files for generated sequences.")
    ap.add_argument("--save-normals", action="store_true", help="Render GT normal EXR files for generated sequences.")
    ap.add_argument("--position-jitter", type=float, default=0.0)
    ap.add_argument("--height-jitter", type=float, default=0.0)
    ap.add_argument("--velocity-jitter", type=float, default=0.0)
    ap.add_argument("--mass-jitter", type=float, default=0.0)
    ap.add_argument("--force-generate", action="store_true", help="Regenerate the raw GT sequence.")
    ap.add_argument("--skip-generation", action="store_true", help="Require an existing raw GT sequence.")
    ap.add_argument("--input-mode", choices=("symlink", "copy"), default="copy")
    ap.add_argument("--base-transformer", type=Path, default=DEFAULT_BASE_TRANSFORMER)
    ap.add_argument("--physics-transformer", type=Path, default=DEFAULT_PHYSICS_TRANSFORMER)
    ap.add_argument(
        "--base-pred-dir",
        type=Path,
        default=None,
        help="Existing base inference output to reuse. Defaults to newest <base-tag>_* under predictions.",
    )
    ap.add_argument(
        "--run-base-inference",
        action="store_true",
        help="Generate a fresh base prediction instead of reusing an existing one.",
    )
    ap.add_argument("--base-tag", default="base")
    ap.add_argument("--physics-tag", default="physics_30400_003000")
    ap.add_argument("--base-weights-dir", default="pretrained_weights/TripoSG")
    ap.add_argument("--num-tokens", type=int, default=1024)
    ap.add_argument("--dynamic-ar-block-size", type=int, default=4)
    ap.add_argument("--dynamic-max-memory-frames", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def run(cmd: list[str], cwd: Path = REPO_ROOT, dry_run: bool = False) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=True)


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return dict(data.get("summary", {}))


def load_json_rows(path: Path, key: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get(key, [])
    return list(rows) if isinstance(rows, list) else []


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.is_file() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    keys = sorted(set(row).union(*(set(item) for item in existing)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.is_file() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    keys = sorted(set().union(*(set(item) for item in existing), *(set(item) for item in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)


def newest_tag_dir(root: Path, tag: str, before: set[Path]) -> Path:
    candidates = [path for path in root.glob(f"{tag}_*") if path.is_dir() and path not in before]
    if not candidates:
        candidates = [path for path in root.glob(f"{tag}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No inference outputs found for tag {tag!r} under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def newest_existing_tag_dir(root: Path, tag: str) -> Path:
    candidates = [path for path in root.glob(f"{tag}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No existing inference outputs found for tag {tag!r} under {root}. "
            "Pass --base-pred-dir or --run-base-inference."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def generate_gt(args: argparse.Namespace, raw_dir: Path) -> None:
    if raw_dir.exists() and args.force_generate:
        shutil.rmtree(raw_dir)
    if raw_dir.exists() and (raw_dir / "physics_metadata.json").is_file() and not args.force_generate:
        print(f"Reusing existing GT sequence: {raw_dir}")
        return
    if args.skip_generation:
        raise FileNotFoundError(f"GT sequence does not exist: {raw_dir}")

    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        TWO_BALL_PIPELINE,
        "--output-dir",
        raw_dir,
        "--blender-bin",
        args.blender_bin,
        "--scenario",
        args.scenario,
        "--num-frames",
        args.num_frames,
        "--seed",
        args.seed,
        "--random-view",
        "--view-seed",
        args.seed,
        "--random-light",
        "--light-seed",
        args.seed,
        "--resolution",
        args.resolution,
        "--samples",
        args.render_samples,
        "--device",
        args.blender_device,
        "--position-jitter",
        args.position_jitter,
        "--height-jitter",
        args.height_jitter,
        "--velocity-jitter",
        args.velocity_jitter,
        "--mass-jitter",
        args.mass_jitter,
    ]
    if args.save_depth:
        cmd.append("--save-depth")
    if args.save_normals:
        cmd.append("--save-normals")
    run(cmd, dry_run=args.dry_run)


def prepare_input(args: argparse.Namespace, raw_dir: Path, input_dir: Path) -> None:
    run(
        [
            sys.executable,
            PREPARE_INPUT,
            "--raw-dir",
            raw_dir,
            "--output-dir",
            input_dir,
            "--mode",
            args.input_mode,
            "--overwrite",
        ],
        dry_run=args.dry_run,
    )


def dynamic_part_count(input_dir: Path) -> int:
    mask_dir = input_dir / "masks"
    if not mask_dir.is_dir():
        return 2
    first_frame_masks = sorted(mask_dir.glob("frame_0000_object_*.png"))
    if first_frame_masks:
        return len(first_frame_masks)
    object_ids = set()
    for path in mask_dir.glob("frame_*_object_*.png"):
        try:
            object_ids.add(path.stem.rsplit("_object_", 1)[1])
        except IndexError:
            pass
    return len(object_ids) if object_ids else 2


def run_inference(args: argparse.Namespace, tag: str, transformer: Path, input_dir: Path, predictions_dir: Path) -> Path:
    before = set(predictions_dir.glob(f"{tag}_*")) if predictions_dir.exists() else set()
    run(
        [
            sys.executable,
            "src/inference_com4d.py",
            "--frames_dir",
            input_dir / "frames",
            "--masks_dir",
            input_dir / "masks",
            "--masks_static_dir",
            input_dir / "masks_static",
            "--output_dir",
            predictions_dir,
            "--tag",
            tag,
            "--transformer_dir",
            transformer,
            "--base_weights_dir",
            args.base_weights_dir,
            "--num_tokens",
            args.num_tokens,
            "--first_frame_index",
            0,
            "--frames_start_idx",
            0,
            "--frames_end_idx",
            args.num_frames,
            "--frame_stride",
            1,
            "--scene_num_parts",
            0,
            "--dynamic_num_parts",
            dynamic_part_count(input_dir),
            "--dynamic_ar_block_size",
            args.dynamic_ar_block_size,
            "--dynamic_max_memory_frames",
            args.dynamic_max_memory_frames,
            "--object_only_condition",
            "--animation",
            "--image_size",
            args.image_size,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--no-render_predicted_room",
            "--no-room_augment_animations",
        ],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return predictions_dir / f"{tag}_DRY_RUN"
    return newest_tag_dir(predictions_dir, tag, before)


def evaluate(args: argparse.Namespace, pred_dir: Path, metadata: Path, output_dir: Path) -> dict[str, Any]:
    run(
        [
            sys.executable,
            EVALUATE_PHYSICS,
            "--inference-dir",
            pred_dir,
            "--metadata",
            metadata,
            "--output-dir",
            output_dir,
        ],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return {}
    return load_summary(output_dir / "metrics.json")


def evaluate_reconstruction(args: argparse.Namespace, pred_dir: Path, raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    run(
        [
            sys.executable,
            EVALUATE_RECONSTRUCTION,
            "--pred-dir",
            pred_dir,
            "--gt-dir",
            raw_dir,
            "--output-dir",
            output_dir,
        ],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return {}
    return load_summary(output_dir / "metrics.json")


def per_frame_rows(metrics_json: Path, sample_name: str, tag: str, pred_dir: Path, transformer: Path) -> list[dict[str, Any]]:
    rows = []
    for level_key in ("scene", "objects"):
        for row in load_json_rows(metrics_json, level_key):
            rows.append(
                {
                    "sample_name": sample_name,
                    "tag": tag,
                    "transformer": str(transformer.expanduser().resolve()),
                    "pred_dir": str(pred_dir),
                    "level": row.get("level"),
                    "object_id": row.get("object_id"),
                    "frame": row.get("frame"),
                    "per_frame_chamfer_distance": row.get("chamfer_distance"),
                    "per_frame_iou": row.get("bbox_iou_3d"),
                    "f_score": row.get("f_score"),
                    "bbox_overlap_volume": row.get("bbox_overlap_volume"),
                    "pred_path": row.get("pred_path"),
                    "gt_path": row.get("gt_path"),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    eval_root = args.eval_root.expanduser().resolve()
    raw_dir = dataset_root / "gt_raw" / args.sample_name
    input_dir = dataset_root / "inference_input" / args.sample_name
    predictions_dir = eval_root / "predictions"
    metrics_dir = eval_root / "metrics" / args.sample_name
    metadata = raw_dir / "physics_metadata.json"

    generate_gt(args, raw_dir)
    prepare_input(args, raw_dir, input_dir)

    if args.run_base_inference:
        base_pred = run_inference(args, args.base_tag, args.base_transformer.expanduser().resolve(), input_dir, predictions_dir)
    elif args.base_pred_dir is not None:
        base_pred = args.base_pred_dir.expanduser().resolve()
        if not base_pred.is_dir():
            raise FileNotFoundError(f"--base-pred-dir does not exist: {base_pred}")
        print(f"Reusing base prediction: {base_pred}")
    else:
        base_pred = newest_existing_tag_dir(predictions_dir, args.base_tag)
        print(f"Reusing newest base prediction: {base_pred}")
    physics_pred = run_inference(args, args.physics_tag, args.physics_transformer.expanduser().resolve(), input_dir, predictions_dir)

    base_physics_summary = evaluate(args, base_pred, metadata, metrics_dir / args.base_tag / "physics")
    physics_physics_summary = evaluate(args, physics_pred, metadata, metrics_dir / args.physics_tag / "physics")
    base_reconstruction_dir = metrics_dir / args.base_tag / "reconstruction"
    physics_reconstruction_dir = metrics_dir / args.physics_tag / "reconstruction"
    base_reconstruction_summary = evaluate_reconstruction(args, base_pred, raw_dir, base_reconstruction_dir)
    physics_reconstruction_summary = evaluate_reconstruction(args, physics_pred, raw_dir, physics_reconstruction_dir)

    if args.dry_run:
        return

    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_name": args.sample_name,
        "raw_dir": str(raw_dir),
        "input_dir": str(input_dir),
        "base_pred": str(base_pred),
        "physics_pred": str(physics_pred),
        "base_transformer": str(args.base_transformer.expanduser().resolve()),
        "physics_transformer": str(args.physics_transformer.expanduser().resolve()),
    }
    row.update({f"base_reconstruction_{key}": value for key, value in base_reconstruction_summary.items()})
    row.update({f"physics_reconstruction_{key}": value for key, value in physics_reconstruction_summary.items()})
    row.update({f"base_physics_{key}": value for key, value in base_physics_summary.items()})
    row.update({f"physics_physics_{key}": value for key, value in physics_physics_summary.items()})
    append_row(eval_root / "two_ball_comparison.csv", row)
    append_rows(
        eval_root / "two_ball_per_frame_metrics.csv",
        per_frame_rows(base_reconstruction_dir / "metrics.json", args.sample_name, args.base_tag, base_pred, args.base_transformer)
        + per_frame_rows(
            physics_reconstruction_dir / "metrics.json",
            args.sample_name,
            args.physics_tag,
            physics_pred,
            args.physics_transformer,
        ),
    )

    print(f"GT sequence: {raw_dir}")
    print(f"Prepared input: {input_dir}")
    print(f"Base prediction: {base_pred}")
    print(f"Physics prediction: {physics_pred}")
    print(f"Metrics: {metrics_dir}")
    print(f"Comparison CSV: {eval_root / 'two_ball_comparison.csv'}")
    print(f"Per-frame metrics CSV: {eval_root / 'two_ball_per_frame_metrics.csv'}")


if __name__ == "__main__":
    main()

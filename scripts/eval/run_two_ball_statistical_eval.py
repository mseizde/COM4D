#!/usr/bin/env python3

"""Run multi-model two-ball evaluation across all prepared comparison samples.

Outputs are intentionally long-form so they work for more than one physics model:
  <eval-root>/two_ball_statistical_summary.csv
  <eval-root>/two_ball_statistical_tests.csv
  <eval-root>/two_ball_per_frame_metrics_full.csv
  <eval-root>/two_ball_model_runs.csv
  <eval-root>/metrics/<sample>/<model>/{physics,reconstruction}/...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
PREPARE_INPUT = SCRIPT_DIR / "prepare_two_ball_inference_input.py"
EVALUATE_PHYSICS = SCRIPT_DIR / "evaluate_physics.py"
EVALUATE_RECONSTRUCTION = SCRIPT_DIR / "evaluate_reconstruction.py"

DEFAULT_DATASET_ROOT = Path("/mnt/mocap_b/work/com4d/datasets/synthetic/two_ball_compare")
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "two_ball_compare"
DEFAULT_BASE_WEIGHTS = "pretrained_weights/TripoSG"
DEFAULT_MODELS = [
    ("base", REPO_ROOT / "pretrained_weights" / "COM4D" / "transformer_ema"),
    ("joint_1500", PROJECT_ROOT / "outputs" / "ckpts" / "physics_st_mix_joint_from_t2s_humoto_1000" / "checkpoints" / "001500"),
    ("mix_s2t_1000", PROJECT_ROOT / "outputs" / "ckpts" / "physics_st_mix_s2t_humoto" / "checkpoints" / "001000"),
    ("mix_t2s_1000", PROJECT_ROOT / "outputs" / "ckpts" / "physics_st_mix_t2s_humoto" / "checkpoints" / "001000"),
]


RECON_SUMMARY_FIELDS = [
    "scene_chamfer_distance_mean",
    "scene_bbox_iou_3d_mean",
    "object_chamfer_distance_mean",
    "object_bbox_iou_3d_mean",
    "scene_per_frame_chamfer_distance_mean",
    "scene_per_frame_iou_mean",
    "object_per_frame_chamfer_distance_mean",
    "object_per_frame_iou_mean",
]
PHYSICS_SUMMARY_FIELDS = [
    "bbox_collision_rate",
    "floor_penetration_mean",
    "floor_penetration_max",
    "scale_error_mean",
    "trajectory_speed_max_mean",
    "trajectory_acceleration_max_mean",
]


def parse_model(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("models must be TAG=TRANSFORMER_PATH")
    tag, path = raw.split("=", 1)
    tag = tag.strip()
    if not tag:
        raise argparse.ArgumentTypeError("model tag cannot be empty")
    return tag, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--sample-glob", default="two_ball_eval_*")
    ap.add_argument("--sample", action="append", default=None, help="Specific sample name. Can be repeated.")
    ap.add_argument("--model", action="append", type=parse_model, default=None, help="TAG=TRANSFORMER_PATH. Can be repeated.")
    ap.add_argument("--base-model-tag", default="base", help="Baseline tag used for paired statistical tests.")
    ap.add_argument("--base-weights-dir", default=DEFAULT_BASE_WEIGHTS)
    ap.add_argument("--num-tokens", type=int, default=1024)
    ap.add_argument("--dynamic-ar-block-size", type=int, default=4)
    ap.add_argument("--dynamic-max-memory-frames", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    ap.add_argument("--input-mode", choices=("symlink", "copy"), default="copy")
    ap.add_argument("--skip-prepare-input", action="store_true")
    ap.add_argument("--reuse-predictions", action="store_true", help="Reuse newest prediction matching <model>_<sample>_* if present.")
    ap.add_argument("--skip-existing-metrics", action="store_true", help="Do not re-evaluate runs that already have reconstruction metrics.")
    ap.add_argument("--only-inference", action="store_true", help="Run/collect predictions only; skip metrics and aggregate CSVs.")
    ap.add_argument("--only-aggregate", action="store_true", help="Skip inference/evaluation and rebuild aggregate CSVs from existing metrics.")
    ap.add_argument("--parallel-workers", type=int, default=1, help="Number of independent model/sample jobs to run concurrently.")
    ap.add_argument("--gpu-ids", default=None, help="Comma-separated GPU ids assigned round-robin to concurrent jobs.")
    ap.add_argument("--force", action="store_true", help="Run inference/evaluation even if outputs exist.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def run(
    cmd: list[object],
    *,
    cwd: Path = REPO_ROOT,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=True, env=env)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return float(statistics.fmean(values)) if values else float("nan")


def stdev(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return float(statistics.stdev(values)) if len(values) > 1 else float("nan")


def median(values: list[float]) -> float:
    values = sorted(value for value in values if not math.isnan(value))
    return float(statistics.median(values)) if values else float("nan")


def maybe_scipy_tests(a: list[float], b: list[float]) -> dict[str, float]:
    pairs = [(x, y) for x, y in zip(a, b) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return {"paired_t_p": float("nan"), "wilcoxon_p": float("nan")}
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return {"paired_t_p": float("nan"), "wilcoxon_p": float("nan")}
    result = {"paired_t_p": float("nan"), "wilcoxon_p": float("nan")}
    try:
        result["paired_t_p"] = float(stats.ttest_rel(xs, ys, nan_policy="omit").pvalue)
    except Exception:
        pass
    try:
        if any(abs(x - y) > 0.0 for x, y in pairs):
            result["wilcoxon_p"] = float(stats.wilcoxon(xs, ys).pvalue)
    except Exception:
        pass
    return result


def discover_samples(dataset_root: Path, sample_glob: str, requested: list[str] | None) -> list[str]:
    if requested:
        names = requested
    else:
        raw_root = dataset_root / "gt_raw"
        names = [path.name for path in sorted(raw_root.glob(sample_glob)) if path.is_dir()]
    valid = []
    for name in names:
        raw_dir = dataset_root / "gt_raw" / name
        if (raw_dir / "physics_metadata.json").is_file():
            valid.append(name)
        else:
            print(f"[warn] skipping {name}: missing {raw_dir / 'physics_metadata.json'}", flush=True)
    if not valid:
        raise RuntimeError(f"No valid samples found under {dataset_root / 'gt_raw'}")
    return valid


def resolve_transformer(path: Path) -> Path:
    path = path.expanduser().resolve()
    nested = path / "transformer_ema"
    if nested.is_dir():
        return nested
    return path


def prepare_input(args: argparse.Namespace, sample: str, raw_dir: Path, input_dir: Path) -> None:
    if args.skip_prepare_input and input_dir.is_dir():
        return
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


def newest_prediction(predictions_dir: Path, tag_prefix: str) -> Path | None:
    candidates = [path for path in predictions_dir.glob(f"{tag_prefix}_*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_inference(
    args: argparse.Namespace,
    model_tag: str,
    transformer: Path,
    sample: str,
    input_dir: Path,
    predictions_dir: Path,
    gpu_id: str | None = None,
) -> Path:
    run_tag = f"{model_tag}_{sample}"
    if args.reuse_predictions and not args.force:
        existing = newest_prediction(predictions_dir, run_tag)
        if existing is not None:
            print(f"Reusing prediction: {existing}", flush=True)
            return existing
    before = set(predictions_dir.glob(f"{run_tag}_*")) if predictions_dir.exists() else set()
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
            run_tag,
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
            len(sorted((input_dir / "frames").glob("*.png"))) if input_dir.is_dir() else 32,
            "--frame_stride",
            1,
            "--scene_num_parts",
            0,
            "--dynamic_num_parts",
            2,
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
        env=env,
    )
    if args.dry_run:
        return predictions_dir / f"{run_tag}_DRY_RUN"
    candidates = [path for path in predictions_dir.glob(f"{run_tag}_*") if path.is_dir() and path not in before]
    if not candidates:
        candidates = [path for path in predictions_dir.glob(f"{run_tag}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No prediction output found for tag {run_tag}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def evaluate_run(args: argparse.Namespace, sample: str, model_tag: str, pred_dir: Path, raw_dir: Path, metrics_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    recon_dir = metrics_dir / sample / model_tag / "reconstruction"
    physics_dir = metrics_dir / sample / model_tag / "physics"
    recon_json = recon_dir / "metrics.json"
    physics_json = physics_dir / "metrics.json"
    if not args.skip_existing_metrics or args.force or not recon_json.is_file():
        run(
            [sys.executable, EVALUATE_RECONSTRUCTION, "--pred-dir", pred_dir, "--gt-dir", raw_dir, "--output-dir", recon_dir],
            dry_run=args.dry_run,
        )
    if not args.skip_existing_metrics or args.force or not physics_json.is_file():
        run(
            [
                sys.executable,
                EVALUATE_PHYSICS,
                "--inference-dir",
                pred_dir,
                "--metadata",
                raw_dir / "physics_metadata.json",
                "--output-dir",
                physics_dir,
            ],
            dry_run=args.dry_run,
        )
    if args.dry_run:
        return {}, {}, []
    recon = read_json(recon_json)
    physics = read_json(physics_json)
    per_frame = []
    for level_key in ("scene", "objects"):
        for row in recon.get(level_key, []):
            per_frame.append(
                {
                    "sample_name": sample,
                    "model_tag": model_tag,
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
    return dict(recon.get("summary", {})), dict(physics.get("summary", {})), per_frame



def existing_metric_row(args: argparse.Namespace, sample: str, model_tag: str, transformer: Path, dataset_root: Path, eval_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    metrics_dir = eval_root / "metrics"
    recon_json = metrics_dir / sample / model_tag / "reconstruction" / "metrics.json"
    physics_json = metrics_dir / sample / model_tag / "physics" / "metrics.json"
    if not recon_json.is_file() or not physics_json.is_file():
        return None
    pred_dir = ""
    recon = read_json(recon_json)
    physics = read_json(physics_json)
    recon_summary = dict(recon.get("summary", {}))
    physics_summary = dict(physics.get("summary", {}))
    pred_dir = str(recon_summary.get("pred_dir", ""))
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_name": sample,
        "model_tag": model_tag,
        "transformer": str(transformer),
        "pred_dir": pred_dir,
        "animation_gif": str(Path(pred_dir) / "animation.gif") if pred_dir else "",
        "raw_dir": str(dataset_root / "gt_raw" / sample),
        "input_dir": str(dataset_root / "inference_input" / sample),
    }
    row.update(recon_summary)
    row.update(physics_summary)
    per_frame = []
    for level_key in ("scene", "objects"):
        for pf in recon.get(level_key, []):
            per_frame.append(
                {
                    "sample_name": sample,
                    "model_tag": model_tag,
                    "transformer": str(transformer),
                    "pred_dir": pred_dir,
                    "level": pf.get("level"),
                    "object_id": pf.get("object_id"),
                    "frame": pf.get("frame"),
                    "per_frame_chamfer_distance": pf.get("chamfer_distance"),
                    "per_frame_iou": pf.get("bbox_iou_3d"),
                    "f_score": pf.get("f_score"),
                    "bbox_overlap_volume": pf.get("bbox_overlap_volume"),
                    "pred_path": pf.get("pred_path"),
                    "gt_path": pf.get("gt_path"),
                }
            )
    return row, per_frame


def run_one_job(
    args: argparse.Namespace,
    sample: str,
    model_tag: str,
    transformer: Path,
    dataset_root: Path,
    eval_root: Path,
    gpu_id: str | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    raw_dir = dataset_root / "gt_raw" / sample
    input_dir = dataset_root / "inference_input" / sample
    pred_dir = run_inference(args, model_tag, transformer, sample, input_dir, eval_root / "predictions", gpu_id=gpu_id)
    if args.only_inference:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sample_name": sample,
            "model_tag": model_tag,
            "transformer": str(transformer),
            "pred_dir": str(pred_dir),
            "animation_gif": str(pred_dir / "animation.gif"),
            "raw_dir": str(raw_dir),
            "input_dir": str(input_dir),
        }
        return row, []
    recon_summary, physics_summary, per_frame = evaluate_run(args, sample, model_tag, pred_dir, raw_dir, eval_root / "metrics")
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_name": sample,
        "model_tag": model_tag,
        "transformer": str(transformer),
        "pred_dir": str(pred_dir),
        "animation_gif": str(pred_dir / "animation.gif"),
        "raw_dir": str(raw_dir),
        "input_dir": str(input_dir),
    }
    row.update(recon_summary)
    row.update(physics_summary)
    for pf in per_frame:
        pf.update({"transformer": str(transformer), "pred_dir": str(pred_dir)})
    return row, per_frame


def model_order(model_rows: list[dict[str, Any]]) -> list[str]:
    models = sorted({row["model_tag"] for row in model_rows})
    if "base" in models:
        models.remove("base")
        models.insert(0, "base")
    return models


def sample_order(model_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({row["sample_name"] for row in model_rows})


def row_metric(rows: list[dict[str, Any]], model: str, metric: str, field: str = "mean") -> float:
    for row in rows:
        if row.get("model_tag") == model and row.get("metric") == metric:
            return to_float(row.get(field))
    return float("nan")


def test_metric(test_rows: list[dict[str, Any]], model: str, metric: str, field: str) -> float:
    for row in test_rows:
        if row.get("model_tag") == model and row.get("metric") == metric:
            return to_float(row.get(field))
    return float("nan")


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "nan"
    if abs(value) < 0.001 and value != 0.0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def winner_counts(model_rows: list[dict[str, Any]], metric: str, direction: str) -> dict[str, int]:
    models = model_order(model_rows)
    counts = {model: 0 for model in models}
    for sample in sample_order(model_rows):
        values = []
        for row in model_rows:
            if row.get("sample_name") == sample:
                value = to_float(row.get(metric))
                if not math.isnan(value):
                    values.append((value, row["model_tag"]))
        if not values:
            continue
        target = min(value for value, _ in values) if direction == "min" else max(value for value, _ in values)
        for value, model in values:
            if abs(value - target) < 1e-12:
                counts[model] += 1
    return counts


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{model}: {count}" for model, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if count > 0)


def best_model(summary_rows: list[dict[str, Any]], models: list[str], metric: str, direction: str) -> str:
    values = [(row_metric(summary_rows, model, metric), model) for model in models]
    values = [(value, model) for value, model in values if not math.isnan(value)]
    if not values:
        return ""
    return (min(values) if direction == "min" else max(values))[1]


def generate_report(
    model_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    base_model_tag: str,
) -> str:
    models = model_order(model_rows)
    samples = sample_order(model_rows)
    reconstruction_rows = []
    for model in models:
        reconstruction_rows.append(
            [
                model,
                fmt(row_metric(summary_rows, model, "scene_chamfer_distance_mean"), 3),
                fmt(row_metric(summary_rows, model, "scene_bbox_iou_3d_mean"), 4),
                fmt(row_metric(summary_rows, model, "object_chamfer_distance_mean"), 3),
                fmt(row_metric(summary_rows, model, "object_bbox_iou_3d_mean"), 5),
            ]
        )

    win_specs = [
        ("Scene Chamfer ↓", "scene_chamfer_distance_mean", "min"),
        ("Scene IoU ↑", "scene_bbox_iou_3d_mean", "max"),
        ("Object Chamfer ↓", "object_chamfer_distance_mean", "min"),
        ("Object IoU ↑", "object_bbox_iou_3d_mean", "max"),
    ]
    win_rows = [[label, format_counts(winner_counts(model_rows, metric, direction))] for label, metric, direction in win_specs]

    physics_rows = []
    for model in models:
        physics_rows.append(
            [
                model,
                fmt(row_metric(summary_rows, model, "bbox_collision_rate"), 4),
                fmt(row_metric(summary_rows, model, "floor_penetration_mean"), 3),
                fmt(row_metric(summary_rows, model, "scale_error_mean"), 3),
                fmt(row_metric(summary_rows, model, "trajectory_acceleration_max_mean"), 1),
            ]
        )

    best_scene_chamfer = best_model(summary_rows, models, "scene_chamfer_distance_mean", "min")
    best_object_chamfer = best_model(summary_rows, models, "object_chamfer_distance_mean", "min")
    best_scene_iou = best_model(summary_rows, models, "scene_bbox_iou_3d_mean", "max")
    best_object_iou = best_model(summary_rows, models, "object_bbox_iou_3d_mean", "max")
    best_collision = best_model(summary_rows, models, "bbox_collision_rate", "min")
    best_scale = best_model(summary_rows, models, "scale_error_mean", "min")
    best_accel = best_model(summary_rows, models, "trajectory_acceleration_max_mean", "min")

    physics_best = max(
        models,
        key=lambda model: sum(
            model == winner
            for winner in (best_collision, best_scale, best_accel)
        ),
    ) if models else ""

    lines = [
        f"Here is the summary from the full {len(samples)}-sample / {len(models)}-model run.",
        "",
        "Metrics are sample-level means; Chamfer lower is better, IoU higher is better.",
        "",
        "## Main Reconstruction Metrics",
        "",
        markdown_table(
            ["Model", "Scene Chamfer ↓", "Scene IoU ↑", "Object Chamfer ↓", "Object IoU ↑"],
            reconstruction_rows,
        ),
        "",
        "## Per-Sample Wins",
        "",
        markdown_table(["Metric", "Winner Count"], win_rows),
        "",
        "Object IoU can have ties, so counts can exceed the number of samples.",
        "",
        "## Physics Metrics",
        "",
        markdown_table(
            ["Model", "Collision Rate ↓", "Floor Pen. Mean ↓", "Scale Error ↓", "Accel. Max Mean ↓"],
            physics_rows,
        ),
        "",
        "## Interpretation",
        "",
    ]

    if len({best_scene_chamfer, best_object_chamfer, best_scene_iou, best_object_iou, physics_best}) > 1:
        lines.append("The statistics do not show a clean overall winner.")
    else:
        lines.append(f"The statistics point to {best_scene_chamfer} as the strongest overall model on these selected metrics.")
    lines.append("")

    if best_object_chamfer:
        t_p = test_metric(test_rows, best_object_chamfer, "object_chamfer_distance_mean", "paired_t_p")
        w_p = test_metric(test_rows, best_object_chamfer, "object_chamfer_distance_mean", "wilcoxon_p")
        if best_object_chamfer == base_model_tag:
            lines.append(f"{base_model_tag} has the best object Chamfer mean among all models.")
        else:
            significance = "significant" if (not math.isnan(t_p) and t_p < 0.05) or (not math.isnan(w_p) and w_p < 0.05) else "not significant"
            lines.append(
                f"{best_object_chamfer} best matches visual/object geometry by object Chamfer. "
                f"Versus {base_model_tag}, this object-Chamfer difference is {significance}: "
                f"paired t-test p={fmt(t_p, 3)}, Wilcoxon p={fmt(w_p, 3)}."
            )
    lines.append("")

    lines.append(f"{best_scene_iou} is strongest on scene IoU, and {best_object_iou} is strongest on object IoU.")
    lines.append("")

    if physics_best:
        lines.append(
            f"{physics_best} is strongest on the selected physics metrics overall: "
            f"best collision model is {best_collision}, best scale-error model is {best_scale}, "
            f"and best acceleration-smoothness model is {best_accel}."
        )
        lines.append("")

    if best_object_chamfer and physics_best and best_object_chamfer != physics_best:
        lines.append(
            f"So: if you care most about visual/object geometry, {best_object_chamfer} is the best candidate by Chamfer. "
            f"If you care most about physical plausibility, {physics_best} is strongest. "
            f"If you care strictly about IoU, compare against {best_scene_iou}/{best_object_iou}."
        )
    elif best_object_chamfer:
        lines.append(f"So: {best_object_chamfer} is the main candidate from this metric subset, but inspect animations before making the final choice.")

    return "\n".join(lines).rstrip() + "\n"

def aggregate(model_rows: list[dict[str, Any]], base_model_tag: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    metrics = [(field, "reconstruction") for field in RECON_SUMMARY_FIELDS] + [(field, "physics") for field in PHYSICS_SUMMARY_FIELDS]
    models = sorted({row["model_tag"] for row in model_rows})
    samples = sorted({row["sample_name"] for row in model_rows})
    row_by_key = {(row["sample_name"], row["model_tag"]): row for row in model_rows}

    for model in models:
        rows = [row for row in model_rows if row["model_tag"] == model]
        for metric, group in metrics:
            values = [to_float(row.get(metric)) for row in rows]
            summary_rows.append(
                {
                    "model_tag": model,
                    "metric_group": group,
                    "metric": metric,
                    "n": len([value for value in values if not math.isnan(value)]),
                    "mean": mean(values),
                    "std": stdev(values),
                    "median": median(values),
                    "min": min((v for v in values if not math.isnan(v)), default=float("nan")),
                    "max": max((v for v in values if not math.isnan(v)), default=float("nan")),
                }
            )

    for model in models:
        if model == base_model_tag:
            continue
        for metric, group in metrics:
            base_vals = []
            model_vals = []
            paired_samples = []
            for sample in samples:
                base_row = row_by_key.get((sample, base_model_tag))
                model_row = row_by_key.get((sample, model))
                if base_row is None or model_row is None:
                    continue
                base_value = to_float(base_row.get(metric))
                model_value = to_float(model_row.get(metric))
                if math.isnan(base_value) or math.isnan(model_value):
                    continue
                base_vals.append(base_value)
                model_vals.append(model_value)
                paired_samples.append(sample)
            deltas = [m - b for m, b in zip(model_vals, base_vals)]
            tests = maybe_scipy_tests(model_vals, base_vals)
            test_rows.append(
                {
                    "model_tag": model,
                    "baseline_tag": base_model_tag,
                    "metric_group": group,
                    "metric": metric,
                    "n_pairs": len(deltas),
                    "model_mean": mean(model_vals),
                    "baseline_mean": mean(base_vals),
                    "mean_delta_model_minus_base": mean(deltas),
                    "median_delta_model_minus_base": median(deltas),
                    "paired_t_p": tests["paired_t_p"],
                    "wilcoxon_p": tests["wilcoxon_p"],
                    "paired_samples": ";".join(paired_samples),
                }
            )
    return summary_rows, test_rows


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    eval_root = args.eval_root.expanduser().resolve()
    predictions_dir = eval_root / "predictions"
    metrics_dir = eval_root / "metrics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    models = args.model or DEFAULT_MODELS
    models = [(tag, resolve_transformer(path)) for tag, path in models]
    samples = discover_samples(dataset_root, args.sample_glob, args.sample)
    print(f"Samples ({len(samples)}): {', '.join(samples)}", flush=True)
    print("Models: " + ", ".join(f"{tag}={path}" for tag, path in models), flush=True)

    model_rows: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    if args.only_aggregate:
        for sample in samples:
            for model_tag, transformer in models:
                existing = existing_metric_row(args, sample, model_tag, transformer, dataset_root, eval_root)
                if existing is None:
                    print(f"[warn] missing existing metrics for {sample}/{model_tag}", flush=True)
                    continue
                row, per_frame = existing
                model_rows.append(row)
                run_rows.append(row)
                per_frame_rows.extend(per_frame)
    else:
        for sample in samples:
            raw_dir = dataset_root / "gt_raw" / sample
            input_dir = dataset_root / "inference_input" / sample
            prepare_input(args, sample, raw_dir, input_dir)

        jobs = []
        gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()] if args.gpu_ids else []
        for sample in samples:
            for model_idx, (model_tag, transformer) in enumerate(models):
                existing = None if args.force else existing_metric_row(args, sample, model_tag, transformer, dataset_root, eval_root)
                if existing is not None and (args.skip_existing_metrics or args.reuse_predictions):
                    row, per_frame = existing
                    model_rows.append(row)
                    run_rows.append(row)
                    per_frame_rows.extend(per_frame)
                    continue
                gpu_id = gpu_ids[len(jobs) % len(gpu_ids)] if gpu_ids else None
                jobs.append((sample, model_tag, transformer, gpu_id))

        if args.parallel_workers <= 1 or len(jobs) <= 1:
            for sample, model_tag, transformer, gpu_id in jobs:
                row, per_frame = run_one_job(args, sample, model_tag, transformer, dataset_root, eval_root, gpu_id)
                if row is not None:
                    model_rows.append(row)
                    run_rows.append(row)
                per_frame_rows.extend(per_frame)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
                futures = [
                    executor.submit(run_one_job, args, sample, model_tag, transformer, dataset_root, eval_root, gpu_id)
                    for sample, model_tag, transformer, gpu_id in jobs
                ]
                for future in as_completed(futures):
                    row, per_frame = future.result()
                    if row is not None:
                        model_rows.append(row)
                        run_rows.append(row)
                    per_frame_rows.extend(per_frame)

    if args.dry_run:
        return
    if args.only_inference:
        write_csv(eval_root / "two_ball_model_runs.csv", run_rows)
        print(f"Wrote run table: {eval_root / 'two_ball_model_runs.csv'}")
        return
    summary_rows, test_rows = aggregate(model_rows, args.base_model_tag)
    write_csv(eval_root / "two_ball_model_runs.csv", run_rows)
    write_csv(eval_root / "two_ball_per_frame_metrics_full.csv", per_frame_rows)
    write_csv(eval_root / "two_ball_statistical_summary.csv", summary_rows)
    write_csv(eval_root / "two_ball_statistical_tests.csv", test_rows)
    report = generate_report(model_rows, summary_rows, test_rows, args.base_model_tag)
    report_path = eval_root / "two_ball_statistical_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote run table: {eval_root / 'two_ball_model_runs.csv'}")
    print(f"Wrote per-frame table: {eval_root / 'two_ball_per_frame_metrics_full.csv'}")
    print(f"Wrote summary table: {eval_root / 'two_ball_statistical_summary.csv'}")
    print(f"Wrote paired tests: {eval_root / 'two_ball_statistical_tests.csv'}")
    print(f"Wrote report: {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()

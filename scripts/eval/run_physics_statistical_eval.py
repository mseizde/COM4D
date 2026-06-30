#!/usr/bin/env python3

"""Run multi-model synthetic physics evaluation across all prepared comparison samples.

Outputs are intentionally long-form so they work for more than one physics model:
  <eval-root>/physics_model_runs.csv
  <eval-root>/physics_per_frame_metrics_full.csv
  <eval-root>/physics_statistical_report.md
  <eval-root>/physics_statistical_summary.csv
  <eval-root>/physics_statistical_tests.csv
  <eval-root>/physics_case_statistical_summary.csv
  <eval-root>/physics_case_statistical_tests.csv
  <eval-root>/case_reports/<case>_statistical_report.md
  <eval-root>/predictions/<sample>/<model>/...
  <eval-root>/metrics/<sample>/<model>/{physics,reconstruction}/...

Quick visual loop:
  python COM4D/scripts/eval/run_physics_statistical_eval.py --quick

This runs representative physics samples, inference only, one worker.

Use --quick --quick-metrics after visual inspection to compute metrics on that
same small subset. Reserve full metrics for branches that pass the GIF/GLB
visual check: identity preservation, actual motion, no object swapping,
plausible collision response, and no collapse/explosion.
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
PREPARE_INPUT = SCRIPT_DIR / "prepare_physics_inference_input.py"
EVALUATE_PHYSICS = SCRIPT_DIR / "evaluate_physics.py"
EVALUATE_RECONSTRUCTION = SCRIPT_DIR / "evaluate_reconstruction.py"
RENDER_PREDICTION_GIFS = SCRIPT_DIR / "render_prediction_gifs.py"

DEFAULT_DATASET_ROOT = Path("/mnt/mocap_b/work/com4d/datasets/synthetic/physics_compare")
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "physics_compare"
DEFAULT_BASE_WEIGHTS = "pretrained_weights/TripoSG"
QUICK_SAMPLES = ["two_ball_eval_000", "two_ball_eval_003", "two_ball_eval_007"]
DEFAULT_MODELS = [
    ("base", REPO_ROOT / "pretrained_weights" / "COM4D" / "transformer_ema"),
    ("joint_1500", PROJECT_ROOT / "outputs" / "ckpts" / "physics_st_mix_joint_from_t2s_humoto_1000" / "checkpoints" / "001500"),
    ("mix_s2t_1000", PROJECT_ROOT / "outputs" / "ckpts" / "physics_st_mix_s2t_humoto" / "checkpoints" / "001000"),
    ("mix_t2s_1000", PROJECT_ROOT / "outputs" / "ckpts" / "physics_st_mix_t2s_humoto" / "checkpoints" / "001000"),
]


RECON_SUMMARY_FIELDS = [
    # Backward-compatible raw metric aliases from evaluate_reconstruction.py.
    "scene_chamfer_distance_mean",
    "scene_f_score_mean",
    "scene_voxel_iou_mean",
    "object_chamfer_distance_mean",
    "object_f_score_mean",
    "object_voxel_iou_mean",
    "scene_per_frame_chamfer_distance_mean",
    "scene_per_frame_iou_mean",
    "object_per_frame_chamfer_distance_mean",
    "object_per_frame_iou_mean",
    # Explicit raw metrics.
    "scene_raw_chamfer_distance_mean",
    "scene_raw_f_score_mean",
    "scene_raw_voxel_iou_mean",
    "object_raw_chamfer_distance_mean",
    "object_raw_f_score_mean",
    "object_raw_voxel_iou_mean",
    "scene_raw_per_frame_chamfer_distance_mean",
    "scene_raw_per_frame_iou_mean",
    "object_raw_per_frame_chamfer_distance_mean",
    "object_raw_per_frame_iou_mean",
    # Sequence-level aligned metrics, populated when --recon-alignment is not none.
    "scene_aligned_chamfer_distance_mean",
    "scene_aligned_f_score_mean",
    "scene_aligned_voxel_iou_mean",
    "object_aligned_chamfer_distance_mean",
    "object_aligned_f_score_mean",
    "object_aligned_voxel_iou_mean",
    "scene_aligned_per_frame_chamfer_distance_mean",
    "scene_aligned_per_frame_iou_mean",
    "object_aligned_per_frame_chamfer_distance_mean",
    "object_aligned_per_frame_iou_mean",
    # Diagnostic bounding-box metrics are kept in CSV/JSON, but they are not the headline IoU.
    "scene_bbox_iou_3d_mean",
    "object_bbox_iou_3d_mean",
    "scene_raw_bbox_iou_3d_mean",
    "object_raw_bbox_iou_3d_mean",
    "scene_aligned_bbox_iou_3d_mean",
    "object_aligned_bbox_iou_3d_mean",
]
BASIC_PHYSICS_SUMMARY_FIELDS = [
    "bbox_collision_rate",
    "floor_penetration_mean",
    "floor_penetration_max",
    "scale_error_mean",
    "trajectory_speed_max_mean",
    "trajectory_acceleration_max_mean",
]
TRAJECTORY_PHYSICS_SUMMARY_FIELDS = [
    "trajectory_ate_rmse_mean",
    "trajectory_ate_mean_mean",
    "trajectory_ate_median_mean",
    "trajectory_ate_max_mean",
    "trajectory_discrete_frechet_aligned_mean",
    "trajectory_frame_indices_match_rate",
]
PHYSICS_SUMMARY_FIELDS = BASIC_PHYSICS_SUMMARY_FIELDS + TRAJECTORY_PHYSICS_SUMMARY_FIELDS
INCLUDE_BASIC_PHYSICS_SUMMARY = True


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
    ap.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional RGB frame directory for one selected sample; masks and GT still come from --dataset-root.",
    )
    ap.add_argument("--sample-glob", default="*_eval_*")
    ap.add_argument("--sample", action="append", default=None, help="Specific sample name. Can be repeated.")
    ap.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use the visual-inspection loop: representative samples "
            f"{', '.join(QUICK_SAMPLES)}, inference only, and one worker."
        ),
    )
    ap.add_argument(
        "--quick-metrics",
        action="store_true",
        help="With --quick, also run reconstruction/physics metrics on the quick sample subset.",
    )
    ap.add_argument(
        "--model",
        action="append",
        type=parse_model,
        default=None,
        help=(
            "TAG=TRANSFORMER_PATH. By default this augments/replaces the discovered prediction-model intersection; "
            "combine with --no-discover-prediction-models for explicit-only evaluation. Can be repeated."
        ),
    )
    ap.add_argument(
        "--extra-model",
        action="append",
        type=parse_model,
        default=None,
        help=(
            "Additional TAG=TRANSFORMER_PATH to evaluate after the models discovered from existing predictions. "
            "Can be repeated."
        ),
    )
    ap.add_argument(
        "--no-discover-prediction-models",
        action="store_true",
        help=(
            "Do not auto-discover the intersection of model tags present under "
            "<eval-root>/predictions/<sample>/. With this flag, only --model/--extra-model or DEFAULT_MODELS are used."
        ),
    )
    ap.add_argument("--base-model-tag", default="base", help="Baseline tag used for paired statistical tests.")
    ap.add_argument("--base-weights-dir", default=DEFAULT_BASE_WEIGHTS)
    ap.add_argument("--num-tokens", type=int, default=1024)
    ap.add_argument("--dynamic-ar-block-size", type=int, default=4)
    ap.add_argument("--dynamic-max-memory-frames", type=int, default=8)
    ap.add_argument(
        "--dynamic-mix-cutoff",
        "--dynamic_mix_cutoff",
        dest="dynamic_mix_cutoff",
        type=int,
        default=None,
        help="Forwarded to inference_com4d.py --dynamic_mix_cutoff; e.g. 49 keeps dynamic mixing active for 50 denoising steps.",
    )
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    ap.add_argument(
        "--render-animations",
        action="store_true",
        help="Render animation.gif during inference. Metrics only need exported GLBs, so this is off by default.",
    )
    ap.add_argument(
        "--render-diagnostic-gifs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After reconstruction metrics are available, render only animation_fixed_regenerated.gif, "
            "animation_diagnostic.gif, and animation_gt_overlay.gif for each run."
        ),
    )
    ap.add_argument(
        "--force-diagnostic-gifs",
        action="store_true",
        help="Overwrite existing statistical-evaluation diagnostic GIFs.",
    )
    ap.add_argument("--diagnostic-gif-render-size", type=int, default=512, help="Square render size for statistical-evaluation GIFs.")
    ap.add_argument("--diagnostic-gif-fps", type=int, default=0, help="GIF FPS; 0 uses the GT metadata FPS.")
    ap.add_argument("--input-mode", choices=("symlink", "copy"), default="copy")
    ap.add_argument("--skip-prepare-input", action="store_true")
    ap.add_argument("--reuse-predictions", action="store_true", help="Reuse newest prediction matching <model>_<sample>_* if present.")
    ap.add_argument("--skip-existing-metrics", action="store_true", help="Do not re-evaluate runs that already have reconstruction metrics.")
    ap.add_argument(
        "--force-metrics",
        action="store_true",
        help="Recompute metrics even when metrics already exist; useful with --reuse-predictions.",
    )
    ap.add_argument(
        "--recon-alignment",
        choices=("none", "translation", "similarity", "first_frame_similarity"),
        default="first_frame_similarity",
        help="Prediction-to-GT alignment for reconstruction metrics. first_frame_similarity matches the COM4D paper protocol.",
    )
    ap.add_argument(
        "--recon-object-assignment",
        choices=("fixed", "best"),
        default="best",
        help="Object matching mode for reconstruction metrics.",
    )
    ap.add_argument(
        "--recon-iou-num-grids",
        "--iou-num-grids",
        dest="recon_iou_num_grids",
        type=int,
        default=32,
        help="Voxel IoU grid resolution forwarded to reconstruction evaluation.",
    )
    ap.add_argument(
        "--recon-skip-object-level",
        "--skip-object-level",
        dest="recon_skip_object_level",
        action="store_true",
        help="Forward --skip-object-level to reconstruction evaluation.",
    )
    ap.add_argument(
        "--recon-skip-voxel-iou",
        "--skip-voxel-iou",
        dest="recon_skip_voxel_iou",
        action="store_true",
        help="Forward --skip-voxel-iou to reconstruction evaluation.",
    )
    ap.add_argument(
        "--recon-skip-raw-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute only aligned reconstruction metrics when alignment is enabled (default: true).",
    )
    ap.add_argument(
        "--reconstruction-timeout-seconds",
        type=float,
        default=1800.0,
        help="Terminate a reconstruction evaluator after this many seconds; 0 disables the timeout.",
    )
    ap.add_argument(
        "--max-recon-glb-mb",
        type=float,
        default=0.0,
        help=(
            "Skip reconstruction metrics when any predicted GLB is larger than this many MiB. "
            "Use 0 to disable the guard. Physics metrics still run."
        ),
    )
    ap.add_argument(
        "--max-recon-total-glb-gb",
        type=float,
        default=0.0,
        help=(
            "Skip reconstruction metrics when all predicted GLBs for a run exceed this total GiB. "
            "Use 0 to disable the guard. Physics metrics still run."
        ),
    )
    ap.add_argument(
        "--physics-com-method",
        choices=("vertex_centroid", "volume_center_mass", "bbox_center"),
        default="vertex_centroid",
        help="COM method forwarded to evaluate_physics.py for trajectory ATE/Frechet metrics.",
    )
    ap.add_argument(
        "--skip-basic-physics-summary",
        action="store_true",
        help="Forward --skip-basic-physics-summary to evaluate_physics.py.",
    )
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
    timeout: float | None = None,
) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=True, env=env, timeout=timeout)


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


def prediction_has_reconstruction_outputs(path: Path) -> bool:
    return any((path / "dynamic").glob("dynamic_scene_frame_*.glb")) or any(
        (path / "dynamic").glob("object_*/frame_*.glb")
    )


def prediction_parent(predictions_root: Path, sample: str, model_tag: str) -> Path:
    return predictions_root / sample / model_tag


def newest_prediction(predictions_root: Path, sample: str, model_tag: str) -> Path | None:
    stable = prediction_parent(predictions_root, sample, model_tag)
    candidates = [stable] if stable.is_dir() else []

    # Backward compatibility for previous nested layouts:
    #   predictions/<sample>/<model>/<model>
    #   predictions/<sample>/<model>/<model>_<timestamp>
    candidates.extend(path for path in stable.glob(model_tag) if path.is_dir())
    candidates.extend(path for path in stable.glob(f"{model_tag}_*") if path.is_dir())

    # Backward compatibility for the old flat layout:
    #   predictions/<model>_<sample>_<timestamp>/...
    legacy_prefix = f"{model_tag}_{sample}"
    candidates.extend(path for path in predictions_root.glob(f"{legacy_prefix}_*") if path.is_dir())

    complete = [path for path in candidates if prediction_has_reconstruction_outputs(path)]
    skipped = len(candidates) - len(complete)
    if skipped:
        print(f"[warn] ignoring {skipped} incomplete reused prediction(s) for {sample}/{model_tag}", flush=True)
    if not complete:
        return None
    return max(complete, key=lambda path: path.stat().st_mtime)


def discover_prediction_model_tags(predictions_root: Path, samples: list[str]) -> list[str]:
    """Return model tags with complete predictions for every selected sample."""
    sample_sets: list[set[str]] = []
    for sample in samples:
        sample_dir = predictions_root / sample
        if not sample_dir.is_dir():
            print(f"[warn] no prediction directory for {sample}: {sample_dir}", flush=True)
            sample_sets.append(set())
            continue
        tags = {path.name for path in sample_dir.iterdir() if path.is_dir()}
        complete_tags = {tag for tag in tags if newest_prediction(predictions_root, sample, tag) is not None}
        sample_sets.append(complete_tags)
    if not sample_sets:
        return []
    common = set.intersection(*sample_sets)
    return sorted(common)


def model_path_for_log(transformer: Path | None) -> str:
    return str(transformer) if transformer is not None else ""


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


def run_inference(
    args: argparse.Namespace,
    model_tag: str,
    transformer: Path | None,
    sample: str,
    input_dir: Path,
    predictions_dir: Path,
    gpu_id: str | None = None,
) -> Path:
    run_tag = model_tag
    output_root = predictions_dir / sample
    output_root.mkdir(parents=True, exist_ok=True)
    if args.reuse_predictions and not args.force:
        existing = newest_prediction(predictions_dir, sample, model_tag)
        if existing is not None:
            print(f"Reusing prediction: {existing}", flush=True)
            return existing
    if transformer is None:
        raise RuntimeError(
            f"No transformer path is known for discovered model {model_tag!r} on {sample}; "
            "rerun with --reuse-predictions or provide it via --extra-model/--model."
        )
    stable_output = prediction_parent(predictions_dir, sample, model_tag)
    before = set(output_root.glob(f"{run_tag}_*")) if output_root.exists() else set()
    env = os.environ.copy()
    if gpu_id is not None:
        parent_visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        gpu_id_str = str(gpu_id)
        if parent_visible:
            try:
                gpu_index = int(gpu_id_str)
            except ValueError:
                gpu_index = -1
            if 0 <= gpu_index < len(parent_visible):
                gpu_id_str = parent_visible[gpu_index]
        env["CUDA_VISIBLE_DEVICES"] = gpu_id_str
    frames_dir = args.frames_dir if args.frames_dir is not None else input_dir / "frames"
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")
    if args.frames_dir is not None:
        print(f"Using frame override for {sample}: {frames_dir}", flush=True)
    cmd = [
        sys.executable,
        "src/inference_com4d.py",
        "--frames_dir",
        frames_dir,
        "--masks_dir",
        input_dir / "masks",
        "--masks_static_dir",
        input_dir / "masks_static",
        "--output_dir",
        output_root,
        "--tag",
        run_tag,
        "--no-timestamp-output",
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
        len(frame_paths),
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
        "--image_size",
        args.image_size,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--no-render_predicted_room",
        "--no-room_augment_animations",
    ]
    if args.dynamic_mix_cutoff is not None:
        cmd.extend(["--dynamic_mix_cutoff", args.dynamic_mix_cutoff])
    if args.render_animations:
        cmd.append("--animation")
    run(
        cmd,
        dry_run=args.dry_run,
        env=env,
    )
    if args.dry_run:
        return stable_output
    if prediction_has_reconstruction_outputs(stable_output):
        return stable_output
    candidates = [path for path in output_root.glob(f"{run_tag}_*") if path.is_dir() and path not in before]
    if not candidates:
        candidates = [path for path in output_root.glob(f"{run_tag}_*") if path.is_dir()]
    complete = [path for path in candidates if prediction_has_reconstruction_outputs(path)]
    if not complete:
        raise FileNotFoundError(f"No complete prediction output found for tag {run_tag}")
    return max(complete, key=lambda path: path.stat().st_mtime)


def prediction_glb_stats(pred_dir: Path) -> dict[str, Any]:
    paths = sorted((pred_dir / "dynamic").glob("dynamic_scene_frame_*.glb"))
    paths.extend(sorted((pred_dir / "dynamic").glob("object_*/frame_*.glb")))
    sizes = [path.stat().st_size for path in paths if path.is_file()]
    largest_idx = max(range(len(paths)), key=lambda idx: paths[idx].stat().st_size) if paths else None
    largest_path = str(paths[largest_idx]) if largest_idx is not None else ""
    largest_size = sizes[largest_idx] if largest_idx is not None else 0
    return {
        "num_pred_glbs": len(sizes),
        "max_pred_glb_bytes": largest_size,
        "max_pred_glb_mib": largest_size / (1024 * 1024),
        "max_pred_glb_path": largest_path,
        "total_pred_glb_bytes": sum(sizes),
        "total_pred_glb_gib": sum(sizes) / (1024 * 1024 * 1024),
    }


def reconstruction_skip_reason(args: argparse.Namespace, stats: dict[str, Any]) -> str | None:
    max_mib = float(getattr(args, "max_recon_glb_mb", 0.0) or 0.0)
    if max_mib > 0.0 and float(stats.get("max_pred_glb_mib", 0.0)) > max_mib:
        return (
            f"max predicted GLB size {stats['max_pred_glb_mib']:.2f} MiB exceeds "
            f"--max-recon-glb-mb {max_mib:.2f}"
        )
    max_total_gib = float(getattr(args, "max_recon_total_glb_gb", 0.0) or 0.0)
    if max_total_gib > 0.0 and float(stats.get("total_pred_glb_gib", 0.0)) > max_total_gib:
        return (
            f"total predicted GLB size {stats['total_pred_glb_gib']:.2f} GiB exceeds "
            f"--max-recon-total-glb-gb {max_total_gib:.2f}"
        )
    return None


def write_skipped_reconstruction_metrics(
    recon_dir: Path,
    pred_dir: Path,
    raw_dir: Path,
    args: argparse.Namespace,
    stats: dict[str, Any],
    reason: str,
) -> None:
    summary: dict[str, Any] = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(raw_dir),
        "num_scene_pairs": 0,
        "num_object_pairs": 0,
        "num_samples": None,
        "threshold": None,
        "iou_num_grids": None,
        "iou_scale": None,
        "alignment": args.recon_alignment,
        "object_assignment": args.recon_object_assignment,
        "alignment_num_points": 0,
        "pred_to_gt_transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "object_assignment_map": {},
        "reconstruction_skipped": True,
        "reconstruction_skip_reason": reason,
    }
    summary.update(stats)
    for field in RECON_SUMMARY_FIELDS:
        summary.setdefault(field, float("nan"))
    recon_dir.mkdir(parents=True, exist_ok=True)
    with (recon_dir / "metrics.json").open("w") as f:
        json.dump({"summary": summary, "scene": [], "objects": []}, f, indent=2, allow_nan=True)
    (recon_dir / "scene_metrics.csv").write_text("")
    (recon_dir / "object_metrics.csv").write_text("")
    write_csv(recon_dir / "summary.csv", [summary])
    print(f"[warn] skipped reconstruction metrics for {pred_dir}: {reason}", flush=True)


def physics_metrics_stale(physics_json: Path) -> bool:
    if not physics_json.is_file():
        return True
    try:
        summary = dict(read_json(physics_json).get("summary", {}))
    except Exception:
        return True
    required = (
        "trajectory_ate_rmse_mean",
        "trajectory_ate_mean_mean",
        "trajectory_ate_max_mean",
        "trajectory_discrete_frechet_aligned_mean",
        "trajectory_frame_indices_match_rate",
    )
    return any(field not in summary for field in required)


def evaluate_run(args: argparse.Namespace, sample: str, model_tag: str, pred_dir: Path, raw_dir: Path, metrics_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    recon_dir = metrics_dir / sample / model_tag / "reconstruction"
    physics_dir = metrics_dir / sample / model_tag / "physics"
    recon_json = recon_dir / "metrics.json"
    physics_json = physics_dir / "metrics.json"
    if args.force_metrics or not args.skip_existing_metrics or args.force or not recon_json.is_file():
        glb_stats = prediction_glb_stats(pred_dir)
        skip_reason = reconstruction_skip_reason(args, glb_stats)
        if skip_reason is not None:
            if args.dry_run:
                print(f"[dry-run] would skip reconstruction metrics for {pred_dir}: {skip_reason}", flush=True)
            else:
                write_skipped_reconstruction_metrics(recon_dir, pred_dir, raw_dir, args, glb_stats, skip_reason)
        else:
            reconstruction_cmd = [
                sys.executable,
                EVALUATE_RECONSTRUCTION,
                "--pred-dir",
                pred_dir,
                "--gt-dir",
                raw_dir,
                "--output-dir",
                recon_dir,
                "--alignment",
                args.recon_alignment,
                "--object-assignment",
                args.recon_object_assignment,
                "--iou-num-grids",
                args.recon_iou_num_grids,
                *(["--skip-object-level"] if args.recon_skip_object_level else []),
                *(["--skip-voxel-iou"] if args.recon_skip_voxel_iou else []),
                *(["--skip-raw-metrics"] if args.recon_skip_raw_metrics else []),
            ]
            timeout = args.reconstruction_timeout_seconds if args.reconstruction_timeout_seconds > 0.0 else None
            try:
                run(reconstruction_cmd, dry_run=args.dry_run, timeout=timeout)
            except subprocess.TimeoutExpired:
                reason = f"reconstruction evaluator exceeded {args.reconstruction_timeout_seconds:.1f} seconds"
                write_skipped_reconstruction_metrics(recon_dir, pred_dir, raw_dir, args, glb_stats, reason)
    if args.force_metrics or not args.skip_existing_metrics or args.force or physics_metrics_stale(physics_json):
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
                "--com-method",
                args.physics_com_method,
                *( ["--skip-basic-physics-summary"] if args.skip_basic_physics_summary else [] ),
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
                    "gt_object_id": row.get("gt_object_id"),
                    "per_frame_chamfer_distance": row.get("chamfer_distance"),
                    "per_frame_iou": row.get("voxel_iou"),
                    "per_frame_bbox_iou_3d": row.get("bbox_iou_3d"),
                    "f_score": row.get("f_score"),
                    "bbox_overlap_volume": row.get("bbox_overlap_volume"),
                    "raw_per_frame_chamfer_distance": row.get("raw_chamfer_distance"),
                    "raw_per_frame_iou": row.get("raw_voxel_iou"),
                    "raw_per_frame_bbox_iou_3d": row.get("raw_bbox_iou_3d"),
                    "raw_f_score": row.get("raw_f_score"),
                    "aligned_per_frame_chamfer_distance": row.get("aligned_chamfer_distance"),
                    "aligned_per_frame_iou": row.get("aligned_voxel_iou"),
                    "aligned_per_frame_bbox_iou_3d": row.get("aligned_bbox_iou_3d"),
                    "aligned_f_score": row.get("aligned_f_score"),
                    "pred_path": row.get("pred_path"),
                    "gt_path": row.get("gt_path"),
                }
            )
    return dict(recon.get("summary", {})), dict(physics.get("summary", {})), per_frame


def render_statistical_diagnostic_gifs(
    args: argparse.Namespace,
    sample: str,
    model_tag: str,
    pred_dir: Path,
    raw_dir: Path,
    input_dir: Path,
    metrics_dir: Path,
) -> None:
    if not args.render_diagnostic_gifs:
        return
    recon_json = metrics_dir / sample / model_tag / "reconstruction" / "metrics.json"
    metadata = raw_dir / "physics_metadata.json"
    if not recon_json.is_file():
        print(f"[warn] skipping diagnostic GIFs for {sample}/{model_tag}: missing {recon_json}", flush=True)
        return
    if not metadata.is_file():
        print(f"[warn] skipping diagnostic GIFs for {sample}/{model_tag}: missing {metadata}", flush=True)
        return
    frames_dir = args.frames_dir if args.frames_dir is not None else input_dir / "frames"
    cmd = [
        sys.executable,
        RENDER_PREDICTION_GIFS,
        "--export-dir",
        pred_dir,
        "--camera-metadata",
        metadata,
        "--alignment-metadata",
        recon_json,
        "--gt-geometry-root",
        raw_dir,
        "--source-frames-dir",
        frames_dir,
        "--output-name",
        "animation_fixed_regenerated.gif",
        "--diagnostic-output-name",
        "animation_diagnostic.gif",
        "--gt-overlay-output-name",
        "animation_gt_overlay.gif",
        "--no-default-orbit-artifact",
        "--render-size",
        args.diagnostic_gif_render_size,
    ]
    if args.diagnostic_gif_fps and args.diagnostic_gif_fps > 0:
        cmd.extend(["--fps", args.diagnostic_gif_fps])
    if args.force_diagnostic_gifs or args.force:
        cmd.append("--overwrite")
    run(cmd, dry_run=args.dry_run)


def existing_metric_row(args: argparse.Namespace, sample: str, model_tag: str, transformer: Path | None, dataset_root: Path, eval_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    metrics_dir = eval_root / "metrics"
    recon_json = metrics_dir / sample / model_tag / "reconstruction" / "metrics.json"
    physics_json = metrics_dir / sample / model_tag / "physics" / "metrics.json"
    if not recon_json.is_file() or physics_metrics_stale(physics_json):
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
        "transformer": model_path_for_log(transformer),
        "pred_dir": pred_dir,
        "animation_gif": str(Path(pred_dir) / "animation.gif") if pred_dir else "",
        "animation_fixed_regenerated_gif": str(Path(pred_dir) / "animation_fixed_regenerated.gif") if pred_dir else "",
        "animation_diagnostic_gif": str(Path(pred_dir) / "animation_diagnostic.gif") if pred_dir else "",
        "animation_gt_overlay_gif": str(Path(pred_dir) / "animation_gt_overlay.gif") if pred_dir else "",
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
                    "transformer": model_path_for_log(transformer),
                    "pred_dir": pred_dir,
                    "level": pf.get("level"),
                    "object_id": pf.get("object_id"),
                    "frame": pf.get("frame"),
                    "gt_object_id": pf.get("gt_object_id"),
                    "per_frame_chamfer_distance": pf.get("chamfer_distance"),
                    "per_frame_iou": pf.get("voxel_iou"),
                    "per_frame_bbox_iou_3d": pf.get("bbox_iou_3d"),
                    "f_score": pf.get("f_score"),
                    "bbox_overlap_volume": pf.get("bbox_overlap_volume"),
                    "raw_per_frame_chamfer_distance": pf.get("raw_chamfer_distance"),
                    "raw_per_frame_iou": pf.get("raw_voxel_iou"),
                    "raw_per_frame_bbox_iou_3d": pf.get("raw_bbox_iou_3d"),
                    "raw_f_score": pf.get("raw_f_score"),
                    "aligned_per_frame_chamfer_distance": pf.get("aligned_chamfer_distance"),
                    "aligned_per_frame_iou": pf.get("aligned_voxel_iou"),
                    "aligned_per_frame_bbox_iou_3d": pf.get("aligned_bbox_iou_3d"),
                    "aligned_f_score": pf.get("aligned_f_score"),
                    "pred_path": pf.get("pred_path"),
                    "gt_path": pf.get("gt_path"),
                }
            )
    return row, per_frame


def run_one_job(
    args: argparse.Namespace,
    sample: str,
    model_tag: str,
    transformer: Path | None,
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
            "transformer": model_path_for_log(transformer),
            "pred_dir": str(pred_dir),
            "animation_gif": str(pred_dir / "animation.gif"),
            "raw_dir": str(raw_dir),
            "input_dir": str(input_dir),
        }
        return row, []
    metrics_dir = eval_root / "metrics"
    recon_summary, physics_summary, per_frame = evaluate_run(args, sample, model_tag, pred_dir, raw_dir, metrics_dir)
    render_statistical_diagnostic_gifs(args, sample, model_tag, pred_dir, raw_dir, input_dir, metrics_dir)
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample_name": sample,
        "model_tag": model_tag,
        "transformer": model_path_for_log(transformer),
        "pred_dir": str(pred_dir),
        "animation_gif": str(pred_dir / "animation.gif"),
        "animation_fixed_regenerated_gif": str(pred_dir / "animation_fixed_regenerated.gif"),
        "animation_diagnostic_gif": str(pred_dir / "animation_diagnostic.gif"),
        "animation_gt_overlay_gif": str(pred_dir / "animation_gt_overlay.gif"),
        "raw_dir": str(raw_dir),
        "input_dir": str(input_dir),
    }
    row.update(recon_summary)
    row.update(physics_summary)
    for pf in per_frame:
        pf.update({"transformer": model_path_for_log(transformer), "pred_dir": str(pred_dir)})
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


CASE_NAMES = ("ball_drop", "rolling_occluder", "two_ball", "wall_impact")


def case_name_from_sample(sample: str) -> str:
    for case in CASE_NAMES:
        if sample.startswith(f"{case}_eval_"):
            return case
    return sample.rsplit("_eval_", 1)[0] if "_eval_" in sample else sample


def add_case_names(model_rows: list[dict[str, Any]]) -> None:
    for row in model_rows:
        row.setdefault("case_name", case_name_from_sample(str(row.get("sample_name", ""))))


def selected_physics_summary_fields() -> list[str]:
    if INCLUDE_BASIC_PHYSICS_SUMMARY:
        return PHYSICS_SUMMARY_FIELDS
    return TRAJECTORY_PHYSICS_SUMMARY_FIELDS


def metric_specs() -> list[tuple[str, str]]:
    return [(field, "reconstruction") for field in RECON_SUMMARY_FIELDS] + [
        (field, "physics") for field in selected_physics_summary_fields()
    ]


def physics_metric_table_spec() -> list[tuple[str, str, int]]:
    basic = [
        ("Collision Rate ↓", "bbox_collision_rate", 4),
        ("Floor Pen. Mean ↓", "floor_penetration_mean", 3),
        ("Floor Pen. Max ↓", "floor_penetration_max", 3),
        ("Scale Error ↓", "scale_error_mean", 3),
        ("Speed Max Mean ↓", "trajectory_speed_max_mean", 1),
        ("Accel. Max Mean ↓", "trajectory_acceleration_max_mean", 1),
    ]
    trajectory = [
        ("ATE RMSE ↓", "trajectory_ate_rmse_mean", 4),
        ("ATE Mean ↓", "trajectory_ate_mean_mean", 4),
        ("ATE Max ↓", "trajectory_ate_max_mean", 4),
        ("Fréchet ↓", "trajectory_discrete_frechet_aligned_mean", 4),
        ("Frame Match ↑", "trajectory_frame_indices_match_rate", 3),
    ]
    return (basic if INCLUDE_BASIC_PHYSICS_SUMMARY else []) + trajectory


def physics_metric_rows(summary_rows: list[dict[str, Any]], models: list[str]) -> list[list[str]]:
    spec = physics_metric_table_spec()
    return [[model] + [fmt(row_metric(summary_rows, model, metric), precision) for _label, metric, precision in spec] for model in models]


def physics_best_models(summary_rows: list[dict[str, Any]], models: list[str]) -> dict[str, str]:
    winners = {
        "ate": best_model(summary_rows, models, "trajectory_ate_rmse_mean", "min"),
        "frechet": best_model(summary_rows, models, "trajectory_discrete_frechet_aligned_mean", "min"),
    }
    if INCLUDE_BASIC_PHYSICS_SUMMARY:
        winners.update(
            {
                "collision": best_model(summary_rows, models, "bbox_collision_rate", "min"),
                "floor_penetration": best_model(summary_rows, models, "floor_penetration_mean", "min"),
                "scale": best_model(summary_rows, models, "scale_error_mean", "min"),
                "speed": best_model(summary_rows, models, "trajectory_speed_max_mean", "min"),
                "acceleration": best_model(summary_rows, models, "trajectory_acceleration_max_mean", "min"),
            }
        )
    return winners


def strongest_physics_model(summary_rows: list[dict[str, Any]], models: list[str]) -> str:
    winners = physics_best_models(summary_rows, models)
    return max(
        models,
        key=lambda model: sum(model == winner for winner in winners.values()),
    ) if models else ""


def generate_case_group_report(
    case: str,
    case_rows: list[dict[str, Any]],
    base_model_tag: str,
) -> str:
    summary_rows, test_rows = aggregate(case_rows, base_model_tag)
    report = generate_report(case_rows, summary_rows, test_rows, base_model_tag)
    lines = [
        f"# {case} Statistical Report",
        "",
        f"Case-level report for {len(sample_order(case_rows))} samples. Within this case, samples are averaged normally.",
        "",
        report,
    ]
    return "\n".join(lines).rstrip() + "\n"


def generate_physics_report(
    model_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    base_model_tag: str,
) -> str:
    models = model_order(model_rows)
    samples = sample_order(model_rows)
    best_physics = physics_best_models(summary_rows, models)
    physics_best = strongest_physics_model(summary_rows, models)

    win_specs = [
        (label, metric, "max" if metric == "trajectory_frame_indices_match_rate" else "min")
        for label, metric, _precision in physics_metric_table_spec()
    ]
    win_rows = [[label, format_counts(winner_counts(model_rows, metric, direction))] for label, metric, direction in win_specs]

    pair_rows = []
    for model in models:
        if model == base_model_tag:
            continue
        for label, metric, _precision in physics_metric_table_spec():
            pair_rows.append(
                [
                    model,
                    label,
                    fmt(test_metric(test_rows, model, metric, "mean_delta_model_minus_base"), 4),
                    fmt(test_metric(test_rows, model, metric, "paired_t_p"), 3),
                    fmt(test_metric(test_rows, model, metric, "wilcoxon_p"), 3),
                ]
            )

    lines = [
        "# Aggregate Physics Statistical Report",
        "",
        f"Physics-only summary from {len(samples)} samples and {len(models)} models.",
        "",
        "Lower is better for every metric in this report.",
        "",
        "## Physics Metrics",
        "",
        markdown_table(
            [
                "Model",
                "Collision Rate ↓",
                "Floor Pen. Mean ↓",
                "Floor Pen. Max ↓",
                "Scale Error ↓",
                "Speed Max Mean ↓",
                "Accel. Max Mean ↓",
                "ATE RMSE ↓",
                "ATE Mean ↓",
                "ATE Max ↓",
                "Fréchet ↓",
                "Frame Match ↑",
            ],
            physics_metric_rows(summary_rows, models),
        ),
        "",
        "## Per-Sample Physics Wins",
        "",
        markdown_table(["Metric", "Winner Count"], win_rows),
        "",
        "## Paired Tests vs Baseline",
        "",
        markdown_table(["Model", "Metric", "Mean Delta", "Paired t p", "Wilcoxon p"], pair_rows),
        "",
        "## Interpretation",
        "",
    ]
    if physics_best:
        if INCLUDE_BASIC_PHYSICS_SUMMARY:
            lines.append(
                f"{physics_best} is strongest by majority across selected physics metrics. "
                f"Best collision={best_physics.get('collision') or 'n/a'}, "
                f"floor penetration={best_physics.get('floor_penetration') or 'n/a'}, "
                f"scale={best_physics.get('scale') or 'n/a'}, speed={best_physics.get('speed') or 'n/a'}, "
                f"acceleration={best_physics.get('acceleration') or 'n/a'}, ATE={best_physics.get('ate') or 'n/a'}, "
                f"Fréchet={best_physics.get('frechet') or 'n/a'}."
            )
        else:
            lines.append(
                f"{physics_best} is strongest by majority across selected trajectory metrics. "
                f"Best ATE={best_physics.get('ate') or 'n/a'}, "
                f"Fréchet={best_physics.get('frechet') or 'n/a'}."
            )
        lines.append("")
    lines.append("Use this alongside the per-case reports to check whether aggregate improvements are broad or dominated by a small number of cases.")
    return "\n".join(lines).rstrip() + "\n"


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
                fmt(row_metric(summary_rows, model, "scene_aligned_chamfer_distance_mean"), 3),
                fmt(row_metric(summary_rows, model, "scene_aligned_voxel_iou_mean"), 4),
                fmt(row_metric(summary_rows, model, "object_aligned_chamfer_distance_mean"), 3),
                fmt(row_metric(summary_rows, model, "object_aligned_f_score_mean"), 4),
                fmt(row_metric(summary_rows, model, "object_aligned_voxel_iou_mean"), 5),
            ]
        )

    win_specs = [
        ("Aligned Scene Chamfer ↓", "scene_aligned_chamfer_distance_mean", "min"),
        ("Aligned Scene Voxel IoU ↑", "scene_aligned_voxel_iou_mean", "max"),
        ("Aligned Object Chamfer ↓", "object_aligned_chamfer_distance_mean", "min"),
        ("Aligned Object Voxel IoU ↑", "object_aligned_voxel_iou_mean", "max"),
    ]
    win_rows = [[label, format_counts(winner_counts(model_rows, metric, direction))] for label, metric, direction in win_specs]

    physics_rows = physics_metric_rows(summary_rows, models)

    best_scene_chamfer = best_model(summary_rows, models, "scene_aligned_chamfer_distance_mean", "min")
    best_object_chamfer = best_model(summary_rows, models, "object_aligned_chamfer_distance_mean", "min")
    best_scene_iou = best_model(summary_rows, models, "scene_aligned_voxel_iou_mean", "max")
    best_object_iou = best_model(summary_rows, models, "object_aligned_voxel_iou_mean", "max")
    best_physics = physics_best_models(summary_rows, models)
    best_collision = best_physics.get("collision", "")
    best_scale = best_physics.get("scale", "")
    best_accel = best_physics.get("acceleration", "")
    best_ate = best_physics.get("ate", "")
    best_frechet = best_physics.get("frechet", "")
    physics_best = strongest_physics_model(summary_rows, models)

    lines = [
        f"Here is the summary from the full {len(samples)}-sample / {len(models)}-model run.",
        "",
        "Metrics are sample-level means; Chamfer lower is better, F-score and voxel IoU higher are better.",
        "",
        "## Main Reconstruction Metrics",
        "",
        markdown_table(
            [
                "Model",
                "Aligned Scene Chamfer ↓",
                "Aligned Scene IoU ↑",
                "Aligned Object Chamfer ↓",
                "Aligned Object F ↑",
                "Aligned Object IoU ↑",
            ],
            reconstruction_rows,
        ),
        "",
        "## Per-Sample Wins",
        "",
        markdown_table(["Metric", "Winner Count"], win_rows),
        "",
        "Object voxel IoU can have ties, so counts can exceed the number of samples.",
        "",
        "## Physics Metrics",
        "",
        markdown_table(["Model"] + [label for label, _metric, _precision in physics_metric_table_spec()], physics_rows),
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
        t_p = test_metric(test_rows, best_object_chamfer, "object_aligned_chamfer_distance_mean", "paired_t_p")
        w_p = test_metric(test_rows, best_object_chamfer, "object_aligned_chamfer_distance_mean", "wilcoxon_p")
        if best_object_chamfer == base_model_tag:
            lines.append(f"{base_model_tag} has the best aligned object Chamfer mean among all models.")
        else:
            significance = "significant" if (not math.isnan(t_p) and t_p < 0.05) or (not math.isnan(w_p) and w_p < 0.05) else "not significant"
            lines.append(
                f"{best_object_chamfer} best matches visual/object geometry by aligned object Chamfer. "
                f"Versus {base_model_tag}, this object-Chamfer difference is {significance}: "
                f"paired t-test p={fmt(t_p, 3)}, Wilcoxon p={fmt(w_p, 3)}."
            )
    lines.append("")

    lines.append(f"{best_scene_iou} is strongest on aligned scene voxel IoU, and {best_object_iou} is strongest on aligned object voxel IoU.")
    lines.append("")

    if physics_best:
        if INCLUDE_BASIC_PHYSICS_SUMMARY:
            lines.append(
                f"{physics_best} is strongest on the selected physics metrics overall: "
                f"best collision model is {best_collision}, best scale-error model is {best_scale}, "
                f"and best acceleration-smoothness model is {best_accel}."
            )
        else:
            lines.append(
                f"{physics_best} is strongest on the selected trajectory metrics overall: "
                f"best ATE model is {best_ate}, and best Fréchet model is {best_frechet}."
            )
        lines.append("")

    if best_object_chamfer and physics_best and best_object_chamfer != physics_best:
        lines.append(
            f"So: if you care most about visual/object geometry, {best_object_chamfer} is the best candidate by Chamfer. "
            f"If you care most about physical plausibility, {physics_best} is strongest. "
            f"If you care strictly about voxel IoU, compare against {best_scene_iou}/{best_object_iou}."
        )
    elif best_object_chamfer:
        lines.append(f"So: {best_object_chamfer} is the main candidate from this metric subset, but inspect animations before making the final choice.")

    return "\n".join(lines).rstrip() + "\n"

def aggregate_case_weighted(model_rows: list[dict[str, Any]], base_model_tag: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate by case means so each physics case has equal weight."""
    add_case_names(model_rows)
    summary_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    metrics = metric_specs()
    models = sorted({row["model_tag"] for row in model_rows})
    cases = sorted({row["case_name"] for row in model_rows})

    case_mean_by_key: dict[tuple[str, str, str], float] = {}
    for model in models:
        for case in cases:
            case_rows = [row for row in model_rows if row.get("model_tag") == model and row.get("case_name") == case]
            for metric, _group in metrics:
                case_mean_by_key[(case, model, metric)] = mean([to_float(row.get(metric)) for row in case_rows])

    for model in models:
        for metric, group in metrics:
            values = [case_mean_by_key[(case, model, metric)] for case in cases]
            valid = [value for value in values if not math.isnan(value)]
            summary_rows.append(
                {
                    "model_tag": model,
                    "metric_group": group,
                    "metric": metric,
                    "aggregation": "equal_case_mean",
                    "n": len(valid),
                    "mean": mean(values),
                    "std": stdev(values),
                    "median": median(values),
                    "min": min(valid, default=float("nan")),
                    "max": max(valid, default=float("nan")),
                    "cases": ";".join(case for case in cases if not math.isnan(case_mean_by_key[(case, model, metric)])),
                }
            )

    for model in models:
        if model == base_model_tag:
            continue
        for metric, group in metrics:
            base_vals = []
            model_vals = []
            paired_cases = []
            for case in cases:
                base_value = case_mean_by_key.get((case, base_model_tag, metric), float("nan"))
                model_value = case_mean_by_key.get((case, model, metric), float("nan"))
                if math.isnan(base_value) or math.isnan(model_value):
                    continue
                base_vals.append(base_value)
                model_vals.append(model_value)
                paired_cases.append(case)
            deltas = [m - b for m, b in zip(model_vals, base_vals)]
            tests = maybe_scipy_tests(model_vals, base_vals)
            test_rows.append(
                {
                    "model_tag": model,
                    "baseline_tag": base_model_tag,
                    "metric_group": group,
                    "metric": metric,
                    "aggregation": "equal_case_mean",
                    "n_pairs": len(deltas),
                    "model_mean": mean(model_vals),
                    "baseline_mean": mean(base_vals),
                    "mean_delta_model_minus_base": mean(deltas),
                    "median_delta_model_minus_base": median(deltas),
                    "paired_t_p": tests["paired_t_p"],
                    "wilcoxon_p": tests["wilcoxon_p"],
                    "paired_cases": ";".join(paired_cases),
                }
            )
    return summary_rows, test_rows


def aggregate(model_rows: list[dict[str, Any]], base_model_tag: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    metrics = metric_specs()
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
    global INCLUDE_BASIC_PHYSICS_SUMMARY
    args = parse_args()
    INCLUDE_BASIC_PHYSICS_SUMMARY = not args.skip_basic_physics_summary
    if args.quick_metrics and not args.quick:
        raise SystemExit("--quick-metrics requires --quick")
    if args.quick:
        if args.sample is None:
            args.sample = list(QUICK_SAMPLES)
        args.parallel_workers = 1
        args.only_inference = not args.quick_metrics
    dataset_root = args.dataset_root.expanduser().resolve()
    eval_root = args.eval_root.expanduser().resolve()
    predictions_dir = eval_root / "predictions"
    metrics_dir = eval_root / "metrics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(dataset_root, args.sample_glob, args.sample)
    if args.frames_dir is not None:
        args.frames_dir = args.frames_dir.expanduser().resolve()
        if len(samples) != 1:
            raise SystemExit("--frames-dir requires exactly one selected sample")
        if not args.frames_dir.is_dir():
            raise SystemExit(f"--frames-dir does not exist or is not a directory: {args.frames_dir}")
        print(f"Frame override: {args.frames_dir}", flush=True)

    models: list[tuple[str, Path | None]] = []
    if not args.no_discover_prediction_models:
        discovered_tags = discover_prediction_model_tags(predictions_dir, samples)
        if discovered_tags:
            models.extend((tag, None) for tag in discovered_tags)
        elif args.model is None and args.extra_model is None:
            print("[warn] no complete prediction-model intersection found; falling back to DEFAULT_MODELS", flush=True)
            models.extend((tag, resolve_transformer(path)) for tag, path in DEFAULT_MODELS)
    elif args.model is None and args.extra_model is None:
        models.extend((tag, resolve_transformer(path)) for tag, path in DEFAULT_MODELS)

    def add_or_replace_model(tag: str, transformer: Path) -> None:
        nonlocal models
        for idx, (existing_tag, _) in enumerate(models):
            if existing_tag == tag:
                models[idx] = (tag, transformer)
                return
        models.append((tag, transformer))

    for tag, path in args.model or []:
        add_or_replace_model(tag, resolve_transformer(path))
    for tag, path in args.extra_model or []:
        add_or_replace_model(tag, resolve_transformer(path))
    if args.quick:
        mode = "metrics on quick subset" if args.quick_metrics else "inference-only visual inspection"
        print(f"Quick mode: {mode}; parallel_workers={args.parallel_workers}", flush=True)
    print(f"Samples ({len(samples)}): {', '.join(samples)}", flush=True)
    print("Models: " + ", ".join(f"{tag}={model_path_for_log(path) or '<discovered-prediction>'}" for tag, path in models), flush=True)

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
                if existing is not None and (args.skip_existing_metrics or args.reuse_predictions) and not args.force_metrics:
                    row, per_frame = existing
                    pred_dir_value = row.get("pred_dir")
                    if pred_dir_value:
                        render_statistical_diagnostic_gifs(
                            args,
                            sample,
                            model_tag,
                            Path(str(pred_dir_value)),
                            raw_dir,
                            input_dir,
                            eval_root / "metrics",
                        )
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
        write_csv(eval_root / "physics_model_runs.csv", run_rows)
        print(f"Wrote run table: {eval_root / 'physics_model_runs.csv'}")
        print("Visual inspection checklist:")
        print("  - identity preservation")
        print("  - actual motion")
        print("  - no object swapping")
        print("  - collision response plausibility")
        print("  - no collapse/explosion")
        print("Run full metrics only after this visual pass; use --quick --quick-metrics for a small metric subset.")
        return
    add_case_names(model_rows)
    add_case_names(run_rows)
    add_case_names(per_frame_rows)
    sample_summary_rows, sample_test_rows = aggregate(model_rows, args.base_model_tag)
    summary_rows, test_rows = aggregate_case_weighted(model_rows, args.base_model_tag)
    write_csv(eval_root / "physics_model_runs.csv", run_rows)
    write_csv(eval_root / "physics_per_frame_metrics_full.csv", per_frame_rows)
    write_csv(eval_root / "physics_statistical_summary.csv", summary_rows)
    write_csv(eval_root / "physics_statistical_tests.csv", test_rows)
    write_csv(eval_root / "physics_sample_weighted_statistical_summary.csv", sample_summary_rows)
    write_csv(eval_root / "physics_sample_weighted_statistical_tests.csv", sample_test_rows)

    case_report_dir = eval_root / "case_reports"
    case_report_dir.mkdir(parents=True, exist_ok=True)
    case_report_paths = []
    case_summary_rows = []
    case_test_rows = []
    for case in sorted({row.get("case_name") for row in model_rows}):
        case_rows = [row for row in model_rows if row.get("case_name") == case]
        one_case_summary, one_case_tests = aggregate(case_rows, args.base_model_tag)
        for row in one_case_summary:
            row["case_name"] = case
        for row in one_case_tests:
            row["case_name"] = case
        case_summary_rows.extend(one_case_summary)
        case_test_rows.extend(one_case_tests)
        case_report = generate_case_group_report(str(case), case_rows, args.base_model_tag)
        case_report_path = case_report_dir / f"{case}_statistical_report.md"
        case_report_path.write_text(case_report, encoding="utf-8")
        case_report_paths.append(case_report_path)
    write_csv(eval_root / "physics_case_statistical_summary.csv", case_summary_rows)
    write_csv(eval_root / "physics_case_statistical_tests.csv", case_test_rows)

    report = generate_report(model_rows, summary_rows, test_rows, args.base_model_tag)
    report_path = eval_root / "physics_statistical_report.md"
    report = (
        "Aggregate statistics use equal case weighting: ball_drop, rolling_occluder, two_ball, and wall_impact each contribute 1/4 when present.\n\n"
        + report
    )
    report_path.write_text(report, encoding="utf-8")

    physics_report = generate_physics_report(model_rows, summary_rows, test_rows, args.base_model_tag)
    physics_report_path = eval_root / "physics_aggregate_physics_report.md"
    physics_report_path.write_text(physics_report, encoding="utf-8")

    print(f"Wrote run table: {eval_root / 'physics_model_runs.csv'}")
    print(f"Wrote per-frame table: {eval_root / 'physics_per_frame_metrics_full.csv'}")
    print(f"Wrote equal-case summary table: {eval_root / 'physics_statistical_summary.csv'}")
    print(f"Wrote equal-case paired tests: {eval_root / 'physics_statistical_tests.csv'}")
    print(f"Wrote per-case summary table: {eval_root / 'physics_case_statistical_summary.csv'}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote physics report: {physics_report_path}")
    print(f"Wrote {len(case_report_paths)} case reports under: {case_report_dir}")
    print()
    print(report)


if __name__ == "__main__":
    main()

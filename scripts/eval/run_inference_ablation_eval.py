#!/usr/bin/env python3
"""Run controlled COM4D inference ablations for a single physics sample.

The default ablation grid is for diagnosing the overfit memorization setting:
8 vs 48 frames, AR block size 8 vs 4, and guidance 1.0 vs 7.0.
Each variant runs inference, reconstruction metrics, physics metrics, and aligned GIF rendering.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
INFERENCE = REPO_ROOT / "src" / "inference_com4d.py"
EVALUATE_RECONSTRUCTION = REPO_ROOT / "scripts" / "eval" / "evaluate_reconstruction.py"
EVALUATE_PHYSICS = REPO_ROOT / "scripts" / "eval" / "evaluate_physics.py"
RENDER_GIFS = REPO_ROOT / "scripts" / "eval" / "render_prediction_gifs.py"

DEFAULT_PYTHON = Path("/data/mseizde/micromamba/envs/com4d/bin/python")
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs" / "eval_inputs" / "physics_train_4_clean_static"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "overfit_inference_ablation_ball_drop"
DEFAULT_TRANSFORMER = (
    PROJECT_ROOT
    / "outputs"
    / "ckpts"
    / "overfit_phys4clean_seqnorm32768_ballonly_bs16"
    / "checkpoints"
    / "006000"
    / "transformer"
)
DEFAULT_BASE_WEIGHTS = "pretrained_weights/TripoSG"


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    transformer_dir: Path


@dataclass(frozen=True)
class Variant:
    tag: str
    frames: int
    ar_block: int
    guidance: float
    description: str


VARIANTS = [
    Variant("8f_b8_g1", 8, 8, 1.0, "clean baseline"),
    Variant("8f_b4_g1", 8, 4, 1.0, "AR block effect only"),
    Variant("8f_b8_g7", 8, 8, 7.0, "CFG effect only"),
    Variant("8f_b4_g7", 8, 4, 7.0, "runner-like settings, short rollout"),
    Variant("48f_b8_g1", 48, 8, 1.0, "rollout length effect"),
    Variant("48f_b4_g1", 48, 4, 1.0, "rollout plus AR block"),
    Variant("48f_b4_g7", 48, 4, 7.0, "closest to old runner behavior"),
]


def parse_model(raw: str) -> ModelSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("models must be TAG=TRANSFORMER_DIR")
    tag, path = raw.split("=", 1)
    tag = tag.strip()
    path = path.strip()
    if not tag:
        raise argparse.ArgumentTypeError("model tag cannot be empty")
    if not path:
        raise argparse.ArgumentTypeError("model path cannot be empty")
    return ModelSpec(tag, Path(path).expanduser())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--sample", default="ball_drop_000005")
    ap.add_argument("--model-tag", default="overfit_006000", help="Tag used with --transformer-dir when --model is not provided.")
    ap.add_argument("--transformer-dir", type=Path, default=DEFAULT_TRANSFORMER, help="Single-model transformer dir used when --model is not provided.")
    ap.add_argument(
        "--model",
        action="append",
        type=parse_model,
        default=None,
        help="TAG=TRANSFORMER_DIR. Can be repeated to run the full ablation grid for multiple checkpoints/models.",
    )
    ap.add_argument("--base-weights-dir", default=DEFAULT_BASE_WEIGHTS)
    ap.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value for inference.")
    ap.add_argument("--gpu-id", default=None, help="Alias for --cuda-visible-devices when using one GPU id.")
    ap.add_argument("--variant", action="append", choices=[v.tag for v in VARIANTS], help="Run only selected variant(s).")
    ap.add_argument("--force-inference", action="store_true", help="Rerun inference even when args.json exists for a variant.")
    ap.add_argument("--force-metrics", action="store_true", help="Rerun reconstruction/physics metrics even when metrics.json exists.")
    ap.add_argument("--force-gifs", action="store_true", help="Overwrite aligned GIFs even when they exist.")
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--skip-reconstruction", action="store_true")
    ap.add_argument("--skip-physics", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--num-tokens", type=int, default=1024)
    ap.add_argument("--dynamic-max-memory-frames", type=int, default=8)
    ap.add_argument("--dynamic-num-parts", type=int, default=None, help="Defaults to 2 for two_ball samples, else 1.")
    ap.add_argument("--scene-num-parts", type=int, default=0)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--first-frame-index", type=int, default=0)
    ap.add_argument("--frames-start-idx", type=int, default=0)
    ap.add_argument("--mesh-dense-depth", type=int, default=7)
    ap.add_argument("--mesh-hierarchical-depth", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    ap.add_argument("--no-animation", action="store_true", help="Do not ask inference_com4d.py to render its own animation.gif.")

    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--metric", default="l2")
    ap.add_argument("--alignment", choices=("none", "translation", "similarity", "first_frame_similarity"), default="similarity")
    ap.add_argument("--object-assignment", choices=("fixed", "best"), default="best")
    ap.add_argument("--com-method", choices=("vertex_centroid", "volume_center_mass", "bbox_center"), default="vertex_centroid")
    ap.add_argument("--skip-raw-metrics", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-voxel-iou", action="store_true")

    ap.add_argument("--render-size", type=int, default=512)
    ap.add_argument("--output-name", default="animation_fixed_regenerated.gif")
    ap.add_argument("--diagnostic-output-name", default="animation_diagnostic.gif")
    ap.add_argument("--gt-overlay-output-name", default="animation_gt_overlay.gif")
    ap.add_argument("--default-orbit-output-name", default="animation_default_orbit.gif")
    return ap.parse_args()


def selected_variants(names: list[str] | None) -> list[Variant]:
    if not names:
        return list(VARIANTS)
    requested = set(names)
    return [variant for variant in VARIANTS if variant.tag in requested]


def selected_models(args: argparse.Namespace) -> list[ModelSpec]:
    if args.model:
        return [ModelSpec(model.tag, model.transformer_dir.expanduser().resolve()) for model in args.model]
    return [ModelSpec(args.model_tag, args.transformer_dir.expanduser().resolve())]


def sample_dynamic_num_parts(sample: str, override: int | None) -> int:
    if override is not None:
        return override
    return 2 if "two_ball" in sample else 1


def run(cmd: list[object], *, cwd: Path, env: dict[str, str] | None, dry_run: bool) -> None:
    text = " ".join(str(part) for part in cmd)
    print(f"+ {text}", flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], cwd=str(cwd), env=env, check=True)


def variant_tag(args: argparse.Namespace, variant: Variant) -> str:
    return f"{args.model_tag}_{variant.tag}"


def paths_for(args: argparse.Namespace, variant: Variant) -> dict[str, Path]:
    tag = variant_tag(args, variant)
    sample_root = args.dataset_root / "inference_input" / args.sample
    gt_root = args.dataset_root / "gt_raw" / args.sample
    pred_parent = args.eval_root / "predictions" / args.sample
    pred_export = pred_parent / tag
    metrics = args.eval_root / "metrics" / args.sample / tag
    return {
        "frames": sample_root / "frames",
        "masks": sample_root / "masks",
        "masks_static": sample_root / "masks_static",
        "gt": gt_root,
        "metadata": gt_root / "physics_metadata.json",
        "pred_parent": pred_parent,
        "pred_export": pred_export,
        "metrics": metrics,
        "reconstruction": metrics / "reconstruction",
        "physics": metrics / "physics",
    }


def ensure_inputs(args: argparse.Namespace, variant: Variant) -> None:
    paths = paths_for(args, variant)
    required = [paths["frames"], paths["masks"], paths["masks_static"], paths["gt"], paths["metadata"]]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise SystemExit(f"Missing required input path(s):\n  {joined}")


def run_inference(args: argparse.Namespace, variant: Variant, env: dict[str, str] | None) -> None:
    paths = paths_for(args, variant)
    args_json = paths["pred_export"] / "args.json"
    if args_json.is_file() and not args.force_inference:
        print(f"[skip] inference exists: {args_json}", flush=True)
        return
    paths["pred_parent"].mkdir(parents=True, exist_ok=True)
    cmd: list[object] = [
        args.python,
        INFERENCE,
        "--frames_dir", paths["frames"],
        "--masks_dir", paths["masks"],
        "--masks_static_dir", paths["masks_static"],
        "--output_dir", paths["pred_parent"],
        "--tag", variant_tag(args, variant),
        "--no-timestamp-output",
        "--transformer_dir", args.transformer_dir,
        "--base_weights_dir", args.base_weights_dir,
        "--num_tokens", args.num_tokens,
        "--first_frame_index", args.first_frame_index,
        "--frames_start_idx", args.frames_start_idx,
        "--frames_end_idx", variant.frames,
        "--frame_stride", args.frame_stride,
        "--scene_num_parts", args.scene_num_parts,
        "--dynamic_num_parts", sample_dynamic_num_parts(args.sample, args.dynamic_num_parts),
        "--dynamic_ar_block_size", variant.ar_block,
        "--dynamic_max_memory_frames", args.dynamic_max_memory_frames,
        "--dynamic_guidance", variant.guidance,
        "--mesh_dense_depth", args.mesh_dense_depth,
        "--mesh_hierarchical_depth", args.mesh_hierarchical_depth,
        "--object_only_condition",
        "--image_size", args.image_size,
        "--device", args.device,
        "--dtype", args.dtype,
        "--no-render_predicted_room",
        "--no-room_augment_animations",
    ]
    if not args.no_animation:
        cmd.append("--animation")
    run(cmd, cwd=REPO_ROOT, env=env, dry_run=args.dry_run)


def run_reconstruction(args: argparse.Namespace, variant: Variant, env: dict[str, str] | None) -> None:
    paths = paths_for(args, variant)
    metrics_json = paths["reconstruction"] / "metrics.json"
    if metrics_json.is_file() and not args.force_metrics:
        print(f"[skip] reconstruction metrics exist: {metrics_json}", flush=True)
        return
    paths["reconstruction"].mkdir(parents=True, exist_ok=True)
    cmd: list[object] = [
        args.python,
        EVALUATE_RECONSTRUCTION,
        "--pred-dir", paths["pred_export"],
        "--gt-dir", paths["gt"],
        "--output-dir", paths["reconstruction"],
        "--num-samples", args.num_samples,
        "--threshold", args.threshold,
        "--metric", args.metric,
        "--alignment", args.alignment,
        "--object-assignment", args.object_assignment,
        "--com-method", args.com_method,
    ]
    if args.skip_raw_metrics:
        cmd.append("--skip-raw-metrics")
    if args.skip_voxel_iou:
        cmd.append("--skip-voxel-iou")
    run(cmd, cwd=REPO_ROOT, env=env, dry_run=args.dry_run)


def run_physics(args: argparse.Namespace, variant: Variant, env: dict[str, str] | None) -> None:
    paths = paths_for(args, variant)
    metrics_json = paths["physics"] / "metrics.json"
    if metrics_json.is_file() and not args.force_metrics:
        print(f"[skip] physics metrics exist: {metrics_json}", flush=True)
        return
    paths["physics"].mkdir(parents=True, exist_ok=True)
    cmd: list[object] = [
        args.python,
        EVALUATE_PHYSICS,
        "--inference-dir", paths["pred_export"],
        "--metadata", paths["metadata"],
        "--output-dir", paths["physics"],
        "--com-method", args.com_method,
    ]
    run(cmd, cwd=REPO_ROOT, env=env, dry_run=args.dry_run)


def run_render(args: argparse.Namespace, variant: Variant, env: dict[str, str] | None) -> None:
    paths = paths_for(args, variant)
    if not args.dry_run and not (paths["reconstruction"] / "metrics.json").is_file():
        raise SystemExit(f"Cannot render without reconstruction metrics: {paths['reconstruction'] / 'metrics.json'}")
    output_path = paths["pred_export"] / args.output_name
    if output_path.is_file() and not args.force_gifs:
        print(f"[skip] rendered GIF exists: {output_path}", flush=True)
        return
    cmd: list[object] = [
        args.python,
        RENDER_GIFS,
        "--export-dir", paths["pred_export"],
        "--output-name", args.output_name,
        "--diagnostic-output-name", args.diagnostic_output_name,
        "--gt-overlay-output-name", args.gt_overlay_output_name,
        "--default-orbit-output-name", args.default_orbit_output_name,
        "--overwrite",
        "--render-size", args.render_size,
        "--camera-metadata", paths["metadata"],
        "--alignment-metadata", paths["reconstruction"] / "metrics.json",
        "--gt-geometry-root", paths["gt"],
        "--source-frames-dir", paths["frames"],
    ]
    run(cmd, cwd=REPO_ROOT, env=env, dry_run=args.dry_run)


def print_summary(args: argparse.Namespace, variants: list[Variant], models: list[ModelSpec]) -> None:
    print("Models:", flush=True)
    for model in models:
        print(f"  {model.tag}={model.transformer_dir}", flush=True)
    print("Ablation variants:", flush=True)
    for variant in variants:
        print(
            f"  {variant.tag}: frames={variant.frames}, ar_block={variant.ar_block}, "
            f"guidance={variant.guidance:g} ({variant.description})",
            flush=True,
        )
    print(f"Predictions: {args.eval_root / 'predictions' / args.sample}", flush=True)
    print(f"Metrics:     {args.eval_root / 'metrics' / args.sample}", flush=True)


def main() -> None:
    args = parse_args()
    args.python = args.python.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.eval_root = args.eval_root.expanduser().resolve()
    models = selected_models(args)
    variants = selected_variants(args.variant)
    if not variants:
        raise SystemExit("No variants selected")

    for variant in variants:
        ensure_inputs(args, variant)

    env = os.environ.copy()
    cuda_value = args.cuda_visible_devices if args.cuda_visible_devices is not None else args.gpu_id
    if cuda_value is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_value)

    print_summary(args, variants, models)
    for model in models:
        args.model_tag = model.tag
        args.transformer_dir = model.transformer_dir
        print(f"\n### model {model.tag}: {model.transformer_dir} ###", flush=True)
        for variant in variants:
            print(f"\n=== {model.tag}_{variant.tag}: {variant.description} ===", flush=True)
            if not args.skip_inference:
                run_inference(args, variant, env)
            if not args.skip_reconstruction:
                run_reconstruction(args, variant, env)
            if not args.skip_physics:
                run_physics(args, variant, env)
            if not args.skip_render:
                run_render(args, variant, env)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Round-trip GT physics surfaces through the frozen TripoSG VAE."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.models.autoencoders import TripoSGVAEModel
from src.utils.inference_utils import field_to_mesh, hierarchical_extract_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "dataset_json/physics_train_4_clean_static.json")
    parser.add_argument("--sample", action="append", default=None)
    parser.add_argument("--object-prefix", default="ball_")
    parser.add_argument("--output-dir", type=Path, default=REPO.parent / "outputs/gt_vae_roundtrip/physics_train_4_clean_static")
    parser.add_argument(
        "--com4d-pred-root",
        type=Path,
        default=None,
        help="Optional root for COM4D-compatible prediction dirs, one subdir per sample.",
    )
    parser.add_argument(
        "--eval-gt-root",
        type=Path,
        default=None,
        help="Optional eval-style GT root containing gt_raw/<sample> or raw sample dirs.",
    )
    parser.add_argument(
        "--eval-output-root",
        type=Path,
        default=None,
        help="Optional metrics root. Defaults to <com4d-pred-root>/../metrics when evaluation is enabled.",
    )
    parser.add_argument("--eval-num-samples", type=int, default=10000)
    parser.add_argument("--eval-threshold", type=float, default=0.1)
    parser.add_argument("--eval-metric", default="l2")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--skip-render-gifs",
        action="store_true",
        help="Do not render animation.gif/diagnostic GIFs after writing COM4D-compatible predictions.",
    )
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--render-fps", type=int, default=3, help="GIF FPS; use 0 to let the renderer use GT metadata FPS.")
    parser.add_argument("--overwrite-render-gifs", action="store_true")
    parser.add_argument("--weights", type=Path, default=REPO / "pretrained_weights/TripoSG")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--surface-points", type=int, default=8192)
    parser.add_argument("--metric-points", type=int, default=10000)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--max-frames-per-sample", type=int, default=6)
    parser.add_argument("--dense-depth", type=int, default=7)
    parser.add_argument("--hierarchical-depth", type=int, default=8)
    parser.add_argument("--mesh-margin", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_part(frame: dict, object_index: int) -> np.ndarray:
    data = np.load(frame["surface_path"], allow_pickle=True).item()
    part = data["parts"][object_index]
    points = np.asarray(part["surface_points"], dtype=np.float32)
    normals = np.asarray(part["surface_normals"], dtype=np.float32)
    return np.concatenate([points, normals], axis=-1)


def choose_frames(frames: list[dict], stride: int, max_count: int) -> list[int]:
    indices = list(range(0, len(frames), max(1, stride)))
    if max_count > 0:
        indices = indices[:max_count]
    return indices


def subsample(surface: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or surface.shape[0] <= count:
        return surface.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    return surface[rng.choice(surface.shape[0], count, replace=False)].astype(np.float32, copy=False)


def point_chamfer(mesh: trimesh.Trimesh, target_points: np.ndarray, count: int, seed: int) -> dict:
    if mesh is None or mesh.is_empty:
        return {"chamfer_l1": math.inf, "chamfer_l2": math.inf}
    rng = np.random.default_rng(seed)
    predicted, _ = trimesh.sample.sample_surface(mesh, count, seed=rng)
    if len(target_points) > count:
        target_points = target_points[rng.choice(len(target_points), count, replace=False)]
    pred_to_target = cKDTree(target_points).query(predicted, workers=-1)[0]
    target_to_pred = cKDTree(predicted).query(target_points, workers=-1)[0]
    return {
        "chamfer_l1": float(pred_to_target.mean() + target_to_pred.mean()),
        "chamfer_l2": float((pred_to_target**2).mean() + (target_to_pred**2).mean()),
    }


def normalize_surface(surface_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    points = surface_np[:, :3].astype(np.float32, copy=False)
    normals = surface_np[:, 3:].astype(np.float32, copy=False)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = ((bounds_min + bounds_max) * 0.5).astype(np.float32)
    max_extent = float((bounds_max - bounds_min).max())
    scale = 2.0 / max(max_extent, 1e-8)
    normalized_points = (points - center[None]) * scale
    normalized = np.concatenate([normalized_points, normals], axis=-1).astype(np.float32, copy=False)
    return normalized, center.astype(np.float64), float(scale)


def denormalize_mesh(mesh: trimesh.Trimesh, center: np.ndarray, scale: float) -> trimesh.Trimesh:
    output = mesh.copy()
    output.apply_scale(1.0 / max(float(scale), 1e-8))
    output.apply_translation(center)
    return output


def reset_dir(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def combine_scene(meshes: list[trimesh.Trimesh]) -> trimesh.Scene:
    scene = trimesh.Scene()
    for index, mesh in enumerate(meshes):
        if mesh is not None and not mesh.is_empty:
            scene.add_geometry(mesh.copy(), geom_name=f"object_{index:03d}")
    return scene


def gt_sample_dir(eval_gt_root: Path, sample: str) -> Path:
    candidate = eval_gt_root / "gt_raw" / sample
    if candidate.is_dir():
        return candidate
    candidate = eval_gt_root / sample
    if candidate.is_dir():
        return candidate
    return eval_gt_root


def run_eval(pred_dir: Path, gt_dir: Path, metadata: Path, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/eval/evaluate_reconstruction.py"),
            "--pred-dir",
            str(pred_dir),
            "--gt-dir",
            str(gt_dir),
            "--output-dir",
            str(output_dir / "reconstruction"),
            "--num-samples",
            str(args.eval_num_samples),
            "--threshold",
            str(args.eval_threshold),
            "--metric",
            str(args.eval_metric),
            "--alignment",
            "first_frame_similarity",
            "--object-assignment",
            "best",
        ],
        cwd=str(REPO),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/eval/evaluate_physics.py"),
            "--inference-dir",
            str(pred_dir),
            "--metadata",
            str(metadata),
            "--output-dir",
            str(output_dir / "physics"),
        ],
        cwd=str(REPO),
        check=True,
    )


def render_gifs(pred_dir: Path, raw_dir: Path, metrics_dir: Path, args: argparse.Namespace) -> None:
    recon_json = metrics_dir / "reconstruction" / "metrics.json"
    metadata = raw_dir / "physics_metadata.json"
    frames_dir = raw_dir / "render_rgb"
    if not recon_json.is_file():
        print(f"[warn] skipping GIF render for {pred_dir}: missing {recon_json}", flush=True)
        return
    if not metadata.is_file():
        print(f"[warn] skipping GIF render for {pred_dir}: missing {metadata}", flush=True)
        return
    if not frames_dir.is_dir():
        print(f"[warn] skipping GIF render for {pred_dir}: missing {frames_dir}", flush=True)
        return
    cmd = [
        sys.executable,
        str(REPO / "scripts/eval/render_prediction_gifs.py"),
        "--export-dir",
        str(pred_dir),
        "--camera-metadata",
        str(metadata),
        "--alignment-metadata",
        str(recon_json),
        "--gt-geometry-root",
        str(raw_dir),
        "--source-frames-dir",
        str(frames_dir),
        "--output-name",
        "animation.gif",
        "--diagnostic-output-name",
        "animation_diagnostic.gif",
        "--gt-overlay-output-name",
        "animation_gt_overlay.gif",
        "--no-default-orbit-artifact",
        "--render-size",
        str(args.render_size),
    ]
    if args.render_fps and args.render_fps > 0:
        cmd.extend(["--fps", str(args.render_fps)])
    if args.overwrite_render_gifs:
        cmd.append("--overwrite")
    subprocess.run(cmd, cwd=str(REPO), check=True)


@torch.no_grad()
def encode_decode_mesh(
    vae: TripoSGVAEModel,
    surface_np: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_tokens: int,
    seed: int,
    dense_depth: int,
    hierarchical_depth: int,
    mesh_margin: float,
) -> trimesh.Trimesh:
    normalized_surface, center, scale = normalize_surface(surface_np)
    surface = torch.from_numpy(normalized_surface)[None].to(device=device, dtype=dtype)
    latent = vae.encode(surface, num_tokens=num_tokens, seed=seed).latent_dist.mode()
    bound = 1.0 + float(mesh_margin)
    field = hierarchical_extract_fields(
        lambda x: vae.decode(latent, sampled_points=x).sample,
        device=device,
        dtype=dtype,
        bounds=(-bound, -bound, -bound, bound, bound, bound),
        dense_octree_depth=dense_depth,
        hierarchical_octree_depth=hierarchical_depth,
        max_num_expanded_coords=100000000,
    )
    normalized_mesh = field_to_mesh(field, (-bound, -bound, -bound, bound, bound, bound), hierarchical_depth, device)
    return denormalize_mesh(normalized_mesh, center, scale)


def main() -> None:
    args = parse_args()
    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    samples = args.sample or list(manifest)
    missing = [sample for sample in samples if sample not in manifest]
    if missing:
        raise KeyError(f"Samples missing from manifest: {missing}")

    vae = TripoSGVAEModel.from_pretrained(args.weights, subfolder="vae").to(device=device, dtype=dtype)
    vae.eval().requires_grad_(False)
    vae.set_flash_decoder()

    rows = []
    for sample in samples:
        frames = manifest[sample]
        object_names = frames[0].get("object_names", [])
        object_indices = [i for i, name in enumerate(object_names) if str(name).startswith(args.object_prefix)]
        object_track_ids = {object_index: track_idx for track_idx, object_index in enumerate(object_indices)}
        pred_dir = None
        pred_dynamic_dir = None
        pred_frame_meshes: dict[int, list[trimesh.Trimesh]] = {}
        metadata_path = Path(frames[0].get("physics_metadata_path", ""))
        if args.com4d_pred_root is not None:
            pred_dir = args.com4d_pred_root / sample
            reset_dir(pred_dir)
            pred_dynamic_dir = pred_dir / "dynamic"
            pred_dynamic_dir.mkdir(parents=True, exist_ok=True)
            if metadata_path.is_file():
                shutil.copy2(metadata_path, pred_dir / "physics_metadata.json")

        for object_index in object_indices:
            object_name = object_names[object_index]
            pred_object_dir = None
            if pred_dynamic_dir is not None:
                pred_object_dir = pred_dynamic_dir / f"object_{object_track_ids[object_index]:03d}"
                pred_object_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in choose_frames(frames, args.frame_stride, args.max_frames_per_sample):
                surface_np = subsample(
                    load_part(frames[frame_index], object_index),
                    args.surface_points,
                    args.seed + frame_index + object_index,
                )
                mesh = encode_decode_mesh(
                    vae,
                    surface_np,
                    device=device,
                    dtype=dtype,
                    num_tokens=args.num_tokens,
                    seed=args.seed + frame_index + object_index,
                    dense_depth=args.dense_depth,
                    hierarchical_depth=args.hierarchical_depth,
                    mesh_margin=args.mesh_margin,
                )
                out_dir = args.output_dir / sample / object_name
                out_dir.mkdir(parents=True, exist_ok=True)
                mesh_path = out_dir / f"frame_{frame_index:04d}_vae_roundtrip.glb"
                mesh.export(mesh_path)
                if pred_object_dir is not None:
                    pred_mesh_path = pred_object_dir / f"frame_{frame_index:04d}.glb"
                    mesh.export(pred_mesh_path)
                    pred_frame_meshes.setdefault(frame_index, []).append(mesh)
                metrics = point_chamfer(mesh, surface_np[:, :3], args.metric_points, args.seed + frame_index)
                row = {
                    "sample": sample,
                    "object": object_name,
                    "frame_index": frame_index,
                    "mesh": str(mesh_path),
                    **metrics,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)

        if pred_dynamic_dir is not None:
            for frame_index, meshes in sorted(pred_frame_meshes.items()):
                combine_scene(meshes).export(pred_dynamic_dir / f"dynamic_scene_frame_{frame_index:04d}.glb")
            if (
                not args.skip_eval
                and args.eval_gt_root is not None
                and pred_dir is not None
                and (pred_dir / "physics_metadata.json").is_file()
            ):
                eval_root = args.eval_output_root
                if eval_root is None:
                    eval_root = args.com4d_pred_root.parent / "metrics"
                raw_dir = gt_sample_dir(args.eval_gt_root, sample)
                sample_metrics_dir = eval_root / sample
                run_eval(
                    pred_dir,
                    raw_dir,
                    pred_dir / "physics_metadata.json",
                    sample_metrics_dir,
                    args,
                )
                if not args.skip_render_gifs:
                    render_gifs(pred_dir, raw_dir, sample_metrics_dir, args)

    (args.output_dir / "metrics.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()

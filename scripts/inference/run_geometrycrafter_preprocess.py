#!/usr/bin/env python3
"""Run GeometryCrafter on COM4D frame folders and save geometry priors.

Run from the dedicated GeometryCrafter environment, not from ``com4d``:

    micromamba run -n geometrycrafter python scripts/inference/run_geometrycrafter_preprocess.py \
        --frames-dir assets/video/frames
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GeometryCrafter preprocessing for a COM4D frame directory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-dir", help="Directory containing extracted RGB frames.")
    source.add_argument("--video-path", help="Original video path. Frames are extracted without COM4D square padding.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <video_root>/geometrycrafter next to the frames directory.",
    )
    parser.add_argument("--geometrycrafter-repo", default="/data/mseizde/com4d/GeometryCrafter")
    parser.add_argument("--cache-dir", default="/data/mseizde/com4d/GeometryCrafter/workspace/cache")
    parser.add_argument("--height", type=int, default=384, help="Processing height; must be divisible by 64.")
    parser.add_argument("--width", type=int, default=384, help="Processing width; must be divisible by 64.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit the number of frames. 0 means no limit.")
    parser.add_argument("--model-type", choices=("determ", "diff"), default="determ")
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=110)
    parser.add_argument("--decode-chunk-size", type=int, default=6)
    parser.add_argument("--overlap", type=int, default=25)
    parser.add_argument("--low-memory-usage", action="store_true", default=True)
    parser.add_argument("--no-low-memory-usage", dest="low_memory_usage", action="store_false")
    parser.add_argument("--force-projection", action="store_true", default=True)
    parser.add_argument("--no-force-projection", dest="force_projection", action="store_false")
    parser.add_argument("--force-fixed-focal", action="store_true", default=True)
    parser.add_argument("--no-force-fixed-focal", dest="force_fixed_focal", action="store_false")
    parser.add_argument("--use-extract-interp", action="store_true")
    parser.add_argument("--save-float16", action="store_true", default=True)
    parser.add_argument("--save-float32", dest="save_float16", action="store_false")
    parser.add_argument("--no-preview-png", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_repo_to_path(repo: str) -> None:
    repo_path = Path(repo)
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))
        moge_path = repo_path / "third_party" / "moge"
        if moge_path.exists():
            sys.path.insert(0, str(moge_path))


def list_frames(frames_dir: Path, stride: int, max_frames: int) -> list[Path]:
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    frames = sorted(p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    frames = frames[::stride]
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"No image frames found in {frames_dir}")
    return frames


def default_output_dir(frames_dir: Path | None = None, video_path: Path | None = None) -> Path:
    if video_path is not None:
        return video_path.parent / f"{video_path.stem}_geometrycrafter"
    if frames_dir is None:
        raise ValueError("Either frames_dir or video_path is required")
    return frames_dir.parent / "geometrycrafter"


def extract_video_frames(video_path: Path, output_dir: Path) -> Path:
    frames_dir = output_dir / "input_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("*.png"):
        old_frame.unlink()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        str(frames_dir / "%06d.png"),
    ]
    subprocess.run(command, check=True)
    return frames_dir


def load_frames(frames: list[Path]) -> torch.Tensor:
    tensors = []
    for path in frames:
        image = Image.open(path)
        if getattr(image, "is_animated", False):
            image.seek(0)
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


def to_numpy(value: torch.Tensor | np.ndarray, save_float16: bool = False) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value.astype(np.float32, copy=False)
    else:
        array = value.detach().cpu().float().numpy()
    if save_float16:
        return array.astype(np.float16)
    return array.astype(np.float32)


def compute_normals_from_points(points: np.ndarray, valid_mask: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    points32 = points.astype(np.float32, copy=False)
    dx = np.zeros_like(points32)
    dy = np.zeros_like(points32)
    dx[:, :, 1:-1] = points32[:, :, 2:] - points32[:, :, :-2]
    dx[:, :, 0] = points32[:, :, 1] - points32[:, :, 0]
    dx[:, :, -1] = points32[:, :, -1] - points32[:, :, -2]
    dy[:, 1:-1, :] = points32[:, 2:, :] - points32[:, :-2, :]
    dy[:, 0, :] = points32[:, 1, :] - points32[:, 0, :]
    dy[:, -1, :] = points32[:, -1, :] - points32[:, -2, :]
    normals = np.cross(dx, dy)
    denom = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(denom, eps)
    normals[~np.isfinite(normals)] = 0.0
    if valid_mask is not None:
        normals = normals * valid_mask[..., None].astype(np.float32)
    return normals


def normalize_scalar_image(values: np.ndarray, value_range: tuple[float, float] | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    if value_range is None:
        lo, hi = np.percentile(values[finite], [2.0, 98.0])
    else:
        lo, hi = value_range
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values[finite].min())
        hi = float(values[finite].max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    normalized[~finite] = 0.0
    return (normalized * 255.0).astype(np.uint8)


def colorize_depth(depth: np.ndarray, value_range: tuple[float, float] | None = None) -> np.ndarray:
    gray = normalize_scalar_image(depth, value_range=value_range)
    x = gray.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def normal_to_rgb(normals: np.ndarray) -> np.ndarray:
    rgb = (np.asarray(normals, dtype=np.float32) * 0.5 + 0.5) * 255.0
    rgb[~np.isfinite(rgb)] = 0.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def write_dense_sequence(sequence: np.ndarray, names: list[str], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, name in enumerate(names):
        out_path = out_dir / f"{Path(name).stem}.npy"
        np.save(out_path, sequence[idx])
        paths.append(str(out_path))
    return paths


def write_png_sequence(sequence: np.ndarray, names: list[str], out_dir: Path, kind: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    value_range = None
    if kind in {"depth", "scalar"}:
        finite = sequence[np.isfinite(sequence)]
        if finite.size:
            lo, hi = np.percentile(finite.astype(np.float32), [2.0, 98.0])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                value_range = (float(lo), float(hi))
    for idx, name in enumerate(names):
        value = sequence[idx]
        if kind == "depth":
            image = colorize_depth(value, value_range=value_range)
        elif kind == "normal":
            image = normal_to_rgb(value)
        else:
            image = normalize_scalar_image(value, value_range=value_range)
        out_path = out_dir / f"{Path(name).stem}.png"
        Image.fromarray(image).save(out_path)
        paths.append(str(out_path))
    return paths


def load_geometrycrafter(args: argparse.Namespace):
    from diffusers.training_utils import set_seed
    from geometrycrafter import (
        GeometryCrafterDetermPipeline,
        GeometryCrafterDiffPipeline,
        PMapAutoencoderKLTemporalDecoder,
        UNetSpatioTemporalConditionModelVid2vid,
    )
    from third_party import MoGe

    set_seed(args.seed)
    os.makedirs(args.cache_dir, exist_ok=True)
    unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
        "TencentARC/GeometryCrafter",
        subfolder="unet_diff" if args.model_type == "diff" else "unet_determ",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
    ).requires_grad_(False).to("cuda", dtype=torch.float16)
    point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
        "TencentARC/GeometryCrafter",
        subfolder="point_map_vae",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir,
    ).requires_grad_(False).to("cuda", dtype=torch.float32)
    prior_model = MoGe(cache_dir=args.cache_dir).requires_grad_(False).to("cuda", dtype=torch.float32)
    pipe_cls = GeometryCrafterDiffPipeline if args.model_type == "diff" else GeometryCrafterDetermPipeline
    pipe = pipe_cls.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=args.cache_dir,
    ).to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        print(f"Xformers is not enabled: {exc}")
    pipe.enable_attention_slicing()
    return pipe, point_map_vae, prior_model


def main() -> None:
    args = parse_args()
    if args.height % 64 != 0 or args.width % 64 != 0:
        raise ValueError("--height and --width must be divisible by 64")
    if not torch.cuda.is_available():
        raise RuntimeError("GeometryCrafter inference requires CUDA")

    add_repo_to_path(args.geometrycrafter_repo)
    video_path = Path(args.video_path).resolve() if args.video_path else None
    frames_dir_arg = Path(args.frames_dir).resolve() if args.frames_dir else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(frames_dir_arg, video_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = extract_video_frames(video_path, output_dir) if video_path is not None else frames_dir_arg
    assert frames_dir is not None

    frames = list_frames(frames_dir, args.stride, args.max_frames)
    frame_names = [path.name for path in frames]
    video = load_frames(frames)

    print(f"Loading GeometryCrafter [{args.model_type}] using cache [{args.cache_dir}]")
    pipe, point_map_vae, prior_model = load_geometrycrafter(args)

    print(f"Running GeometryCrafter on {len(frames)} frame(s), original={tuple(video.shape[-2:])}, process={args.height}x{args.width}")
    with torch.inference_mode():
        point_maps, valid_mask = pipe(
            video,
            point_map_vae,
            prior_model,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            window_size=min(args.window_size, len(frames)),
            decode_chunk_size=args.decode_chunk_size,
            overlap=args.overlap,
            force_projection=args.force_projection,
            force_fixed_focal=args.force_fixed_focal,
            use_extract_interp=args.use_extract_interp,
            low_memory_usage=args.low_memory_usage,
        )

    point_maps_np = to_numpy(point_maps, args.save_float16)
    valid_mask_np = to_numpy(valid_mask, False).astype(np.bool_)
    depth_np = point_maps_np[..., 2]
    normals_np = compute_normals_from_points(point_maps_np, valid_mask_np)
    if args.save_float16:
        normals_np = normals_np.astype(np.float16)

    np.savez_compressed(
        output_dir / "geometrycrafter_outputs.npz",
        point_map=point_maps_np,
        depth=depth_np,
        normals=normals_np,
        mask=valid_mask_np,
        frames=np.array(frame_names),
    )
    outputs = {
        "depth": write_dense_sequence(depth_np, frame_names, output_dir / "depth"),
        "normals": write_dense_sequence(normals_np, frame_names, output_dir / "normals"),
        "point_maps": write_dense_sequence(point_maps_np, frame_names, output_dir / "point_maps"),
        "valid_mask": write_dense_sequence(valid_mask_np.astype(np.uint8), frame_names, output_dir / "valid_mask"),
    }
    previews = {}
    if not args.no_preview_png:
        previews = {
            "depth": write_png_sequence(depth_np.astype(np.float32), frame_names, output_dir / "preview_png" / "depth", "depth"),
            "normals": write_png_sequence(normals_np.astype(np.float32), frame_names, output_dir / "preview_png" / "normals", "normal"),
            "valid_mask": write_png_sequence(valid_mask_np.astype(np.float32), frame_names, output_dir / "preview_png" / "valid_mask", "scalar"),
        }

    summary = {
        "method": "GeometryCrafter",
        "model_type": args.model_type,
        "video_path": str(video_path) if video_path else None,
        "frames_dir": str(frames_dir),
        "num_frames": len(frames),
        "frames": frame_names,
        "original_frame_size": [int(video.shape[-2]), int(video.shape[-1])],
        "processing_size": [args.height, args.width],
        "cache_dir": args.cache_dir,
        "outputs": outputs,
        "preview_png": previews,
        "notes": "GeometryCrafter predicts point maps and valid masks; depth is point_map[..., 2]. It does not directly output camera parameters or tracks.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved GeometryCrafter preprocessing outputs to {output_dir}")


if __name__ == "__main__":
    main()

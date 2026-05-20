#!/usr/bin/env python3
"""Run VGGT on COM4D frame folders and save geometry priors.

This script is intentionally a preprocessing bridge.  VGGT has dependency pins
that conflict with COM4D/SAM2, so run it from the separate ``vggt`` environment:

    micromamba run -n vggt python scripts/inference/run_vggt_preprocess.py \
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
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT preprocessing for a COM4D frame directory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-dir", help="Directory containing extracted RGB frames.")
    source.add_argument("--video-path", help="Original video path. Frames are extracted without COM4D square padding.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <video_root>/vggt next to the frames directory.",
    )
    parser.add_argument(
        "--vggt-repo",
        default="/data/mseizde/com4d/vggt",
        help="VGGT repository path to add to PYTHONPATH when VGGT is not installed as a package.",
    )
    parser.add_argument("--model-id", default="facebook/VGGT-1B", help="Hugging Face model id or local VGGT model dir.")
    parser.add_argument("--preprocess-mode", choices=("pad", "crop"), default="pad")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit the number of frames. 0 means no limit.")
    parser.add_argument("--query-grid-size", type=int, default=16, help="Track an NxN query grid. 0 disables tracks.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--save-float16",
        action="store_true",
        help="Save dense maps as float16 to reduce disk usage. Cameras remain float32 JSON.",
    )
    parser.add_argument(
        "--no-preview-png",
        action="store_true",
        help="Do not export human-readable depth/normal/confidence PNG previews.",
    )
    return parser.parse_args()


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
        return video_path.parent / f"{video_path.stem}_vggt"
    if frames_dir is None:
        raise ValueError("Either frames_dir or video_path is required")
    return frames_dir.parent / "vggt"


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


def frame_sizes(frames: list[Path]) -> list[tuple[int, int]]:
    sizes = []
    for path in frames:
        with Image.open(path) as image:
            sizes.append(image.size)
    return sizes


def add_vggt_to_path(vggt_repo: str) -> None:
    repo = Path(vggt_repo)
    if repo.exists():
        sys.path.insert(0, str(repo))


def choose_device(raw_device: str | None) -> torch.device:
    if raw_device:
        return torch.device(raw_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def make_query_grid(height: int, width: int, grid_size: int, device: torch.device) -> torch.Tensor | None:
    if grid_size <= 0:
        return None
    xs = torch.linspace(0, width - 1, grid_size, device=device)
    ys = torch.linspace(0, height - 1, grid_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def compute_normals_from_points(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Estimate per-pixel normals from a point map shaped [S, H, W, 3]."""
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, :, 1:-1] = points[:, :, 2:] - points[:, :, :-2]
    dx[:, :, 0] = points[:, :, 1] - points[:, :, 0]
    dx[:, :, -1] = points[:, :, -1] - points[:, :, -2]
    dy[:, 1:-1, :] = points[:, 2:, :] - points[:, :-2, :]
    dy[:, 0, :] = points[:, 1, :] - points[:, 0, :]
    dy[:, -1, :] = points[:, -1, :] - points[:, -2, :]
    normals = np.cross(dx, dy)
    denom = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(denom, eps)
    normals[~np.isfinite(normals)] = 0.0
    return normals


def to_numpy(value: torch.Tensor | np.ndarray, save_float16: bool = False) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value.astype(np.float32, copy=False)
    else:
        array = value.detach().cpu().float().numpy()
    if save_float16:
        return array.astype(np.float16)
    return array.astype(np.float32)


def write_dense_sequence(sequence: np.ndarray, names: list[str], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_paths = []
    for idx, name in enumerate(names):
        out_path = out_dir / f"{Path(name).stem}.npy"
        np.save(out_path, sequence[idx])
        rel_paths.append(str(out_path))
    return rel_paths


def normalize_scalar_image(values: np.ndarray, percentile_low: float = 2.0, percentile_high: float = 98.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(values[finite], [percentile_low, percentile_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values[finite].min())
        hi = float(values[finite].max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = (values - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~finite] = 0.0
    return (normalized * 255.0).astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    gray = normalize_scalar_image(depth)
    x = gray.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def normal_to_rgb(normals: np.ndarray) -> np.ndarray:
    rgb = (np.asarray(normals, dtype=np.float32) * 0.5 + 0.5) * 255.0
    rgb[~np.isfinite(rgb)] = 0.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def write_png_sequence(
    sequence: np.ndarray,
    names: list[str],
    out_dir: Path,
    kind: str,
    target_sizes: list[tuple[int, int]] | None = None,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, name in enumerate(names):
        value = sequence[idx]
        if kind == "normal":
            image = normal_to_rgb(value)
        elif kind == "depth":
            image = colorize_depth(value)
        else:
            image = normalize_scalar_image(value)
        out_path = out_dir / f"{Path(name).stem}.png"
        pil_image = Image.fromarray(image)
        if target_sizes is not None and idx < len(target_sizes) and pil_image.size != target_sizes[idx]:
            pil_image = pil_image.resize(target_sizes[idx], Image.Resampling.BILINEAR)
        pil_image.save(out_path)
        paths.append(str(out_path))
    return paths


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).resolve() if args.video_path else None
    frames_dir_arg = Path(args.frames_dir).resolve() if args.frames_dir else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(frames_dir_arg, video_path).resolve()
    frames_dir = extract_video_frames(video_path, output_dir) if video_path is not None else frames_dir_arg
    assert frames_dir is not None
    frames = list_frames(frames_dir, args.stride, args.max_frames)
    frame_names = [p.name for p in frames]
    source_frame_sizes = frame_sizes(frames)

    add_vggt_to_path(args.vggt_repo)
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = choose_device(args.device)
    dtype = choose_dtype(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading VGGT model [{args.model_id}] on [{device}]")
    model = VGGT.from_pretrained(args.model_id).to(device)
    model.eval()

    print(f"Loading {len(frames)} frame(s) from {frames_dir}")
    images = load_and_preprocess_images([str(p) for p in frames], mode=args.preprocess_mode).to(device)
    height, width = int(images.shape[-2]), int(images.shape[-1])
    query_points = make_query_grid(height, width, args.query_grid_size, device)

    print("Running VGGT")
    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images, query_points=query_points)
        else:
            predictions = model(images, query_points=query_points)

    pose_enc = predictions["pose_enc"]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    point_map_by_depth = unproject_depth_map_to_point_map(
        predictions["depth"].squeeze(0),
        extrinsic.squeeze(0),
        intrinsic.squeeze(0),
    )

    depth = to_numpy(predictions["depth"].squeeze(0).squeeze(-1), args.save_float16)
    depth_conf = to_numpy(predictions["depth_conf"].squeeze(0), args.save_float16)
    point_maps = to_numpy(predictions["world_points"].squeeze(0), args.save_float16)
    point_conf = to_numpy(predictions["world_points_conf"].squeeze(0), args.save_float16)
    unprojected_points = to_numpy(point_map_by_depth, args.save_float16)
    normals = compute_normals_from_points(point_maps.astype(np.float32))
    if args.save_float16:
        normals = normals.astype(np.float16)

    outputs = {
        "depth": write_dense_sequence(depth, frame_names, output_dir / "depth"),
        "depth_conf": write_dense_sequence(depth_conf, frame_names, output_dir / "depth_conf"),
        "point_maps": write_dense_sequence(point_maps, frame_names, output_dir / "point_maps"),
        "point_maps_by_depth": write_dense_sequence(unprojected_points, frame_names, output_dir / "point_maps_by_depth"),
        "point_conf": write_dense_sequence(point_conf, frame_names, output_dir / "point_conf"),
        "normals": write_dense_sequence(normals, frame_names, output_dir / "normals"),
    }
    preview_outputs = {}
    if not args.no_preview_png:
        preview_outputs = {
            "depth": write_png_sequence(depth, frame_names, output_dir / "preview_png" / "depth", "depth", source_frame_sizes),
            "depth_conf": write_png_sequence(depth_conf, frame_names, output_dir / "preview_png" / "depth_conf", "scalar", source_frame_sizes),
            "point_conf": write_png_sequence(point_conf, frame_names, output_dir / "preview_png" / "point_conf", "scalar", source_frame_sizes),
            "normals": write_png_sequence(normals, frame_names, output_dir / "preview_png" / "normals", "normal", source_frame_sizes),
        }

    camera_path = output_dir / "cameras.json"
    camera_data = {
        "model_id": args.model_id,
        "video_path": str(video_path) if video_path else None,
        "frames_dir": str(frames_dir),
        "preprocess_mode": args.preprocess_mode,
        "processed_image_size": [height, width],
        "source_frame_sizes": [[h, w] for w, h in source_frame_sizes],
        "stride": args.stride,
        "frames": frame_names,
        "pose_encoding": to_numpy(pose_enc.squeeze(0)).tolist(),
        "extrinsics_camera_from_world": to_numpy(extrinsic.squeeze(0)).tolist(),
        "intrinsics": to_numpy(intrinsic.squeeze(0)).tolist(),
    }
    camera_path.write_text(json.dumps(camera_data, indent=2))

    tracks_path = None
    if query_points is not None and "track" in predictions:
        tracks_path = output_dir / "point_tracks.npz"
        np.savez_compressed(
            tracks_path,
            query_points=to_numpy(query_points),
            tracks=to_numpy(predictions["track"].squeeze(0)),
            visibility=to_numpy(predictions["vis"].squeeze(0)),
            confidence=to_numpy(predictions["conf"].squeeze(0)),
            frames=np.array(frame_names),
        )

    summary = {
        "method": "VGGT",
        "model_id": args.model_id,
        "video_path": str(video_path) if video_path else None,
        "frames_dir": str(frames_dir),
        "frames": frame_names,
        "num_frames": len(frame_names),
        "output_dir": str(output_dir),
        "processed_image_size": [height, width],
        "source_frame_sizes": [[h, w] for w, h in source_frame_sizes],
        "cameras": str(camera_path),
        "point_tracks": str(tracks_path) if tracks_path else None,
        "outputs": outputs,
        "preview_png": preview_outputs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved VGGT preprocessing outputs to {output_dir}")


if __name__ == "__main__":
    main()

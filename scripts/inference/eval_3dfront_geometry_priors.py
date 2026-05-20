#!/usr/bin/env python3
"""Compare geometry-prior preprocessors against selected 3D-FRONT render GT.

The script prepares identical input frames from 3D-FRONT render folders, runs
HY-World, VGGT, and/or GeometryCrafter preprocessors, then evaluates available
outputs against 3D-FRONT depth, normal, and camera metadata.

Camera metrics are reported for methods that save camera parameters
(HY-World, VGGT). GeometryCrafter currently contributes depth/normal metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image


DEFAULT_CASES = [
    "/mnt/mocap_b/work/com4d/datasets/raw/3D-FRONT/data/3D-FRONT-RENDER/0a8d471a-2587-458a-9214-586e003e9cf9",
    "/mnt/mocap_b/work/com4d/datasets/raw/3D-FRONT/data/3D-FRONT-RENDER/0a9c667d-033d-448c-b17c-dc55e6d3c386/DiningRoom-11628",
    "/mnt/mocap_b/work/com4d/datasets/raw/3D-FRONT/data/3D-FRONT-RENDER/0a25c251-7c80-4808-b609-3d6fbae9efad/MasterBedroom-2888",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HY-World/VGGT/GeometryCrafter on 3D-FRONT render GT.")
    parser.add_argument("--case", action="append", dest="cases", default=None, help="3D-FRONT room render dir or house dir.")
    parser.add_argument("--output-root", default="/data/mseizde/com4d/COM4D/assets/3dfront_geometry_prior_eval")
    parser.add_argument("--methods", nargs="+", default=["hyworld", "vggt", "geometrycrafter"], choices=["hyworld", "vggt", "geometrycrafter"])
    parser.add_argument(
        "--image-kind",
        choices=["render", "rerender", "auto"],
        default="render",
        help="Backward-compatible alias for --input-image-kind.",
    )
    parser.add_argument(
        "--input-image-kind",
        choices=["render", "rerender", "auto"],
        default=None,
        help="Images passed to the geometry prior. Defaults to --image-kind.",
    )
    parser.add_argument(
        "--gt-image-kind",
        choices=["render"],
        default="render",
        help="GT depth/normal/semantic/camera files to compare against. 3D-FRONT GT is currently render-based.",
    )
    parser.add_argument(
        "--eval-mask",
        choices=["semantic", "alpha", "all"],
        default="semantic",
        help="Pixel mask for depth/normal metrics. semantic keeps semantic>0 object pixels; alpha uses normal alpha; all uses every finite GT/pred pixel.",
    )
    parser.add_argument("--view-index", type=int, default=None, help="Evaluate one view instead of every available view.")
    parser.add_argument("--max-views", type=int, default=0, help="Limit number of selected views. 0 means no limit.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth selected view.")
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Only evaluate existing method outputs.")
    parser.add_argument("--hyworld-env", default="hyworld2")
    parser.add_argument("--vggt-env", default="vggt")
    parser.add_argument("--geometrycrafter-env", default="geometrycrafter")
    parser.add_argument("--hyworld-script", default="/data/mseizde/com4d/COM4D/scripts/inference/run_hyworld_preprocess.py")
    parser.add_argument("--vggt-script", default="/data/mseizde/com4d/COM4D/scripts/inference/run_vggt_preprocess.py")
    parser.add_argument("--geometrycrafter-script", default="/data/mseizde/com4d/COM4D/scripts/inference/run_geometrycrafter_preprocess.py")
    parser.add_argument("--hyworld-target-size", type=int, default=952)
    parser.add_argument("--vggt-max-frames", type=int, default=0)
    parser.add_argument("--geometrycrafter-height", type=int, default=384)
    parser.add_argument("--geometrycrafter-width", type=int, default=384)
    parser.add_argument("--geometrycrafter-max-frames", type=int, default=0)
    return parser.parse_args()


def resolve_room_dir(path: Path) -> Path:
    if (path / "meta.json").is_file():
        return path
    rooms = sorted(p for p in path.iterdir() if p.is_dir() and (p / "meta.json").is_file())
    if len(rooms) != 1:
        raise ValueError(f"Expected one room with meta.json under {path}, found {len(rooms)}")
    return rooms[0]


def case_name(room_dir: Path) -> str:
    return f"{room_dir.parent.name}_{room_dir.name}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rgba_to_rgb_on_white(src: Path, dst: Path) -> None:
    image = Image.open(src)
    if getattr(image, "is_animated", False):
        image.seek(0)
    if image.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image.convert("RGBA")).convert("RGB")
    else:
        image = image.convert("RGB")
    image.save(dst)


def select_source_images(room_dir: Path, image_kind: str, view_index: int | None, stride: int, max_views: int) -> list[Path]:
    kinds = [image_kind] if image_kind != "auto" else ["render", "rerender"]
    selected: list[Path] = []
    for kind in kinds:
        selected = sorted(room_dir.glob(f"{kind}_*.webp")) + sorted(room_dir.glob(f"{kind}_*.png"))
        if selected:
            break
    if not selected:
        raise ValueError(f"No render/rerender images found in {room_dir}")
    if view_index is not None:
        selected = [p for p in selected if p.stem.endswith(f"_{view_index:04d}")]
        if not selected:
            raise ValueError(f"No source image for view {view_index:04d} in {room_dir}")
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    selected = selected[::stride]
    if max_views > 0:
        selected = selected[:max_views]
    return selected


def prepare_case_frames(room_dir: Path, case_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    input_kind = args.input_image_kind or args.image_kind
    sources = select_source_images(room_dir, input_kind, args.view_index, args.stride, args.max_views)
    frames_dir = case_dir / "input_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    records = []
    for idx, src in enumerate(sources, start=1):
        dst = frames_dir / f"{idx:06d}.png"
        rgba_to_rgb_on_white(src, dst)
        suffix = src.stem.split("_")[-1]
        records.append(
            {
                "input_frame": dst.name,
                "source_name": src.name,
                "input_image_kind": src.stem.rsplit("_", 1)[0],
                "source_image": str(src),
                "view_index": int(suffix),
                "gt_source_name": f"{args.gt_image_kind}_{suffix}.webp",
                "depth": str(room_dir / f"depth_{suffix}.exr"),
                "normal": str(room_dir / f"normal_{suffix}.webp"),
                "semantic": str(room_dir / f"semantic_{suffix}.png"),
            }
        )
    (case_dir / "frame_mapping.json").write_text(json.dumps(records, indent=2))
    return records


def run_command(command: list[str], env_name: str, cuda_visible_devices: str, cwd: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    full = ["micromamba", "run", "-n", env_name] + command
    subprocess.run(full, cwd=str(cwd), check=True, env=env)


def maybe_run_method(method: str, case_dir: Path, args: argparse.Namespace) -> None:
    if args.skip_run:
        return
    frames_dir = case_dir / "input_frames"
    out_dir = case_dir / method
    if not args.force_run and method_outputs_ready(method, out_dir):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path("/data/mseizde/com4d/COM4D")
    if method == "hyworld":
        run_command(
            [
                "python",
                args.hyworld_script,
                "--frames-dir",
                str(frames_dir),
                "--output-dir",
                str(out_dir),
                "--target-size",
                str(args.hyworld_target_size),
                "--save-conf",
            ],
            args.hyworld_env,
            args.cuda_visible_devices,
            cwd,
        )
    elif method == "vggt":
        cmd = ["python", args.vggt_script, "--frames-dir", str(frames_dir), "--output-dir", str(out_dir)]
        if args.vggt_max_frames > 0:
            cmd += ["--max-frames", str(args.vggt_max_frames)]
        run_command(cmd, args.vggt_env, args.cuda_visible_devices, cwd)
    elif method == "geometrycrafter":
        cmd = [
            "python",
            args.geometrycrafter_script,
            "--frames-dir",
            str(frames_dir),
            "--output-dir",
            str(out_dir),
            "--height",
            str(args.geometrycrafter_height),
            "--width",
            str(args.geometrycrafter_width),
        ]
        if args.geometrycrafter_max_frames > 0:
            cmd += ["--max-frames", str(args.geometrycrafter_max_frames)]
        run_command(cmd, args.geometrycrafter_env, args.cuda_visible_devices, cwd)


def method_outputs_ready(method: str, out_dir: Path) -> bool:
    if method == "hyworld":
        return (out_dir / "depth" / "depth_0000.npy").is_file() and (out_dir / "normal" / "normal_0000.png").is_file()
    if method == "vggt":
        return any((out_dir / "depth").glob("*.npy")) and any((out_dir / "normals").glob("*.npy"))
    if method == "geometrycrafter":
        return any((out_dir / "depth").glob("*.npy")) and any((out_dir / "normals").glob("*.npy"))
    return False


def read_depth_exr(path: str) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"Could not read EXR depth: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def read_gt_normal(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32)
    normal = arr[..., :3] / 255.0 * 2.0 - 1.0
    normal = normalize_normals(normal)
    return normal, arr[..., 3] > 16.0


def read_semantic_mask(path: str) -> np.ndarray | None:
    if not Path(path).exists():
        return None
    return np.asarray(Image.open(path)) > 0


def build_eval_mask(record: dict[str, Any], shape_hw: tuple[int, int], eval_mask: str) -> np.ndarray:
    if eval_mask == "all":
        return np.ones(shape_hw, dtype=bool)
    _, alpha = read_gt_normal(record["normal"])
    mask = resize_array(alpha.astype(np.uint8), shape_hw, cv2.INTER_NEAREST).astype(bool)
    if eval_mask == "semantic":
        sem = read_semantic_mask(record["semantic"])
        if sem is not None:
            mask &= resize_array(sem.astype(np.uint8), shape_hw, cv2.INTER_NEAREST).astype(bool)
    return mask


def resize_array(arr: np.ndarray, shape_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    h, w = shape_hw
    return cv2.resize(arr, (w, h), interpolation=interpolation)


def normalize_normals(normals: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(normals, axis=-1, keepdims=True)
    out = normals / np.maximum(denom, 1e-8)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def pred_depth_path(method: str, out_dir: Path, idx: int, frame_name: str) -> Path:
    if method == "hyworld":
        return out_dir / "depth" / f"depth_{idx:04d}.npy"
    return out_dir / "depth" / f"{Path(frame_name).stem}.npy"


def pred_normal_array(method: str, out_dir: Path, idx: int, frame_name: str) -> np.ndarray:
    if method == "hyworld":
        png = out_dir / "normal" / f"normal_{idx:04d}.png"
        return np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0 * 2.0 - 1.0
    npy = out_dir / "normals" / f"{Path(frame_name).stem}.npy"
    return np.load(npy).astype(np.float32)


def depth_metrics(pred: np.ndarray, record: dict[str, Any], shape_hw: tuple[int, int], eval_mask: str) -> dict[str, float]:
    gt = resize_array(read_depth_exr(record["depth"]), shape_hw, cv2.INTER_LINEAR)
    mask = build_eval_mask(record, shape_hw, eval_mask)
    valid = mask & np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    if not valid.any():
        return {"depth_valid_px": 0}
    p = pred[valid].reshape(-1).astype(np.float64)
    g = gt[valid].reshape(-1).astype(np.float64)
    raw_abs_rel = np.mean(np.abs(p - g) / np.maximum(g, 1e-8))
    scale = np.median(g) / max(np.median(p), 1e-8)
    ps = p * scale
    a, b = np.linalg.lstsq(np.stack([p, np.ones_like(p)], axis=1), g, rcond=None)[0]
    pa = p * a + b
    return {
        "depth_valid_px": int(valid.sum()),
        "depth_raw_abs_rel": float(raw_abs_rel),
        "depth_scale": float(scale),
        "depth_scaled_mae": float(np.mean(np.abs(ps - g))),
        "depth_scaled_rmse": float(np.sqrt(np.mean((ps - g) ** 2))),
        "depth_affine_scale": float(a),
        "depth_affine_shift": float(b),
        "depth_affine_mae": float(np.mean(np.abs(pa - g))),
        "depth_affine_rmse": float(np.sqrt(np.mean((pa - g) ** 2))),
    }


def normal_metrics(pred: np.ndarray, record: dict[str, Any], shape_hw: tuple[int, int], eval_mask: str) -> dict[str, Any]:
    pred = normalize_normals(pred)
    gt, alpha = read_gt_normal(record["normal"])
    gt = normalize_normals(resize_array(gt, shape_hw, cv2.INTER_LINEAR))
    if eval_mask == "all":
        mask = np.ones(shape_hw, dtype=bool)
    else:
        mask = resize_array(alpha.astype(np.uint8), shape_hw, cv2.INTER_NEAREST).astype(bool)
        if eval_mask == "semantic":
            sem = read_semantic_mask(record["semantic"])
            if sem is not None:
                mask &= resize_array(sem.astype(np.uint8), shape_hw, cv2.INTER_NEAREST).astype(bool)
    mask &= np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    if not mask.any():
        return {"normal_valid_px": 0}
    best_name = ""
    best_angles = None
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                name = f"sign_{int(sx)}_{int(sy)}_{int(sz)}"
                cand = pred * np.array([sx, sy, sz], dtype=np.float32)
                dot = np.sum(cand[mask] * gt[mask], axis=-1)
                angles = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
                if best_angles is None or float(np.mean(angles)) < float(np.mean(best_angles)):
                    best_name = name
                    best_angles = angles
    assert best_angles is not None
    return {
        "normal_valid_px": int(mask.sum()),
        "normal_mean_deg": float(np.mean(best_angles)),
        "normal_median_deg": float(np.median(best_angles)),
        "normal_rmse_deg": float(np.sqrt(np.mean(best_angles**2))),
        "normal_pct_lt_11_25": float(np.mean(best_angles < 11.25)),
        "normal_pct_lt_22_5": float(np.mean(best_angles < 22.5)),
        "normal_best_sign": best_name,
    }


def load_gt_c2w(meta: dict[str, Any], records: list[dict[str, Any]]) -> np.ndarray:
    by_name = {}
    for loc in meta.get("locations", []):
        for frame in loc.get("frames", []):
            by_name[frame.get("name")] = loc
    mats = []
    for record in records:
        loc = by_name.get(record.get("gt_source_name", record["source_name"]))
        if loc is None:
            raise KeyError(f"No camera metadata for {record.get('gt_source_name', record['source_name'])}")
        mats.append(np.asarray(loc["transform_matrix"], dtype=np.float64))
    return np.stack(mats, axis=0)


def intrinsic_from_meta(meta: dict[str, Any], width: int, height: int) -> np.ndarray:
    angle_x = float(meta["camera_angle_x"])
    fx = 0.5 * width / math.tan(0.5 * angle_x)
    sensor_width = float(meta.get("sensor_width", 36.0))
    lens = float(meta.get("camera_lens", 50.0))
    fy = lens / sensor_width * width
    if abs(width - height) < 1:
        fy = fx
    return np.array([[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_angle_deg(rot: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))))


def load_pred_cameras(method: str, out_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if method == "hyworld":
        path = out_dir / "camera_params.json"
        if not path.is_file():
            return None
        data = load_json(path)
        extr = np.asarray([item["matrix"] for item in data["extrinsics"]], dtype=np.float64)
        intr = np.asarray([item["matrix"] for item in data["intrinsics"]], dtype=np.float64)
        return extr, intr
    if method == "vggt":
        path = out_dir / "cameras.json"
        if not path.is_file():
            return None
        data = load_json(path)
        extr = np.asarray(data["extrinsics_camera_from_world"], dtype=np.float64)
        intr = np.asarray(data["intrinsics"], dtype=np.float64)
        mats = np.tile(np.eye(4, dtype=np.float64), (extr.shape[0], 1, 1))
        mats[:, :3, :4] = extr[:, :3, :4]
        return mats, intr
    return None


def camera_metrics(method: str, out_dir: Path, meta: dict[str, Any], records: list[dict[str, Any]], shape_hw: tuple[int, int]) -> dict[str, float] | None:
    pred = load_pred_cameras(method, out_dir)
    if pred is None:
        return None
    pred_w2c, pred_k = pred
    n = min(len(pred_w2c), len(records))
    pred_c2w = np.linalg.inv(pred_w2c[:n])
    gt_c2w = load_gt_c2w(meta, records[:n])
    gt_k = intrinsic_from_meta(meta, shape_hw[1], shape_hw[0])
    intr = pred_k[:n]
    out = {
        "camera_count": int(n),
        "intrinsics_fx_rel_err_mean": float(np.mean(np.abs(intr[:, 0, 0] - gt_k[0, 0]) / max(abs(gt_k[0, 0]), 1e-8))),
        "intrinsics_fy_rel_err_mean": float(np.mean(np.abs(intr[:, 1, 1] - gt_k[1, 1]) / max(abs(gt_k[1, 1]), 1e-8))),
        "intrinsics_cx_rel_err_mean": float(np.mean(np.abs(intr[:, 0, 2] - gt_k[0, 2]) / max(abs(gt_k[0, 2]), 1e-8))),
        "intrinsics_cy_rel_err_mean": float(np.mean(np.abs(intr[:, 1, 2] - gt_k[1, 2]) / max(abs(gt_k[1, 2]), 1e-8))),
    }
    if n < 2:
        return out
    pred_rot, gt_rot, pred_dist, gt_dist = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            pred_rot.append(rotation_angle_deg(pred_c2w[i, :3, :3].T @ pred_c2w[j, :3, :3]))
            gt_rot.append(rotation_angle_deg(gt_c2w[i, :3, :3].T @ gt_c2w[j, :3, :3]))
            pred_dist.append(np.linalg.norm(pred_c2w[i, :3, 3] - pred_c2w[j, :3, 3]))
            gt_dist.append(np.linalg.norm(gt_c2w[i, :3, 3] - gt_c2w[j, :3, 3]))
    pred_rot = np.asarray(pred_rot)
    gt_rot = np.asarray(gt_rot)
    pred_dist = np.asarray(pred_dist)
    gt_dist = np.asarray(gt_dist)
    valid = (pred_dist > 1e-8) & (gt_dist > 1e-8)
    if valid.any():
        scale = np.median(gt_dist[valid]) / max(np.median(pred_dist[valid]), 1e-8)
        dist_err = np.abs(pred_dist[valid] * scale - gt_dist[valid])
        out.update(
            {
                "pairwise_rotation_mae_deg": float(np.mean(np.abs(pred_rot - gt_rot))),
                "pairwise_rotation_rmse_deg": float(np.sqrt(np.mean((pred_rot - gt_rot) ** 2))),
                "pairwise_distance_scale": float(scale),
                "pairwise_distance_mae": float(np.mean(dist_err)),
                "pairwise_distance_rmse": float(np.sqrt(np.mean(dist_err**2))),
            }
        )
    return out


def evaluate_method(method: str, case_dir: Path, room_dir: Path, records: list[dict[str, Any]], eval_mask: str) -> dict[str, Any]:
    out_dir = case_dir / method
    per_frame = []
    first_depth = np.load(pred_depth_path(method, out_dir, 0, records[0]["input_frame"])).astype(np.float32)
    shape_hw = tuple(first_depth.shape[:2])
    for idx, record in enumerate(records):
        depth = np.load(pred_depth_path(method, out_dir, idx, record["input_frame"])).astype(np.float32)
        normal = pred_normal_array(method, out_dir, idx, record["input_frame"])
        if depth.shape[:2] != shape_hw:
            depth = resize_array(depth, shape_hw, cv2.INTER_LINEAR)
        if normal.shape[:2] != shape_hw:
            normal = resize_array(normal, shape_hw, cv2.INTER_LINEAR)
        row: dict[str, Any] = {
            "frame": record["input_frame"],
            "source": record["source_name"],
            "gt_source": record.get("gt_source_name", record["source_name"]),
            "view_index": record["view_index"],
        }
        row.update(depth_metrics(depth, record, shape_hw, eval_mask))
        row.update(normal_metrics(normal, record, shape_hw, eval_mask))
        per_frame.append(row)
    meta = load_json(room_dir / "meta.json")
    cam = camera_metrics(method, out_dir, meta, records, shape_hw)
    aggregate = aggregate_rows(per_frame)
    aggregate.update({"method": method, "case": case_dir.name, "num_frames": len(records), "eval_mask": eval_mask})
    if cam is not None:
        aggregate["camera"] = cam
    result = {"method": method, "case": case_dir.name, "output_dir": str(out_dir), "aggregate": aggregate, "per_frame": per_frame}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = sorted(k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and k != "view_index")
    for key in keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and np.isfinite(row[key])]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_median"] = float(np.median(vals))
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = [resolve_room_dir(Path(p).resolve()) for p in (args.cases or DEFAULT_CASES)]
    all_results = []
    for room_dir in cases:
        cdir = output_root / case_name(room_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        records = prepare_case_frames(room_dir, cdir, args)
        case_results = []
        for method in args.methods:
            maybe_run_method(method, cdir, args)
            case_results.append(evaluate_method(method, cdir, room_dir, records, args.eval_mask))
        case_summary = {
            "case": cdir.name,
            "room_dir": str(room_dir),
            "input_image_kind": args.input_image_kind or args.image_kind,
            "gt_image_kind": args.gt_image_kind,
            "eval_mask": args.eval_mask,
            "records": records,
            "methods": case_results,
        }
        (cdir / "metrics_summary.json").write_text(json.dumps(case_summary, indent=2))
        all_results.append(case_summary)
    summary = {
        "output_root": str(output_root),
        "input_image_kind": args.input_image_kind or args.image_kind,
        "gt_image_kind": args.gt_image_kind,
        "eval_mask": args.eval_mask,
        "cases": all_results,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved 3D-FRONT geometry-prior evaluation to {output_root}")


if __name__ == "__main__":
    main()

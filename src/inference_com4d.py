import argparse
import inspect
import json
import os
import re
import shutil
import sys
from glob import glob
import time
from pathlib import Path
from typing import Any, List, Optional, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PIL
import numpy as np
import torch
import torch.nn as nn
import trimesh
from accelerate.utils import set_seed
from PIL import Image

from src.utils.inference import (_gather_images, _gather_all_masks, _combine_masks, _resolve_repo_or_dir, _parse_id_string, build_pipeline, _invert_mask)
from src.models.transformers import PartFrameCrafterDiTModel
from src.pipelines.pipeline_partcrafter import (
    PartCrafter3D4DInferencePipeline,
)
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import export_renderings, render_sequence_fixed_camera, render_views_around_mesh


class RoomLayoutAuxiliaryHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 256, out_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, latent_tokens: torch.Tensor, num_parts: torch.Tensor) -> torch.Tensor:
        pooled_objects = []
        ptr = 0
        for count_tensor in num_parts.detach().cpu():
            count = int(count_tensor.item())
            if count <= 0:
                continue
            pooled_objects.append(latent_tokens[ptr:ptr + count].float().mean(dim=(0, 1)))
            ptr += count
        if not pooled_objects:
            return latent_tokens.new_zeros((0, 12), dtype=torch.float32)
        return self.net(torch.stack(pooled_objects, dim=0))


def seed_all(seed: int):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resize_rgb_image(image: Image.Image, image_size: int) -> Image.Image:
    target_size = (int(image_size), int(image_size))
    if image.size == target_size:
        return image
    return image.resize(target_size, Image.LANCZOS)


def _resize_mask_image(mask: Image.Image, image_size: int) -> Image.Image:
    target_size = (int(image_size), int(image_size))
    if mask.size == target_size:
        return mask
    return mask.resize(target_size, Image.NEAREST)


def _discover_dynamic_object_ids(masks_dir: Optional[Path], count: int) -> List[int]:
    if count <= 0:
        return []

    discovered_ids: List[int] = []
    if masks_dir is not None and masks_dir.is_dir():
        pattern = re.compile(r"object_(\d+)\.[^.]+$")
        unique_ids = set()
        for path in sorted(masks_dir.iterdir(), key=lambda p: p.name):
            if not path.is_file():
                continue
            match = pattern.search(path.name)
            if match:
                unique_ids.add(int(match.group(1)))
        discovered_ids = sorted(unique_ids)

    if len(discovered_ids) >= count:
        return discovered_ids[:count]

    used_ids = set(discovered_ids)
    next_id = 1
    while len(discovered_ids) < count:
        while next_id in used_ids:
            next_id += 1
        discovered_ids.append(next_id)
        used_ids.add(next_id)
        next_id += 1
    return discovered_ids


def _load_as_scene(mesh_or_scene) -> trimesh.Scene:
    if mesh_or_scene is None:
        return trimesh.Scene()
    if isinstance(mesh_or_scene, trimesh.Scene):
        return mesh_or_scene.copy()
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return trimesh.Scene(mesh_or_scene.copy())
    return trimesh.Scene(mesh_or_scene)


def _scene_bounds_from_meshes(meshes: List) -> Optional[np.ndarray]:
    scenes = [_load_as_scene(mesh) for mesh in meshes if mesh is not None]
    bounds = []
    for scene in scenes:
        raw = np.asarray(scene.bounds, dtype=np.float64)
        if raw.shape == (2, 3) and np.isfinite(raw).all():
            bounds.append(raw)
    if not bounds:
        return None
    mins = np.stack([b[0] for b in bounds], axis=0).min(axis=0)
    maxs = np.stack([b[1] for b in bounds], axis=0).max(axis=0)
    return np.stack([mins, maxs], axis=0)


def _set_mesh_rgba(mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.asarray(rgba, dtype=np.uint8),
    )
    return mesh


def _make_box_room_shell(
    center: np.ndarray,
    extent: np.ndarray,
    presence: np.ndarray,
    alpha: float,
    wall_thickness: float = 0.025,
) -> trimesh.Scene:
    alpha_i = int(round(np.clip(alpha, 0.0, 1.0) * 255))
    center = np.asarray(center, dtype=np.float64)
    extent = np.maximum(np.asarray(extent, dtype=np.float64), 1e-4)
    half = extent * 0.5
    min_corner = center - half
    max_corner = center + half
    thickness = max(float(wall_thickness), 1e-4)

    scene = trimesh.Scene()

    def add_box(name: str, box_center: list[float], box_extents: list[float], rgba: tuple[int, int, int, int]) -> None:
        mesh = trimesh.creation.box(extents=np.maximum(np.asarray(box_extents, dtype=np.float64), 1e-4))
        mesh.apply_translation(np.asarray(box_center, dtype=np.float64))
        scene.add_geometry(_set_mesh_rgba(mesh, rgba), geom_name=name)

    x_size, y_size, z_size = extent.tolist()
    floor_y = float(min_corner[1])
    ceil_y = float(max_corner[1])
    mid_y = float(center[1])

    if presence[0]:
        add_box(
            "pred_room_floor",
            [center[0], floor_y - thickness * 0.5, center[2]],
            [x_size, thickness, z_size],
            (80, 190, 130, alpha_i),
        )
    if presence[1]:
        add_box(
            "pred_room_ceiling",
            [center[0], ceil_y + thickness * 0.5, center[2]],
            [x_size, thickness, z_size],
            (230, 190, 80, alpha_i),
        )

    wall_rgba = (80, 145, 230, alpha_i)
    if presence[2]:
        add_box("pred_room_front_wall", [center[0], mid_y, min_corner[2] - thickness * 0.5], [x_size, y_size, thickness], wall_rgba)
    if presence[3]:
        add_box("pred_room_back_wall", [center[0], mid_y, max_corner[2] + thickness * 0.5], [x_size, y_size, thickness], wall_rgba)
    if presence[4]:
        add_box("pred_room_left_wall", [min_corner[0] - thickness * 0.5, mid_y, center[2]], [thickness, y_size, z_size], wall_rgba)
    if presence[5]:
        add_box("pred_room_right_wall", [max_corner[0] + thickness * 0.5, mid_y, center[2]], [thickness, y_size, z_size], wall_rgba)

    return scene


def _add_scene_geometries(target: trimesh.Scene, source: trimesh.Scene, prefix: str) -> None:
    for idx, geom in enumerate(source.dump(concatenate=False)):
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
            target.add_geometry(geom, geom_name=f"{prefix}_{idx:03d}")


def _combine_with_room_shell(base_meshes: List, room_shell: trimesh.Scene) -> trimesh.Scene:
    combined = trimesh.Scene()
    for idx, mesh in enumerate(base_meshes):
        if mesh is None:
            continue
        _add_scene_geometries(combined, _load_as_scene(mesh), prefix=f"generated_{idx:03d}")
    _add_scene_geometries(combined, room_shell, prefix="predicted_room")
    return combined


def _render_room_scene_gif(scene: trimesh.Scene, path: Path, args: argparse.Namespace) -> None:
    frames = render_views_around_mesh(
        scene,
        num_views=int(args.predicted_room_num_views),
        radius=float(args.predicted_room_camera_radius),
        image_size=(int(args.render_size), int(args.render_size)),
        light_intensity=5.0,
        bg_color=(255, 255, 255, 255),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    export_renderings(frames, str(path), fps=int(args.animation_fps))


def _render_room_animation_gif(scenes: List[trimesh.Scene], path: Path, args: argparse.Namespace) -> None:
    if not scenes:
        return
    frames = render_sequence_fixed_camera(
        scenes,
        image_size=(int(args.render_size), int(args.render_size)),
        light_intensity=5.0,
        return_type="pil",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    export_renderings(frames, str(path), fps=int(args.animation_fps))


def _to_pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            array = (
                tensor.float()
                .clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .numpy()
            )
            return Image.fromarray(array).convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.dtype != np.uint8:
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        if array.shape[-1] == 4:
            array = array[..., :3]
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported image type for conversion: {type(image)!r}")


def _compose_side_by_side(left: Image.Image, right: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    if left.size != target_size:
        left = left.resize(target_size, Image.LANCZOS)
    if right.size != target_size:
        right = right.resize(target_size, Image.LANCZOS)
    canvas = Image.new("RGB", (target_size[0] * 2, target_size[1]), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (target_size[0], 0))
    return canvas


def _render_room_augmented_animations(
    scenes: List[trimesh.Scene],
    gt_frames: List[Image.Image],
    export_dir: Path,
    args: argparse.Namespace,
) -> None:
    if not scenes:
        return
    target_size = (int(args.render_size), int(args.render_size))
    render_kwargs = {
        "image_size": target_size,
        "light_intensity": 5.0,
        "return_type": "pil",
        "bg_color": (255, 255, 255, 255),
    }
    scene_frames = render_sequence_fixed_camera(scenes, **render_kwargs)
    scene_images = [_to_pil_rgb(frame).resize(target_size, Image.LANCZOS) for frame in scene_frames]

    scene_output_frames: List[Image.Image] = list(scene_images)
    comparison_frames: List[Image.Image] = []
    last_gt: Optional[Image.Image] = None

    for idx, scene_img in enumerate(scene_images):
        if idx < len(gt_frames):
            last_gt = _to_pil_rgb(gt_frames[idx]).resize(target_size, Image.LANCZOS)
        if last_gt is None:
            last_gt = scene_img
        comparison_frames.append(_compose_side_by_side(last_gt, scene_img, target_size))

        if args.insert_rotation_every and args.insert_rotation_every > 0 and (idx + 1) % int(args.insert_rotation_every) == 0:
            rotation_frames = render_views_around_mesh(
                scenes[idx],
                num_views=int(args.predicted_room_num_views),
                radius=float(args.predicted_room_camera_radius),
                image_size=target_size,
                light_intensity=5.0,
                bg_color=(255, 255, 255, 255),
            )
            for rot_frame in rotation_frames:
                rot_img = _to_pil_rgb(rot_frame).resize(target_size, Image.LANCZOS)
                scene_output_frames.append(rot_img)
                comparison_frames.append(_compose_side_by_side(last_gt, rot_img, target_size))

    export_renderings(scene_output_frames, str(export_dir / "animation_scene.gif"), fps=int(args.animation_fps))
    export_renderings(comparison_frames, str(export_dir / "animation.gif"), fps=int(args.animation_fps))


def _auto_room_aux_path(transformer_dir: Path) -> Optional[Path]:
    candidates = [
        transformer_dir / "room_layout_aux_head.pt",
        transformer_dir.parent / "room_layout_aux_head.pt",
        transformer_dir.parent.parent / "room_layout_aux_head.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_room_layout_head(
    path: Optional[Path],
    in_channels: int,
    hidden_dim: int,
    device: torch.device,
) -> Optional[RoomLayoutAuxiliaryHead]:
    if path is None or not path.is_file():
        return None
    head = RoomLayoutAuxiliaryHead(in_channels=in_channels, hidden_dim=hidden_dim, out_dim=12)
    state = torch.load(path, map_location="cpu")
    head.load_state_dict(state)
    head.to(device=device, dtype=torch.float32)
    head.eval()
    print(f"Loaded room-layout auxiliary head from {path}")
    return head


def _predict_room_shell(
    result,
    scene_meshes: List,
    room_head: Optional[RoomLayoutAuxiliaryHead],
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[tuple[trimesh.Scene, dict]]:
    bounds = _scene_bounds_from_meshes(scene_meshes)
    if bounds is None:
        return None
    object_center = 0.5 * (bounds[0] + bounds[1])
    object_extent = np.maximum(bounds[1] - bounds[0], 1e-4)

    source = "generated_bounds_fallback"
    room_center = object_center.copy()
    room_extent = np.maximum(object_extent * float(args.room_bounds_scale), np.asarray(args.room_min_extent, dtype=np.float64))
    presence = np.ones(6, dtype=bool)
    raw_code = None

    scene_latents = getattr(result, "scene_latents", None)
    if room_head is not None and scene_latents is not None and scene_latents.numel() > 0:
        with torch.no_grad():
            latents = scene_latents.to(device=device, dtype=torch.float32)
            num_parts = torch.tensor([latents.shape[0]], device=device, dtype=torch.long)
            code = room_head(latents, num_parts)[0].detach().cpu().float().numpy()
        raw_code = code.tolist()
        rel_center = np.clip(code[:3], -float(args.room_center_clamp), float(args.room_center_clamp))
        pred_extent = np.exp(np.clip(code[3:6], np.log(args.room_extent_min), np.log(args.room_extent_max)))
        pred_extent = np.maximum(pred_extent, object_extent)
        logits = code[6:12]
        presence = 1.0 / (1.0 + np.exp(-logits)) >= float(args.room_presence_threshold)
        if not presence.any():
            presence[:] = True
        room_extent = pred_extent.astype(np.float64)
        room_center = object_center + rel_center.astype(np.float64) * room_extent
        source = "room_layout_aux_head"

    room_shell = _make_box_room_shell(
        center=room_center,
        extent=room_extent,
        presence=presence,
        alpha=float(args.room_alpha),
        wall_thickness=float(args.room_wall_thickness),
    )
    metadata = {
        "source": source,
        "is_learned_room_layout": source == "room_layout_aux_head",
        "room_center": room_center.tolist(),
        "room_extent": room_extent.tolist(),
        "shell_presence": {
            "floor": bool(presence[0]),
            "ceiling": bool(presence[1]),
            "front_wall": bool(presence[2]),
            "back_wall": bool(presence[3]),
            "left_wall": bool(presence[4]),
            "right_wall": bool(presence[5]),
        },
        "object_bounds": bounds.tolist(),
        "raw_room_layout_code": raw_code,
        "warning": (
            None
            if source == "room_layout_aux_head"
            else "This room shell is a generated-bounds fallback, not a learned room-layout auxiliary prediction."
        ),
    }
    return room_shell, metadata



def main():
    parser = argparse.ArgumentParser(description="PartCrafter 3D+4D inference")
    parser.add_argument(
        "--config_key",
        type=str,
        default=None,
        help="Key in the JSON config to load defaults from.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("infer.json"),
        help="Path to JSON config file (default: ./infer.json).",
    )
    parser.add_argument(
        "--first_frame_index",
        type=int,
        default=None,
        help="Index of the first frame of the scene/video",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="Directory containing ordered RGB frames of the scene/video",
    )
    parser.add_argument(
        "--frames_original_dir",
        type=str,
        default=None,
        help="Optional directory containing original RGB frames before background removal.",
    )
    parser.add_argument(
        "--load_frames_no_bg",
        type=int,
        default=0,
        help="When loading from a config data_dir, use frames_no_bg instead of frames.",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default=None,
        help="Optional directory containing binary masks aligned with frames",
    )
    parser.add_argument(
        "--masks_static_dir",
        type=str,
        default=None,
        help="Optional directory containing binary masks aligned with frames",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_3d4d",
        help="Directory to save meshes and renders",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Run identifier appended to output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=2048,
        help="VAE token count",
    )
    parser.add_argument(
        "--scene_steps",
        type=int,
        default=50,
        help="Diffusion steps for static scene reconstruction",
    )
    parser.add_argument(
        "--dynamic_steps",
        type=int,
        default=50,
        help="Diffusion steps for dynamic object reconstruction",
    )
    parser.add_argument(
        "--scene_num_parts",
        type=int,
        default=1,
        help="Number of static scene parts to reconstruct",
    )
    parser.add_argument(
        "--dynamic_num_parts",
        type=int,
        default=1,
        help="Number of dynamic object parts to reconstruct",
    )
    parser.add_argument(
        "--scene_block_size",
        type=int,
        default=1,
        help="Block size for static scene parts to reconstruct",
    )
    parser.add_argument(
        "--scene_guidance",
        type=float,
        default=7.0,
        help="Guidance scale during scene reconstruction",
    )
    parser.add_argument(
        "--dynamic_guidance",
        type=float,
        default=7.0,
        help="Guidance scale during dynamic reconstruction",
    )
    parser.add_argument(
        "--object_only_condition",
        action="store_true",
        help="Use masked moving object for dynamic conditioning",
    )
    parser.add_argument(
        "--dynamic_ar_block_size",
        type=int,
        default=8,
        help="Autoregressive block size for dynamic stage",
    )
    parser.add_argument(
        "--history_mode",
        type=str,
        default="fixed",
        choices=["fixed", "soft"],
        help="History blending strategy for dynamic stage",
    )
    parser.add_argument(
        "--history_soft_alpha",
        type=float,
        default=0.1,
        help="Alpha for soft history blending",
    )
    parser.add_argument(
        "--history_renoise_sigma",
        type=float,
        default=0.0,
        help="Sigma for re-noising history latents",
    )
    parser.add_argument(
        "--animation",
        action="store_true",
        help="Export animated GIF with reconstructed scene",
    )
    parser.add_argument(
        "--animation_fps",
        type=int,
        default=8,
        help="FPS for exported animation",
    )
    parser.add_argument(
        "--insert_rotation_every",
        type=int,
        default=0,
        help="Insert a 360 render every N frames (0 disables)",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=1024,
        help="Render resolution (square)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=518,
        help=(
            "Conditioning image size for inference preprocessing (square). "
            "Use 518 to match training config image_load_size/dino_preprocess_size."
        ),
    )
    parser.add_argument(
        "--base_weights_dir",
        type=str,
        default="wgsxm/PartCrafter-Scene",
        help="HF repo-id or local directory for base weights",
    )
    parser.add_argument(
        "--transformer_dir",
        type=str,
        default=None,
        help="Directory containing transformer weights (config.json + diffusion model)",
    )
    parser.add_argument(
        "--scene_attn_ids",
        type=str,
        default=None,
        help="Optional space/comma separated list of scene attention block ids",
    )
    parser.add_argument(
        "--dynamic_attn_ids",
        type=str,
        default=None,
        help="Optional space/comma separated list of dynamic attention block ids",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--frames_start_idx",
        type=int,
        default=0,
        help="Start index for frames (inclusive)",
    )
    parser.add_argument(
        "--frames_end_idx",
        type=int,
        default=None,
        help="End index for frames (exclusive)",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Stride for sampling frames and masks (1 keeps every frame)",
    )
    parser.add_argument(
        "--use_dino_multi",
        type=int,
        default=0,
        help="Use multi-channel DINO model",
    )
    parser.add_argument(
        "--use_latents_with_timesteps",
        type=int,
        default=0,
        help="Use latents with timesteps",
    )
    parser.add_argument(
        "--prevent_collisions",
        type=int,
        default=0,
        help="Prevent collisions between dynamic and static objects",
    )
    parser.add_argument(
        "--scene_mix_cutoff",
        type=int,
        default=10,
        help="Cutoff for mixing static and dynamic scenes",
    )
    parser.add_argument(
        "--dynamic_mix_cutoff",
        type=int,
        default=10,
        help="Cutoff for mixing static and dynamic scenes",
    )
    parser.add_argument(
        "--dynamic_max_memory_frames",
        type=int,
        default=10,
        help="Maximum number of frames to keep in memory for dynamic objects",
    )
    parser.add_argument(
        "--render_predicted_room",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export a semi-transparent procedural room shell from the room-layout auxiliary head when available.",
    )
    parser.add_argument(
        "--render_predicted_room_gifs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also render extra predicted_room/*.gif previews. The main room-augmented outputs are animation.gif and animation_scene.gif.",
    )
    parser.add_argument(
        "--room_augment_animations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate top-level animation.gif and animation_scene.gif with the predicted room shell included.",
    )
    parser.add_argument("--predicted_room_num_views", type=int, default=36, help="Number of views for predicted-room rotating GIFs.")
    parser.add_argument("--predicted_room_camera_radius", type=float, default=4.0, help="Camera radius for predicted-room rotating GIFs.")
    parser.add_argument(
        "--room_layout_aux_path",
        type=str,
        default=None,
        help="Optional room_layout_aux_head.pt path. If omitted, infer from --transformer_dir parents.",
    )
    parser.add_argument(
        "--room_layout_hidden_dim",
        type=int,
        default=256,
        help="Hidden dim used by the room-layout auxiliary head.",
    )
    parser.add_argument("--room_alpha", type=float, default=0.18, help="Predicted room shell alpha in [0, 1].")
    parser.add_argument("--room_wall_thickness", type=float, default=0.025, help="Procedural wall/floor/ceiling thickness.")
    parser.add_argument("--room_presence_threshold", type=float, default=0.35, help="Sigmoid threshold for predicted shell presence.")
    parser.add_argument("--room_extent_min", type=float, default=0.5, help="Minimum room extent for auxiliary prediction clamp.")
    parser.add_argument("--room_extent_max", type=float, default=30.0, help="Maximum room extent for auxiliary prediction clamp.")
    parser.add_argument("--room_center_clamp", type=float, default=2.0, help="Clamp for relative room center prediction.")
    parser.add_argument(
        "--room_bounds_scale",
        type=float,
        default=1.6,
        help="Fallback scale around generated object bounds if no room-layout head is available.",
    )
    parser.add_argument(
        "--room_min_extent",
        type=float,
        nargs=3,
        default=[2.0, 2.4, 2.0],
        help="Fallback minimum [x y z] room extent.",
    )
    config_args, remaining = parser.parse_known_args()

    seed_all(31)

    config_defaults = {}
    if config_args.config_key:
        config_path = Path(config_args.config_path).expanduser()
        if not config_path.is_file():
            parser.error(f"Configuration file not found: {config_path}")
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as exc:
            parser.error(f"Failed to parse configuration file {config_path}: {exc}")

        if config_args.config_key not in config_data:
            parser.error(
                f"Configuration key '{config_args.config_key}' not found in {config_path}"
            )

        cfg = config_data[config_args.config_key]
        data_dir_value = cfg.get("data_dir")
        if not data_dir_value:
            parser.error(
                f"Configuration '{config_args.config_key}' is missing 'data_dir'"
            )

        data_dir_path = Path(data_dir_value)
        ## CONDITION TEST
        frames_subdir = "frames_no_bg" if config_args.load_frames_no_bg else "frames"
        config_defaults["frames_dir"] = str(data_dir_path / frames_subdir)
        config_defaults["frames_original_dir"] = str(data_dir_path / "frames")
        config_defaults["masks_dir"] = str(data_dir_path / "masks")
        config_defaults["masks_static_dir"] = str(data_dir_path / "masks_static")
        config_defaults["tag"] = cfg.get("tag", data_dir_path.name)

        key_map = {
            "num_tokens": "num_tokens",
            "first_frame_idx": "first_frame_index",
            "frames_start_idx": "frames_start_idx",
            "frames_end_idx": "frames_end_idx",
            "scene_num_parts": "scene_num_parts",
            "dynamic_num_parts": "dynamic_num_parts",
            "dynamic_ar_block_size": "dynamic_ar_block_size",
            "scene_ar_block_size": "scene_block_size",
            "scene_mix_cutoff": "scene_mix_cutoff",
            "dynamic_mix_cutoff": "dynamic_mix_cutoff",
            "dynamic_max_memory_frames": "dynamic_max_memory_frames",
            "history_mode": "history_mode",
            "insert_rotation_every": "insert_rotation_every",
            "use_dino_multi": "use_dino_multi",
            "use_latents_with_timesteps": "use_latents_with_timesteps",
            "transformer_dir": "transformer_dir",
            "seed": "seed",
            "output_dir": "output_dir",
            "scene_attn_ids": "scene_attn_ids",
            "dynamic_attn_ids": "dynamic_attn_ids",
            "base_weights_dir": "base_weights_dir",
            "frame_stride": "frame_stride",
            "image_size": "image_size",
            "render_predicted_room": "render_predicted_room",
            "render_predicted_room_gifs": "render_predicted_room_gifs",
            "room_augment_animations": "room_augment_animations",
            "predicted_room_num_views": "predicted_room_num_views",
            "predicted_room_camera_radius": "predicted_room_camera_radius",
            "room_layout_aux_path": "room_layout_aux_path",
            "room_layout_hidden_dim": "room_layout_hidden_dim",
            "room_alpha": "room_alpha",
            "room_wall_thickness": "room_wall_thickness",
            "room_presence_threshold": "room_presence_threshold",
            "room_extent_min": "room_extent_min",
            "room_extent_max": "room_extent_max",
            "room_center_clamp": "room_center_clamp",
            "room_bounds_scale": "room_bounds_scale",
            "room_min_extent": "room_min_extent",
        }

        for key, arg_name in key_map.items():
            if key in cfg and cfg[key] is not None:
                config_defaults[arg_name] = cfg[key]

    parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    if args.frame_stride <= 0:
        parser.error("--frame_stride must be a positive integer")
    if args.image_size <= 0:
        parser.error("--image_size must be a positive integer")
    if not 0.0 <= args.room_alpha <= 1.0:
        parser.error("--room_alpha must be in [0, 1]")
    if args.room_extent_min <= 0.0 or args.room_extent_max <= 0.0 or args.room_extent_max < args.room_extent_min:
        parser.error("--room_extent_min/--room_extent_max must be positive and max >= min")

    if args.config_key:
        print(
            f"Loaded configuration '{args.config_key}' from {Path(args.config_path).expanduser()}"
        )
    if isinstance(args.config_path, Path):
        args.config_path = str(args.config_path)

    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    
    scene_num_parts = max(0, int(args.scene_num_parts))
    dynamic_num_parts = max(0, int(args.dynamic_num_parts))

    frames_dir = Path(args.frames_dir)
    frames_original_dir = Path(args.frames_original_dir) if args.frames_original_dir else None
    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    masks_static_dir = Path(args.masks_static_dir) if args.masks_static_dir else None
    os.makedirs(masks_static_dir, exist_ok=True)

    frames: List[Image.Image] = _gather_images(frames_dir)
    frames_original: Optional[List[Image.Image]] = _gather_images(frames_original_dir) if frames_original_dir else None
    masks: List[List[Image.Image]] = _gather_all_masks(masks_dir, dynamic_num_parts) if masks_dir else []
    masks_static: List[List[Image.Image]] = _gather_all_masks(masks_static_dir, scene_num_parts) if masks_static_dir else []
    print(f"Loaded {len(frames)} frames from {frames_dir}")
    print(f"Loaded {len(masks)} sets of masks from {masks_dir} each with {len(masks[0])} masks." if masks_dir else "No masks loaded")

    reverse_sequence = False

    if reverse_sequence:
        frames = list(reversed(frames))
        masks = [list(reversed(mask_list)) for mask_list in masks]
        masks_static = [list(reversed(mask_list)) for mask_list in masks_static]

    frames = [_resize_rgb_image(frame, args.image_size) for frame in frames]
    if masks:
        masks = [
            [_resize_mask_image(mask, args.image_size) for mask in mask_list]
            for mask_list in masks
        ]
    if masks_static:
        masks_static = [
            [_resize_mask_image(mask, args.image_size) for mask in mask_list]
            for mask_list in masks_static
        ]
    print(
        f"Conditioning image sizing: dataloader={args.image_size}x{args.image_size}, "
        f"DINO preprocess={args.image_size}x{args.image_size}"
    )

    if len(masks_static) == 0:
        masks_static = []

        collected = []
        for i in range(len(masks[0])):
            combined_mask = _combine_masks([m[i] for m in masks]) if masks else None
            inverted_mask = _invert_mask(combined_mask) if combined_mask else None
            collected.append(inverted_mask)

        masks_static.append(collected)
        print(f"Generated {len(collected)} static masks by inverting dynamic masks.")

    print(f"Loaded {len(masks_static)} sets of static masks from {masks_static_dir} each with {len(masks_static[0])} masks." if masks_static_dir else "No static masks loaded")

    frame_slice = slice(args.frames_start_idx, args.frames_end_idx, args.frame_stride)
    start_idx, stop_idx, stride_step = frame_slice.indices(len(frames))
    selected_indices = list(range(start_idx, stop_idx, stride_step))

    if not selected_indices:
        parser.error(
            "Frame selection produced no frames. Adjust --frames_start_idx/--frames_end_idx/--frame_stride."
        )

    frames_for_pipeline = [frames[i] for i in selected_indices]
    masks_for_pipeline = (
        [[mask_list[i] for i in selected_indices] for mask_list in masks] if masks else None
    )
    masks_static_for_pipeline = (
        [[mask_list[i] for i in selected_indices] for mask_list in masks_static] if masks_static else None
    )

    if args.first_frame_index not in selected_indices:
        print(
            f"Warning: first_frame_index {args.first_frame_index} is not included in the sampled frames."
        )

    print(
        f"Using {len(frames_for_pipeline)} frames (of {len(frames)}) for inference "
        f"with stride={args.frame_stride}, start={start_idx}, end={stop_idx}."
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_tag = (args.tag or Path(args.frames_dir).name).replace(" ", "_")
    tag_components = [
        base_tag,
        timestamp,
    ]
    tag = "_".join(str(t) for t in tag_components if str(t))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    export_dir = output_root / tag
    export_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)
    
    print("Building PartCrafter 3D4D inference pipeline..., base dir:", args.base_weights_dir, "transformer dir:", args.transformer_dir)

    cache_dir = Path("pretrained_weights")
    base_dir = _resolve_repo_or_dir(args.base_weights_dir, cache_dir)
    transformer_dir = Path(args.transformer_dir)

    print("Scene attention IDs:", _parse_id_string(args.scene_attn_ids))
    print("Dynamic attention IDs:", _parse_id_string(args.dynamic_attn_ids))

    pipe = build_pipeline(
        PartFrameCrafterDiTModel, 
        PartCrafter3D4DInferencePipeline, 
        base_dir, 
        transformer_dir, 
        device, 
        dtype, 
        use_dino_multi=args.use_dino_multi, 
        transformer_scene_attn_ids=_parse_id_string(args.scene_attn_ids), 
        transformer_dynamic_attn_ids=_parse_id_string(args.dynamic_attn_ids)
    )
    pipe.set_progress_bar_config(disable=False)

    room_layout_aux_path = Path(args.room_layout_aux_path) if args.room_layout_aux_path else _auto_room_aux_path(transformer_dir)
    room_layout_head = None
    if args.render_predicted_room:
        room_layout_head = _load_room_layout_head(
            room_layout_aux_path,
            in_channels=int(pipe.transformer.config.in_channels),
            hidden_dim=int(args.room_layout_hidden_dim),
            device=device,
        )
        if room_layout_head is None:
            print(
                "Warning: room-layout auxiliary head was not found; "
                "predicted room export will use generated-bounds fallback."
            )

    # pipe.transformer._remove_static_dynamic_embedding()
    
    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    animation_path = None
    if args.animation:
        animation_path = str(export_dir / "animation.gif")

    render_kwargs = {
        "image_size": (args.render_size, args.render_size),
    }

    scene_attn_ids = _parse_id_string(args.scene_attn_ids)
    dynamic_attn_ids = _parse_id_string(args.dynamic_attn_ids)

    first_frame = frames[args.first_frame_index]
    first_frame_mask = _combine_masks([m[args.first_frame_index] for m in masks_static])
    first_frame_mask = _invert_mask(first_frame_mask)
    # first_frame_mask = _combine_masks([m[args.first_frame_index] for m in masks])
    # first_frame_mask.save("first_frame_mask.png")

    try:
        with open(export_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        with open(export_dir / "cmd.txt", "w") as f:
            f.write(" ".join(sys.argv) + "\n")
        script_path = os.path.abspath(__file__)
        shutil.copy(script_path, export_dir / os.path.basename(script_path))
        pipeline_src = inspect.getsourcefile(PartCrafter3D4DInferencePipeline) or inspect.getfile(PartCrafter3D4DInferencePipeline)
        if pipeline_src and os.path.isfile(pipeline_src):
            shutil.copy(pipeline_src, export_dir / os.path.basename(pipeline_src))
        transformer_src = inspect.getsourcefile(PartFrameCrafterDiTModel) or inspect.getfile(PartFrameCrafterDiTModel)
        if transformer_src and os.path.isfile(transformer_src):
            shutil.copy(transformer_src, export_dir / os.path.basename(transformer_src))
    except Exception as exc:
        print(f"Warning: failed to save run metadata: {exc}")

    result = pipe(
        first_frame=first_frame,
        first_frame_mask=first_frame_mask,
        frames=frames_for_pipeline,
        masks=masks_for_pipeline,
        masks_static=masks_static_for_pipeline,
        all_masks=masks,
        num_tokens=args.num_tokens,
        scene_inference_steps=args.scene_steps,
        dynamic_inference_steps=args.dynamic_steps,
        guidance_scale_scene=args.scene_guidance,
        guidance_scale_dynamic=args.dynamic_guidance,
        generator=generator,
        use_object_only_condition=args.object_only_condition,
        dynamic_ar_block_size=args.dynamic_ar_block_size,
        history_renoise_sigma=args.history_renoise_sigma,
        history_soft_alpha=args.history_soft_alpha,
        history_mode=args.history_mode,
        animation_path=animation_path,
        animation_fps=args.animation_fps,
        insert_rotation_every=args.insert_rotation_every,
        render_kwargs=render_kwargs,
        scene_attention_ids=scene_attn_ids,
        dynamic_attention_ids=dynamic_attn_ids,
        scene_num_parts=scene_num_parts,
        dynamic_num_parts=dynamic_num_parts,
        scene_block_size=args.scene_block_size,
        prevent_collisions=args.prevent_collisions,
        first_frame_index=args.first_frame_index,
        scene_mix_cutoff=args.scene_mix_cutoff,
        dynamic_mix_cutoff=args.dynamic_mix_cutoff,
        dynamic_max_memory_frames=args.dynamic_max_memory_frames,
        image_size=args.image_size,
    )
    scene_dir = export_dir / "scene"
    dynamic_dir = export_dir / "dynamic"
    scene_dir.mkdir(exist_ok=True)
    dynamic_dir.mkdir(exist_ok=True)

    raw_scene_meshes = list(getattr(result, "scene_meshes", None) or [])
    scene_static_meshes = [mesh for mesh in raw_scene_meshes[:scene_num_parts] if mesh is not None]
    scene_meshes: List = []
    if raw_scene_meshes:
        for idx, mesh in enumerate(raw_scene_meshes):
            if mesh is None:
                continue
            mesh.export(scene_dir / f"scene_{idx:03d}.glb")
            scene_meshes.append(mesh)

    if scene_meshes:
        try:
            full_scene = get_colored_mesh_composition(scene_meshes)
            if full_scene is not None:
                full_scene.export(scene_dir / "scene_all.glb")
        except Exception as exc:
            print(f"Warning: failed to export scene_all.glb: {exc}")

    if scene_static_meshes:
        try:
            static_scene = get_colored_mesh_composition(scene_static_meshes, is_random=False)
            if static_scene is not None:
                static_scene.export(scene_dir / "scene.glb")
        except Exception as exc:
            print(f"Warning: failed to export static-only scene: {exc}")
    elif scene_meshes:
        try:
            fallback_scene = get_colored_mesh_composition(scene_meshes, is_random=False)
            if fallback_scene is not None:
                fallback_scene.export(scene_dir / "scene.glb")
            print("Warning: scene.glb fell back to the full scene because no static scene slice was available.")
        except Exception as exc:
            print(f"Warning: failed to export fallback scene.glb: {exc}")

    dynamic_meshes: List = []
    dynamic_frame_mesh_exports: dict[int, List] = {}
    if getattr(result, "dynamic_meshes", None):
        max_dynamic_objects = max(
            (len(frame_meshes) for frame_meshes in result.dynamic_meshes if frame_meshes),
            default=0,
        )
        dynamic_object_ids = _discover_dynamic_object_ids(masks_dir, max_dynamic_objects)
        for frame_idx, frame_meshes in enumerate(result.dynamic_meshes):
            if not frame_meshes:
                continue
            frame_mesh_exports: List = []
            for part_idx, mesh in enumerate(frame_meshes):
                if mesh is None:
                    continue
                object_id = dynamic_object_ids[part_idx] if part_idx < len(dynamic_object_ids) else (part_idx + 1)
                object_dir = dynamic_dir / f"object_{object_id:03d}"
                object_dir.mkdir(exist_ok=True)
                mesh.export(object_dir / f"frame_{frame_idx:03d}.glb")
                dynamic_meshes.append(mesh)
                frame_mesh_exports.append(mesh)
            if not frame_mesh_exports:
                continue
            dynamic_frame_mesh_exports[frame_idx] = frame_mesh_exports
            try:
                combined_input = scene_static_meshes + frame_mesh_exports if scene_static_meshes else frame_mesh_exports
                combined = get_colored_mesh_composition(combined_input, is_random=False)
                if combined is not None:
                    combined.export(dynamic_dir / f"dynamic_scene_frame_{frame_idx:03d}.glb")
            except Exception as exc:
                print(f"Warning: failed to export dynamic composite for frame {frame_idx}: {exc}")

    if args.render_predicted_room:
        room_source_meshes = scene_static_meshes or scene_meshes
        room_result = _predict_room_shell(result, room_source_meshes, room_layout_head, args, device)
        if room_result is None:
            print("Warning: could not export predicted room shell because generated scene bounds were unavailable.")
        else:
            room_shell, room_metadata = room_result
            room_dir = export_dir / "predicted_room"
            room_dir.mkdir(exist_ok=True)
            room_shell.export(room_dir / "room_shell.glb")
            with open(room_dir / "room_layout.json", "w") as f:
                json.dump(room_metadata, f, indent=2)
            if not room_metadata.get("is_learned_room_layout", False):
                fallback_note = (
                    "NOT A LEARNED ROOM LAYOUT\n"
                    "\n"
                    "room_layout_aux_head.pt was not available or usable for this inference run.\n"
                    "The exported room shell was built from generated object bounds using the fallback\n"
                    "room_bounds_scale/room_min_extent settings.\n"
                )
                with open(room_dir / "README_FALLBACK_NOT_LEARNED.txt", "w") as f:
                    f.write(fallback_note)
                room_shell.export(room_dir / "room_shell_FALLBACK_NOT_LEARNED.glb")
                print(
                    "Warning: exported room shell is NOT learned; "
                    f"fallback marker written to {room_dir / 'README_FALLBACK_NOT_LEARNED.txt'}"
                )
            print(f"Wrote predicted room shell to {room_dir / 'room_shell.glb'}")

            room_shell_for_render = room_shell
            room_static_for_render = None
            room_scene_for_render = None
            try:
                if scene_meshes:
                    scene_with_room = _combine_with_room_shell(scene_meshes, room_shell)
                    scene_with_room.export(scene_dir / "scene_all_with_room.glb")
                    room_scene_for_render = scene_with_room
                if scene_static_meshes:
                    static_with_room = _combine_with_room_shell(scene_static_meshes, room_shell)
                    static_with_room.export(scene_dir / "scene_with_room.glb")
                    room_static_for_render = static_with_room
            except Exception as exc:
                print(f"Warning: failed to export scene with predicted room shell: {exc}")

            room_dynamic_dir = dynamic_dir / "with_predicted_room"
            room_dynamic_dir.mkdir(exist_ok=True)
            room_dynamic_scenes_for_render: List[trimesh.Scene] = []
            for frame_idx, frame_mesh_exports in dynamic_frame_mesh_exports.items():
                try:
                    combined_input = (scene_static_meshes + frame_mesh_exports) if scene_static_meshes else frame_mesh_exports
                    frame_with_room = _combine_with_room_shell(combined_input, room_shell)
                    frame_with_room.export(room_dynamic_dir / f"dynamic_scene_frame_{frame_idx:03d}.glb")
                    room_dynamic_scenes_for_render.append(frame_with_room)
                except Exception as exc:
                    print(f"Warning: failed to export dynamic frame {frame_idx} with predicted room shell: {exc}")

            if args.render_predicted_room_gifs:
                try:
                    _render_room_scene_gif(room_shell_for_render, room_dir / "room_shell.gif", args)
                    if room_scene_for_render is not None:
                        _render_room_scene_gif(room_scene_for_render, room_dir / "scene_all_with_room.gif", args)
                    elif room_static_for_render is not None:
                        _render_room_scene_gif(room_static_for_render, room_dir / "scene_with_room.gif", args)
                    if room_dynamic_scenes_for_render:
                        _render_room_animation_gif(room_dynamic_scenes_for_render, room_dir / "dynamic_with_room.gif", args)
                    print(f"Wrote predicted room renders to {room_dir}")
                except Exception as exc:
                    print(f"Warning: failed to render predicted room GIFs: {exc}")

            if args.room_augment_animations:
                try:
                    animation_scenes = room_dynamic_scenes_for_render
                    if not animation_scenes:
                        animation_scenes = [room_scene_for_render or room_static_for_render or room_shell_for_render]
                    _render_room_augmented_animations(animation_scenes, frames_for_pipeline, export_dir, args)
                    print(f"Wrote room-augmented animation GIFs to {export_dir / 'animation.gif'} and {export_dir / 'animation_scene.gif'}")
                except Exception as exc:
                    print(f"Warning: failed to render room-augmented animation GIFs: {exc}")

    try:
        with open(export_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        with open(export_dir / "cmd.txt", "w") as f:
            f.write(" ".join(sys.argv) + "\n")
        script_path = os.path.abspath(__file__)
        shutil.copy(script_path, export_dir / os.path.basename(script_path))
        pipeline_src = inspect.getsourcefile(PartCrafter3D4DInferencePipeline) or inspect.getfile(PartCrafter3D4DInferencePipeline)
        if pipeline_src and os.path.isfile(pipeline_src):
            shutil.copy(pipeline_src, export_dir / os.path.basename(pipeline_src))
        transformer_src = inspect.getsourcefile(PartFrameCrafterDiTModel) or inspect.getfile(PartFrameCrafterDiTModel)
        if transformer_src and os.path.isfile(transformer_src):
            shutil.copy(transformer_src, export_dir / os.path.basename(transformer_src))
    except Exception as exc:
        print(f"Warning: failed to save run metadata: {exc}")

    if result.animation_path is not None:
        print(f"Saved animation to {result.animation_path}")

    print(f"Inference complete. Outputs written to {export_dir}")


if __name__ == "__main__":
    main()

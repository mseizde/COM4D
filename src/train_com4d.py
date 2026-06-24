import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

from src.utils.typing_utils import *

import argparse
import json
import logging
import time
import signal
import math
import gc
import random
from datetime import timedelta
from contextlib import nullcontext
from packaging import version
from pathlib import Path

import trimesh
from PIL import Image
import numpy as np
import wandb
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as tF
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin, InitProcessGroupKwargs
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3
)

from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from src.schedulers import RectifiedFlowScheduler
from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import PartFrameCrafterDiTModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline, FourDCrafterPipeline

# Datasets: 3D parts and 4D frames (frames-as-parts)
from src.datasets.objaverse_part import (
    ObjaversePartDataset as ObjaversePartDataset3D,
    BatchedObjaversePartDataset as BatchedObjaversePartDataset3D,
    ObjaversePartDatasetOriginal as ObjaversePartDatasetOriginal,
    BatchedObjaversePartDatasetOriginal as BatchedObjaversePartDatasetOriginal,
)
from src.datasets.animated_frame import (
    ObjaversePartDataset as ObjaversePartDataset4D,
    BatchedObjaversePartDataset as BatchedObjaversePartDataset4D,
)
from src.datasets import (
    MultiEpochsDataLoader,
    yield_forever,
)
from src.datasets.local_cache import configure_dataset_cache, prefetch_data_configs, resolve_path
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.train_utils import (
    MyEMAModel, 
    get_configs,
    get_optimizer,
    get_lr_scheduler,
    save_experiment_params,
    save_model_architecture,
)
from src.utils.render_utils import (
    render_views_around_mesh, 
    render_normal_views_around_mesh, 
    make_grid_for_images_or_videos,
    export_renderings
)
from src.utils.metric_utils import compute_cd_and_f_score_in_training
import copy
from src.models.attention_processor import (
    trace_sequence_parallel_event,
    PartCrafterAttnProcessor,
    PartFrameCrafterAttnProcessor,
    TripoSGAttnProcessor2_0,
)

HUMOTO_PHYSICS_DATASET_JSON = "/data/mseizde/com4d/COM4D/dataset_json/humoto.json"


class ValidationSampleTimeout(RuntimeError):
    pass


class LayoutPoseAuxiliaryHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 256, out_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        pooled = latent_tokens.float().mean(dim=1)
        return self.net(pooled)


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


def _checkpoint_aux_head_path(
    pretrained_root: Optional[str],
    pretrained_ckpt: Optional[int],
    filename: str,
) -> Optional[str]:
    if pretrained_root is None or pretrained_ckpt is None:
        return None
    root = os.path.abspath(pretrained_root)
    ckpt_name = f"{int(pretrained_ckpt):06d}"
    candidates = [
        os.path.join(root, "checkpoints", ckpt_name, filename),
        os.path.join(root, ckpt_name, filename),
        os.path.join(root, filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _load_aux_head_state_if_available(
    head: Optional[nn.Module],
    path: Optional[str],
    name: str,
    logger,
) -> None:
    if head is None or path is None:
        return
    if not os.path.isfile(path):
        logger.info(f"{name} auxiliary init path does not exist, keeping random init: {path}\n")
        return
    state = torch.load(path, map_location="cpu")
    head.load_state_dict(state)
    logger.info(f"Initialized {name} auxiliary head from {path}\n")


_ROOM_METADATA_CACHE: dict[str, Optional[dict]] = {}
_ROOM_BOUNDS_CACHE: dict[tuple[str, ...], Optional[tuple[list[float], list[float]]]] = {}
_ROOM_LAYOUT_DEBUG_COUNTS: dict[str, int] = {}


def _note_room_layout_debug(reason: str, detail: str = "", limit: int = 8) -> None:
    count = _ROOM_LAYOUT_DEBUG_COUNTS.get(reason, 0)
    _ROOM_LAYOUT_DEBUG_COUNTS[reason] = count + 1
    if count < limit:
        suffix = f": {detail}" if detail else ""
        print(f"[layout_pose_aux] {reason}{suffix}", flush=True)


def _load_room_metadata(room_geometry: dict) -> Optional[dict]:
    path = room_geometry.get("geometry_metadata_path") if isinstance(room_geometry, dict) else None
    if not path:
        _note_room_layout_debug("missing_geometry_metadata_path")
        return None
    if path in _ROOM_METADATA_CACHE:
        return _ROOM_METADATA_CACHE[path]
    try:
        with open(resolve_path(path), "r") as f:
            metadata = json.load(f)
    except Exception as exc:
        _note_room_layout_debug("metadata_load_failed", f"{path} ({type(exc).__name__}: {exc})")
        metadata = None
    _ROOM_METADATA_CACHE[path] = metadata
    return metadata


def _metadata_room_extent(metadata: Optional[dict]) -> Optional[list[float]]:
    if not isinstance(metadata, dict):
        return None
    raw_size = metadata.get("room_size")
    if isinstance(raw_size, (list, tuple)) and len(raw_size) >= 3:
        try:
            size = [abs(float(v)) for v in raw_size[:3]]
        except (TypeError, ValueError):
            return None
        if min(size) > 1e-6:
            return size
    return None


def _mesh_or_scene_bounds(path: str) -> Optional[np.ndarray]:
    try:
        geom = trimesh.load(resolve_path(path), process=False)
        bounds = np.asarray(geom.bounds, dtype=np.float64)
    except Exception as exc:
        _note_room_layout_debug("shell_bounds_load_failed", f"{path} ({type(exc).__name__}: {exc})")
        return None
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        return None
    extent = bounds[1] - bounds[0]
    if np.max(extent) <= 1e-6:
        return None
    return bounds


def _room_shell_center_extent(room_geometry: dict) -> Optional[tuple[list[float], list[float]]]:
    paths = tuple(
        str(room_geometry.get(key))
        for key in ("floor_path", "wall_path", "ceiling_path")
        if isinstance(room_geometry, dict) and room_geometry.get(key)
    )
    if not paths:
        return None
    if paths in _ROOM_BOUNDS_CACHE:
        return _ROOM_BOUNDS_CACHE[paths]

    bounds_list = []
    for path in paths:
        bounds = _mesh_or_scene_bounds(path)
        if bounds is not None:
            bounds_list.append(bounds)
    if not bounds_list:
        _ROOM_BOUNDS_CACHE[paths] = None
        return None

    mins = np.stack([b[0] for b in bounds_list], axis=0).min(axis=0)
    maxs = np.stack([b[1] for b in bounds_list], axis=0).max(axis=0)
    extent = np.maximum(maxs - mins, 1e-6)
    center = 0.5 * (mins + maxs)
    result = (center.astype(np.float32).tolist(), extent.astype(np.float32).tolist())
    _ROOM_BOUNDS_CACHE[paths] = result
    return result


def _room_layout_reference(
    room_geometry: dict,
    metadata: Optional[dict],
    obj_center: torch.Tensor,
    obj_extent: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor, str]]:
    shell = _room_shell_center_extent(room_geometry)
    if shell is not None:
        center, extent = shell
        return (
            torch.tensor(center, device=obj_center.device, dtype=torch.float32),
            torch.tensor(extent, device=obj_center.device, dtype=torch.float32).clamp_min(1e-6),
            "shell_bounds",
        )

    room_extent = _metadata_room_extent(metadata)
    if room_extent is not None:
        return (
            obj_center,
            torch.tensor(room_extent, device=obj_center.device, dtype=torch.float32).clamp_min(1e-6),
            "metadata_extent",
        )

    raw_size = metadata.get("room_size") if isinstance(metadata, dict) else None
    if raw_size is not None:
        try:
            area = abs(float(raw_size))
        except (TypeError, ValueError):
            area = 0.0
        if area > 1e-6:
            side = math.sqrt(area)
            extent = torch.tensor([side, max(float(obj_extent[1].item()), 1.0), side], device=obj_center.device)
            return (obj_center, extent.clamp_min(1e-6), "metadata_area")

    return None


def _build_layout_pose_targets(
    part_surfaces: torch.Tensor,
    num_parts: torch.Tensor,
    room_geometries: list[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = part_surfaces[..., :3].float()
    part_min = xyz.amin(dim=1)
    part_max = xyz.amax(dim=1)
    part_center = 0.5 * (part_min + part_max)
    part_size = (part_max - part_min).clamp_min(1e-6)

    targets = torch.zeros((xyz.shape[0], 6), device=xyz.device, dtype=torch.float32)
    valid = torch.zeros((xyz.shape[0],), device=xyz.device, dtype=torch.bool)

    ptr = 0
    for obj_idx, count_tensor in enumerate(num_parts.detach().cpu()):
        count = int(count_tensor.item())
        if count <= 0:
            continue
        obj_slice = slice(ptr, ptr + count)
        obj_min = part_min[obj_slice].amin(dim=0)
        obj_max = part_max[obj_slice].amax(dim=0)
        obj_center = 0.5 * (obj_min + obj_max)
        obj_extent = (obj_max - obj_min).clamp_min(1e-6)

        room_geometry = room_geometries[obj_idx] if obj_idx < len(room_geometries) else {}
        metadata = _load_room_metadata(room_geometry)
        reference = _room_layout_reference(room_geometry, metadata, obj_center, obj_extent)
        if reference is None:
            _note_room_layout_debug(
                "room_reference_unavailable",
                str(room_geometry.get("geometry_metadata_path") if isinstance(room_geometry, dict) else ""),
            )
            ptr += count
            continue
        room_center, room_extent, _source = reference
        room_extent = torch.maximum(room_extent, obj_extent)

        targets[obj_slice, :3] = (part_center[obj_slice] - room_center) / room_extent
        targets[obj_slice, 3:] = torch.log(part_size[obj_slice].clamp_min(1e-6))
        valid[obj_slice] = True
        ptr += count

    return targets, valid


def _build_room_layout_targets(
    part_surfaces: torch.Tensor,
    num_parts: torch.Tensor,
    room_geometries: list[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = part_surfaces[..., :3].float()
    part_min = xyz.amin(dim=1)
    part_max = xyz.amax(dim=1)

    targets = torch.zeros((num_parts.shape[0], 12), device=xyz.device, dtype=torch.float32)
    valid = torch.zeros((num_parts.shape[0],), device=xyz.device, dtype=torch.bool)

    ptr = 0
    for obj_idx, count_tensor in enumerate(num_parts.detach().cpu()):
        count = int(count_tensor.item())
        if count <= 0:
            continue
        obj_slice = slice(ptr, ptr + count)
        obj_min = part_min[obj_slice].amin(dim=0)
        obj_max = part_max[obj_slice].amax(dim=0)
        obj_center = 0.5 * (obj_min + obj_max)
        obj_extent = (obj_max - obj_min).clamp_min(1e-6)

        room_geometry = room_geometries[obj_idx] if obj_idx < len(room_geometries) else {}
        metadata = _load_room_metadata(room_geometry)
        reference = _room_layout_reference(room_geometry, metadata, obj_center, obj_extent)
        if reference is None:
            ptr += count
            continue

        room_center, room_extent, _source = reference
        room_extent = torch.maximum(room_extent, obj_extent).clamp_min(1e-6)
        targets[obj_idx, :3] = (room_center - obj_center) / room_extent
        targets[obj_idx, 3:6] = torch.log(room_extent)
        targets[obj_idx, 6] = 1.0 if room_geometry.get("floor_path") else 0.0
        targets[obj_idx, 7] = 1.0 if room_geometry.get("ceiling_path") else 0.0
        targets[obj_idx, 8:12] = 1.0 if room_geometry.get("wall_path") else 0.0
        valid[obj_idx] = True
        ptr += count

    return targets, valid


def _predict_clean_latents(
    raw_model_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    sigmas: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    if objective == "x0":
        return raw_model_pred.float() * (-sigmas.float()) + noisy_latents.float()
    if objective == "v":
        return noisy_latents.float() - sigmas.float() * raw_model_pred.float()
    if objective == "-v":
        return noisy_latents.float() + sigmas.float() * raw_model_pred.float()
    return raw_model_pred.float()


def _sample_surface_queries(
    part_surfaces: torch.Tensor,
    num_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if part_surfaces.ndim != 3 or part_surfaces.shape[-1] < 6:
        raise ValueError("part_surfaces must have shape [N, P, >=6].")
    total_points = part_surfaces.shape[1]
    count = min(max(int(num_points), 1), total_points)
    indices = torch.randint(
        low=0,
        high=total_points,
        size=(part_surfaces.shape[0], count),
        device=part_surfaces.device,
    )
    gather_idx = indices[..., None].expand(-1, -1, part_surfaces.shape[-1])
    sampled = torch.gather(part_surfaces, dim=1, index=gather_idx)
    points = sampled[..., :3].float().detach()
    normals = tF.normalize(sampled[..., 3:6].float().detach(), dim=-1, eps=1e-6)
    return points, normals


def _temporal_grid_pairs(
    num_parts: torch.Tensor,
    num_frames: torch.Tensor | None,
    num_spatial_parts: torch.Tensor | None,
    *,
    fallback_consecutive: bool,
    skip_first_spatial_parts: int = 0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], int]:
    pairs: list[tuple[int, int]] = []
    triples: list[tuple[int, int, int]] = []
    valid_objects = 0
    ptr = 0
    counts = [int(v) for v in num_parts.detach().cpu().tolist()]
    frame_counts = [int(v) for v in num_frames.detach().cpu().tolist()] if num_frames is not None else []
    spatial_counts = [int(v) for v in num_spatial_parts.detach().cpu().tolist()] if num_spatial_parts is not None else []

    for obj_idx, count in enumerate(counts):
        if count <= 1:
            ptr += count
            continue
        frames = frame_counts[obj_idx] if obj_idx < len(frame_counts) else 0
        spatial = spatial_counts[obj_idx] if obj_idx < len(spatial_counts) else 0
        if frames > 1 and spatial > 0 and frames * spatial == count:
            first_spatial = min(max(int(skip_first_spatial_parts), 0), spatial)
            if first_spatial < spatial:
                valid_objects += 1
            for spatial_idx in range(first_spatial, spatial):
                seq = [ptr + frame_idx * spatial + spatial_idx for frame_idx in range(frames)]
                pairs.extend((seq[i], seq[i + 1]) for i in range(len(seq) - 1))
                triples.extend((seq[i], seq[i + 1], seq[i + 2]) for i in range(len(seq) - 2))
        elif fallback_consecutive:
            start = ptr + min(max(int(skip_first_spatial_parts), 0), count)
            if start < ptr + count:
                valid_objects += 1
            seq = list(range(start, ptr + count))
            pairs.extend((seq[i], seq[i + 1]) for i in range(len(seq) - 1))
            triples.extend((seq[i], seq[i + 1], seq[i + 2]) for i in range(len(seq) - 2))
        ptr += count

    return pairs, triples, valid_objects


def _sample_bbox_aligned_surface_queries(
    source_surfaces: torch.Tensor,
    target_surfaces: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    total_points = source_surfaces.shape[1]
    count = min(max(int(num_points), 1), total_points)
    indices = torch.randint(0, total_points, (source_surfaces.shape[0], count), device=source_surfaces.device)
    points = torch.gather(source_surfaces[..., :3].float(), 1, indices[..., None].expand(-1, -1, 3)).detach()

    src_xyz = source_surfaces[..., :3].float().detach()
    dst_xyz = target_surfaces[..., :3].float().detach()
    src_min, src_max = src_xyz.amin(dim=1), src_xyz.amax(dim=1)
    dst_min, dst_max = dst_xyz.amin(dim=1), dst_xyz.amax(dim=1)
    src_center = 0.5 * (src_min + src_max)
    dst_center = 0.5 * (dst_min + dst_max)
    src_extent = (src_max - src_min).clamp_min(1e-6)
    dst_extent = (dst_max - dst_min).clamp_min(1e-6)
    local = (points - src_center[:, None, :]) / src_extent[:, None, :]
    return dst_center[:, None, :] + local * dst_extent[:, None, :]


def _compute_temporal_geometry_auxiliary_loss(
    vae: TripoSGVAEModel,
    clean_latents: torch.Tensor,
    part_surfaces: torch.Tensor,
    num_parts: torch.Tensor,
    num_frames: torch.Tensor | None,
    num_spatial_parts: torch.Tensor | None,
    layout_pose_pred: torch.Tensor | None,
    *,
    latent_weight: float,
    surface_weight: float,
    scale_weight: float,
    accel_weight: float,
    num_surface_points: int,
    decoder_num_chunks: int,
    decoder_dtype: torch.dtype,
    fallback_consecutive: bool,
    skip_first_spatial_parts: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = clean_latents.sum() * 0.0
    terms = {
        "latent": zero.detach(),
        "surface": zero.detach(),
        "scale": zero.detach(),
        "accel": zero.detach(),
        "pairs": torch.tensor(0, device=clean_latents.device, dtype=torch.long),
        "objects": torch.tensor(0, device=clean_latents.device, dtype=torch.long),
    }
    if latent_weight <= 0.0 and surface_weight <= 0.0 and scale_weight <= 0.0 and accel_weight <= 0.0:
        return zero, terms

    pairs, triples, valid_objects = _temporal_grid_pairs(
        num_parts,
        num_frames,
        num_spatial_parts,
        fallback_consecutive=fallback_consecutive,
        skip_first_spatial_parts=skip_first_spatial_parts,
    )
    terms["pairs"] = torch.tensor(len(pairs), device=clean_latents.device, dtype=torch.long)
    terms["objects"] = torch.tensor(valid_objects, device=clean_latents.device, dtype=torch.long)
    if not pairs:
        return zero, terms

    pair_idx = torch.tensor(pairs, device=clean_latents.device, dtype=torch.long)
    src_idx, dst_idx = pair_idx[:, 0], pair_idx[:, 1]
    total = zero

    if latent_weight > 0.0:
        latent_loss = (
            clean_latents[src_idx].float().view(len(pairs), -1)
            - clean_latents[dst_idx].float().view(len(pairs), -1)
        ).pow(2).mean()
        terms["latent"] = latent_loss.detach()
        total = total + float(latent_weight) * latent_loss

    if surface_weight > 0.0:
        src_surfaces = part_surfaces[src_idx]
        dst_surfaces = part_surfaces[dst_idx]
        src_to_dst_points = _sample_bbox_aligned_surface_queries(src_surfaces, dst_surfaces, num_surface_points)
        dst_to_src_points = _sample_bbox_aligned_surface_queries(dst_surfaces, src_surfaces, num_surface_points)
        dst_field = vae.decode(
            clean_latents[dst_idx].to(dtype=decoder_dtype),
            sampled_points=src_to_dst_points.to(dtype=decoder_dtype),
            num_chunks=int(decoder_num_chunks),
        ).sample.float()
        src_field = vae.decode(
            clean_latents[src_idx].to(dtype=decoder_dtype),
            sampled_points=dst_to_src_points.to(dtype=decoder_dtype),
            num_chunks=int(decoder_num_chunks),
        ).sample.float()
        surface_loss = 0.5 * (
            dst_field.abs().mean() + dst_field.pow(2).mean()
            + src_field.abs().mean() + src_field.pow(2).mean()
        )
        terms["surface"] = surface_loss.detach()
        total = total + float(surface_weight) * surface_loss

    if scale_weight > 0.0 and layout_pose_pred is not None:
        scale_loss = (layout_pose_pred[src_idx, 3:].float() - layout_pose_pred[dst_idx, 3:].float()).pow(2).mean()
        terms["scale"] = scale_loss.detach()
        total = total + float(scale_weight) * scale_loss

    if accel_weight > 0.0 and triples:
        triple_idx = torch.tensor(triples, device=clean_latents.device, dtype=torch.long)
        prev_latents = clean_latents[triple_idx[:, 0]].float().view(len(triples), -1)
        cur_latents = clean_latents[triple_idx[:, 1]].float().view(len(triples), -1)
        next_latents = clean_latents[triple_idx[:, 2]].float().view(len(triples), -1)
        accel_loss = (next_latents - 2.0 * cur_latents + prev_latents).pow(2).mean()
        terms["accel"] = accel_loss.detach()
        total = total + float(accel_weight) * accel_loss

    return total, terms


def _compute_geometry_field_auxiliary_loss(
    vae: TripoSGVAEModel,
    clean_latents: torch.Tensor,
    part_surfaces: torch.Tensor,
    num_points: int,
    sdf_weight: float,
    normal_weight: float,
    eikonal_weight: float,
    second_order: bool,
    eikonal_grad_norm_clip: float,
    decoder_num_chunks: int,
    decoder_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Important: COM4D point-sampling files contain only surface points and
    # surface normals. They do not contain off-surface signed-distance/TSDF
    # samples. The "sdf" term below is therefore only a surface-zero
    # implicit-field constraint D(x_surface)=0, not TripoSG's full SDF loss.
    if sdf_weight <= 0.0 and normal_weight <= 0.0 and eikonal_weight <= 0.0:
        zero = clean_latents.sum() * 0.0
        return zero, {
            "sdf": zero.detach(),
            "normal": zero.detach(),
            "eikonal": zero.detach(),
        }

    query_points, target_normals = _sample_surface_queries(part_surfaces, num_points)
    query_points = query_points.to(device=clean_latents.device, dtype=torch.float32)
    target_normals = target_normals.to(device=clean_latents.device, dtype=torch.float32)
    use_second_order_terms = bool(second_order) and (normal_weight > 0.0 or eikonal_weight > 0.0)
    query_points.requires_grad_(use_second_order_terms)

    if use_second_order_terms:
        # Surface normal/eikonal supervision needs second-order gradients through
        # the decoder attention path. CUDA flash SDPA does not implement that
        # derivative, so use math SDPA only for this small auxiliary decode.
        sdp_context = (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
            if clean_latents.is_cuda
            else nullcontext()
        )
        with sdp_context:
            field_values = vae.decode(
                clean_latents.to(dtype=decoder_dtype),
                sampled_points=query_points.to(dtype=decoder_dtype),
                num_chunks=int(decoder_num_chunks),
            ).sample.float()
    else:
        field_values = vae.decode(
            clean_latents.to(dtype=decoder_dtype),
            sampled_points=query_points.to(dtype=decoder_dtype),
            num_chunks=int(decoder_num_chunks),
        ).sample.float()

    surface_zero_loss = field_values.abs().mean() + field_values.pow(2).mean()
    normal_loss = field_values.sum() * 0.0
    eikonal_loss = field_values.sum() * 0.0

    if use_second_order_terms:
        gradients = torch.autograd.grad(
            outputs=field_values.sum(),
            inputs=query_points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_norm = gradients.norm(dim=-1).clamp_min(1e-6)
        if normal_weight > 0.0:
            pred_normals = gradients / grad_norm[..., None]
            normal_loss = (1.0 - (pred_normals * target_normals).sum(dim=-1)).mean()
        if eikonal_weight > 0.0:
            if eikonal_grad_norm_clip > 0.0:
                grad_norm = grad_norm.clamp(max=float(eikonal_grad_norm_clip))
            eikonal_loss = (grad_norm - 1.0).pow(2).mean()

    total = (
        float(sdf_weight) * surface_zero_loss
        + float(normal_weight) * normal_loss
        + float(eikonal_weight) * eikonal_loss
    )
    return total, {
        "sdf": surface_zero_loss.detach(),
        "normal": normal_loss.detach(),
        "eikonal": eikonal_loss.detach(),
    }


def _unproject_depth_samples(
    pixels_yx: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = depth[pixels_yx[:, 0], pixels_yx[:, 1]].float()
    return _unproject_pixels_with_depth_values(pixels_yx, z, intrinsics, camera_to_world)


def _unproject_pixels_with_depth_values(
    pixels_yx: torch.Tensor,
    depth_values: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = pixels_yx[:, 0].float()
    x = pixels_yx[:, 1].float()
    z = depth_values.float()
    fx = intrinsics[0, 0].float().clamp_min(1e-6)
    fy = intrinsics[1, 1].float().clamp_min(1e-6)
    cx = intrinsics[0, 2].float()
    cy = intrinsics[1, 2].float()
    # 3D-FRONT render metadata uses the Blender/OpenGL camera convention:
    # camera looks down -Z while EXR depth is stored as a positive distance.
    camera_points = torch.stack(
        [
            (x - cx) / fx * z,
            (y - cy) / fy * z,
            -z,
            torch.ones_like(z),
        ],
        dim=-1,
    )
    world_h = camera_points @ camera_to_world.float().T
    origin = camera_to_world[:3, 3].float()
    points = world_h[:, :3]
    ray_dirs = tF.normalize(points - origin[None], dim=-1, eps=1e-6)
    return points, ray_dirs


def _decode_scene_sdf(
    vae: TripoSGVAEModel,
    object_latents: torch.Tensor,
    query_points: torch.Tensor,
    decoder_num_chunks: int,
    decoder_dtype: torch.dtype,
) -> torch.Tensor:
    if object_latents.numel() == 0 or query_points.numel() == 0:
        return query_points.new_zeros((object_latents.shape[0], query_points.shape[0]))
    num_objects, num_points = object_latents.shape[0], query_points.shape[0]
    points = query_points[None].expand(num_objects, num_points, 3)
    values = vae.decode(
        object_latents.to(dtype=decoder_dtype),
        sampled_points=points.to(dtype=decoder_dtype),
        num_chunks=int(decoder_num_chunks),
    ).sample.float()
    return values.reshape(num_objects, num_points)


def _compute_geometry_image_auxiliary_loss(
    vae: TripoSGVAEModel,
    clean_latents: torch.Tensor,
    geometry_image: Optional[dict[str, torch.Tensor]],
    num_parts: torch.Tensor,
    num_surface_pixels: int,
    num_empty_pixels: int,
    depth_weight: float,
    normal_weight: float,
    mask_weight: float,
    empty_weight: float,
    surface_margin: float,
    empty_margin: float,
    max_empty_backprop_pixels: int,
    ray_sign_weight: float,
    ray_sign_epsilon: float,
    decoder_num_chunks: int,
    decoder_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = clean_latents.sum() * 0.0
    terms = {
        "depth": zero.detach(),
        "normal": zero.detach(),
        "mask": zero.detach(),
        "empty": zero.detach(),
        "valid": torch.tensor(0, device=clean_latents.device, dtype=torch.long),
    }
    if geometry_image is None or depth_weight <= 0.0 and normal_weight <= 0.0 and mask_weight <= 0.0 and empty_weight <= 0.0:
        return zero, terms
    required = {"mask", "intrinsics", "camera_to_world", "world_to_sdf_center", "world_to_sdf_scale"}
    if not required.issubset(geometry_image.keys()):
        return zero, terms

    masks = geometry_image["mask"].to(device=clean_latents.device, dtype=torch.bool)
    depth_maps = geometry_image.get("depth")
    normal_maps = geometry_image.get("normal")
    intrinsics = geometry_image["intrinsics"].to(device=clean_latents.device, dtype=torch.float32)
    camera_to_world = geometry_image["camera_to_world"].to(device=clean_latents.device, dtype=torch.float32)
    world_to_sdf_center = geometry_image["world_to_sdf_center"].to(device=clean_latents.device, dtype=torch.float32)
    world_to_sdf_scale = geometry_image["world_to_sdf_scale"].to(device=clean_latents.device, dtype=torch.float32)
    if depth_maps is None:
        return zero, terms
    depth_maps = depth_maps.to(device=clean_latents.device, dtype=torch.float32)
    if normal_maps is not None:
        normal_maps = normal_maps.to(device=clean_latents.device, dtype=torch.float32)

    depth_losses = []
    normal_losses = []
    mask_losses = []
    empty_losses = []
    ptr = 0
    for obj_idx, count_tensor in enumerate(num_parts.detach().cpu()):
        count = int(count_tensor.item())
        object_latents = clean_latents[ptr:ptr + count]
        ptr += count
        if count <= 0 or obj_idx >= masks.shape[0]:
            continue
        valid_yx = masks[obj_idx].nonzero(as_tuple=False)
        if valid_yx.numel() == 0:
            continue
        sample_count = min(int(num_surface_pixels), valid_yx.shape[0])
        perm = torch.randperm(valid_yx.shape[0], device=valid_yx.device)[:sample_count]
        surface_yx = valid_yx[perm]
        query_points, ray_dirs = _unproject_depth_samples(
            surface_yx,
            depth_maps[obj_idx],
            intrinsics[obj_idx],
            camera_to_world[obj_idx],
        )
        query_points = (query_points - world_to_sdf_center[obj_idx][None]) * world_to_sdf_scale[obj_idx].reshape(1, 1)
        point_valid = torch.isfinite(query_points).all(dim=-1) & (query_points.norm(dim=-1) <= 4.0)
        if not bool(point_valid.any()):
            continue
        surface_yx = surface_yx[point_valid]
        query_points = query_points[point_valid].detach().float()
        ray_dirs = ray_dirs[point_valid]
        use_normals = normal_weight > 0.0 and normal_maps is not None
        query_points.requires_grad_(use_normals)

        sdp_context = (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
            if clean_latents.is_cuda and use_normals
            else nullcontext()
        )
        with sdp_context:
            scene_fields = _decode_scene_sdf(
                vae,
                object_latents,
                query_points,
                decoder_num_chunks=decoder_num_chunks,
                decoder_dtype=decoder_dtype,
            )
        abs_fields = scene_fields.abs()
        min_abs, nearest_idx = abs_fields.min(dim=0)
        selected_sdf = scene_fields.gather(0, nearest_idx[None]).squeeze(0)

        if depth_weight > 0.0:
            surface_loss = tF.relu(min_abs - float(surface_margin)).mean()
            if ray_sign_weight > 0.0:
                eps = float(ray_sign_epsilon)
                front = (query_points.detach() - ray_dirs * eps).float()
                back = (query_points.detach() + ray_dirs * eps).float()
                front_sdf = _decode_scene_sdf(vae, object_latents, front, decoder_num_chunks, decoder_dtype).min(dim=0).values
                back_sdf = _decode_scene_sdf(vae, object_latents, back, decoder_num_chunks, decoder_dtype).min(dim=0).values
                sign_loss = tF.softplus(-front_sdf).mean() + tF.softplus(back_sdf).mean()
                surface_loss = surface_loss + float(ray_sign_weight) * sign_loss
            depth_losses.append(surface_loss)

        if mask_weight > 0.0:
            mask_losses.append(tF.softplus(min_abs - float(surface_margin)).mean())

        if use_normals:
            gradients = torch.autograd.grad(
                outputs=selected_sdf.sum(),
                inputs=query_points,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            pred_normals = tF.normalize(gradients, dim=-1, eps=1e-6)
            target_normals = tF.normalize(normal_maps[obj_idx, surface_yx[:, 0], surface_yx[:, 1]], dim=-1, eps=1e-6)
            normal_losses.append((1.0 - (pred_normals * target_normals).sum(dim=-1)).mean())

        if empty_weight > 0.0 and int(num_empty_pixels) > 0:
            empty_yx = (~masks[obj_idx]).nonzero(as_tuple=False)
            if empty_yx.numel() > 0:
                empty_count = min(int(num_empty_pixels), empty_yx.shape[0])
                empty_perm = torch.randperm(empty_yx.shape[0], device=empty_yx.device)[:empty_count]
                empty_sample_yx = empty_yx[empty_perm]
                median_depth = depth_maps[obj_idx][masks[obj_idx]].median().clamp_min(1e-4)
                empty_depth = torch.full(
                    (empty_sample_yx.shape[0],),
                    float(median_depth.item()),
                    device=empty_sample_yx.device,
                    dtype=torch.float32,
                )
                empty_points, _ = _unproject_pixels_with_depth_values(
                    empty_sample_yx,
                    empty_depth,
                    intrinsics[obj_idx],
                    camera_to_world[obj_idx],
                )
                empty_points = (empty_points - world_to_sdf_center[obj_idx][None]) * world_to_sdf_scale[obj_idx].reshape(1, 1)
                empty_valid = torch.isfinite(empty_points).all(dim=-1) & (empty_points.norm(dim=-1) <= 4.0)
                if not bool(empty_valid.any()):
                    continue
                empty_points = empty_points[empty_valid]

                # Empty-space supervision is a sparse collision-style penalty:
                # most sampled background points already have zero loss. Probe
                # without building a graph, then backpropagate only through the
                # hardest violating points to keep DDP ranks from diverging in
                # backward cost on large mixed batches.
                with torch.no_grad():
                    empty_probe_fields = _decode_scene_sdf(
                        vae,
                        object_latents.detach(),
                        empty_points.detach().float(),
                        decoder_num_chunks=decoder_num_chunks,
                        decoder_dtype=decoder_dtype,
                    )
                    empty_probe_min_abs = empty_probe_fields.abs().min(dim=0).values
                    empty_violation = float(empty_margin) - empty_probe_min_abs
                    violating = empty_violation > 0.0
                    if not bool(violating.any()):
                        continue
                    hard_count = min(
                        max(int(max_empty_backprop_pixels), 1),
                        int(violating.long().sum().item()),
                    )
                    hard_local = torch.topk(empty_violation[violating], k=hard_count).indices
                    hard_indices = violating.nonzero(as_tuple=False).squeeze(1)[hard_local]
                empty_points = empty_points[hard_indices]
                empty_fields = _decode_scene_sdf(
                    vae,
                    object_latents,
                    empty_points.detach().float(),
                    decoder_num_chunks=decoder_num_chunks,
                    decoder_dtype=decoder_dtype,
                )
                empty_min_abs = empty_fields.abs().min(dim=0).values
                empty_losses.append(tF.relu(float(empty_margin) - empty_min_abs).mean())

    def _mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero

    depth_loss = _mean_or_zero(depth_losses)
    normal_loss = _mean_or_zero(normal_losses)
    mask_loss = _mean_or_zero(mask_losses)
    empty_loss = _mean_or_zero(empty_losses)
    total = (
        float(depth_weight) * depth_loss
        + float(normal_weight) * normal_loss
        + float(mask_weight) * mask_loss
        + float(empty_weight) * empty_loss
    )
    return total, {
        "depth": depth_loss.detach(),
        "normal": normal_loss.detach(),
        "mask": mask_loss.detach(),
        "empty": empty_loss.detach(),
        "valid": torch.tensor(len(depth_losses), device=clean_latents.device, dtype=torch.long),
    }


def main():
    PROJECT_NAME = "COM4D_1"

    parser = argparse.ArgumentParser(
        description="Train a diffusion model for 3D object generation",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=None,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=2,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches each DataLoader worker prefetches. Only used when num_workers > 0."
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs. MultiEpochsDataLoader already keeps one iterator alive, but this makes the intent explicit."
    )
    parser.add_argument(
        "--dataset_cache_root",
        type=str,
        default=None,
        help="Optional local cache root for dataset files."
    )
    parser.add_argument(
        "--dataset_source_root",
        type=str,
        default="/mnt/mocap_b/work/com4d/datasets",
        help="Dataset source root mirrored under --dataset_cache_root."
    )
    parser.add_argument(
        "--dataset_cache_prefetch_window",
        type=int,
        default=16,
        help="Number of upcoming dataset entries to stage in background per DataLoader worker."
    )
    parser.add_argument(
        "--dataset_cache_prefetch_workers",
        type=int,
        default=2,
        help="Background copy threads per DataLoader worker for local dataset staging."
    )
    parser.add_argument(
        "--dataset_cache_max_gb",
        type=float,
        default=300.0,
        help="Approximate local cache size cap. Set 0 to disable cache eviction."
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for training"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--val_guidance_scales",
        type=list,
        nargs="+",
        default=[7.0],
        help="CFG scale used for validation"
    )
    parser.add_argument(
        "--val_only_rank0",
        action="store_true",
        help="Run validation only on the main process and make other ranks wait"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )

    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Train from scratch"
    )
    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Tag of a pretrained PartFrameCrafterDiTModel in this project"
    )
    parser.add_argument(
        "--load_pretrained_model_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained PartFrameCrafterDiTModel checkpoint"
    )
    parser.add_argument(
        "--load_layout_pose_aux_head",
        type=str,
        default=None,
        help="Optional layout_pose_aux_head.pt used to initialize the layout/pose auxiliary head",
    )
    parser.add_argument(
        "--load_room_layout_aux_head",
        type=str,
        default=None,
        help="Optional room_layout_aux_head.pt used to initialize the room-layout auxiliary head",
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()
    if args.load_pretrained_model_ckpt is not None and args.load_pretrained_model_ckpt < 0:
        args.load_pretrained_model_ckpt = None
    # Parse the config file
    configs = get_configs(args.config, extras)  # change yaml configs by `extras`
    dataset_cache_enabled = args.dataset_cache_root is not None
    configure_dataset_cache(
        args.dataset_source_root,
        args.dataset_cache_root,
        enabled=dataset_cache_enabled,
        prefetch_window=args.dataset_cache_prefetch_window,
        prefetch_workers=args.dataset_cache_prefetch_workers,
        max_cache_gb=args.dataset_cache_max_gb,
    )
    configs["dataset_cache"] = {
        "enabled": dataset_cache_enabled,
        "source_root": args.dataset_source_root,
        "cache_root": args.dataset_cache_root,
        "prefetch_window": args.dataset_cache_prefetch_window if dataset_cache_enabled else 0,
        "prefetch_workers": args.dataset_cache_prefetch_workers,
        "max_gb": args.dataset_cache_max_gb,
    }

    args.val_guidance_scales = [float(x[0]) if isinstance(x, list) else float(x) for x in args.val_guidance_scales]
    if args.max_val_steps > 0: 
        # If enable validation, the max_val_steps must be a multiple of nrow
        # Always keep validation batchsize 1
        divider = configs["val"]["nrow"]
        args.max_val_steps = max(args.max_val_steps, divider)
        if args.max_val_steps % divider != 0:
            args.max_val_steps = (args.max_val_steps // divider + 1) * divider

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
        
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    eval_dir = os.path.join(exp_dir, "evaluations")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    # Initialize the accelerator
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ddp_timeout_minutes = int(os.environ.get("COM4D_DDP_TIMEOUT_MINUTES", "180"))
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=ddp_timeout_minutes))
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[process_group_kwargs],
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)
    logger.info(f"Accelerator state:\n{accelerator.state}\n")
    logger.info(
        f"Distributed process group timeout: [{ddp_timeout_minutes}] minutes\n"
    )
    trace_sequence_parallel_event(
        "startup.after_accelerator",
        world_size=accelerator.num_processes,
        process_index=accelerator.process_index,
        local_process_index=accelerator.local_process_index,
        distributed_type=accelerator.distributed_type,
        device=accelerator.device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Build two dataset configs by overriding the 'dataset' key
    cfgs_3d = copy.deepcopy(configs)
    cfgs_4d = copy.deepcopy(configs)
    cfgs_physics = None
    if 'dataset_3d' in configs:
        cfgs_3d['dataset'] = copy.deepcopy(configs['dataset_3d'])
    if 'dataset_4d' in configs:
        cfgs_4d['dataset'] = copy.deepcopy(configs['dataset_4d'])
    physics_dataset_specs = []
    if 'dataset_physics' in configs:
        cfgs_physics = copy.deepcopy(configs)
        cfgs_physics['dataset'] = copy.deepcopy(configs['dataset_physics'])

        def _config_list(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple, ListConfig)):
                return list(value)
            text = str(value).strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            if "," in text:
                return [item.strip() for item in text.split(",") if item.strip()]
            return [text]

        physics_dataset_jsons = _config_list(configs["train"].get("physics_dataset_jsons", None))
        if not physics_dataset_jsons:
            physics_dataset_jsons = _config_list(configs["train"].get("physics_dataset_json", HUMOTO_PHYSICS_DATASET_JSON))
        def _optional_int(value):
            if value is None:
                return None
            text = str(value).strip().lower()
            if text in {"", "none", "null", "auto"}:
                return None
            return int(value)

        def _bool_value(value):
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
            raise ValueError(f"Expected a boolean value, got {value!r}")

        physics_dataset_num_spatial_parts = _config_list(configs["train"].get("physics_dataset_num_spatial_parts", None))
        if physics_dataset_num_spatial_parts:
            physics_dataset_num_spatial_parts = [_optional_int(num_parts) for num_parts in physics_dataset_num_spatial_parts]
            if len(physics_dataset_num_spatial_parts) != len(physics_dataset_jsons):
                raise ValueError(
                    "train.physics_dataset_num_spatial_parts must have the same length as "
                    "train.physics_dataset_jsons."
                )
        else:
            physics_dataset_num_spatial_parts = [cfgs_physics['dataset'].get('num_spatial_parts', None)] * len(physics_dataset_jsons)

        physics_dataset_spatiotemporal_grid = _config_list(configs["train"].get("physics_dataset_spatiotemporal_grid", None))

        physics_dataset_surface_num_points = _config_list(
            configs["train"].get("physics_dataset_surface_num_points", None)
        )
        if physics_dataset_surface_num_points:
            physics_dataset_surface_num_points = [int(value) for value in physics_dataset_surface_num_points]
            if len(physics_dataset_surface_num_points) != len(physics_dataset_jsons):
                raise ValueError(
                    "train.physics_dataset_surface_num_points must have the same length as "
                    "train.physics_dataset_jsons."
                )
        else:
            physics_dataset_surface_num_points = [
                cfgs_physics['dataset'].get(
                    'surface_num_points',
                    configs["train"].get("surface_num_points", 204800),
                )
            ] * len(physics_dataset_jsons)
        if physics_dataset_spatiotemporal_grid:
            physics_dataset_spatiotemporal_grid = [_bool_value(value) for value in physics_dataset_spatiotemporal_grid]
            if len(physics_dataset_spatiotemporal_grid) != len(physics_dataset_jsons):
                raise ValueError(
                    "train.physics_dataset_spatiotemporal_grid must have the same length as "
                    "train.physics_dataset_jsons."
                )
        else:
            physics_dataset_spatiotemporal_grid = [True] * len(physics_dataset_jsons)

        physics_dataset_probs = _config_list(configs["train"].get("physics_dataset_probs", None))
        if physics_dataset_probs:
            physics_dataset_probs = [float(prob) for prob in physics_dataset_probs]
            if len(physics_dataset_probs) != len(physics_dataset_jsons):
                raise ValueError(
                    "train.physics_dataset_probs must have the same length as "
                    "train.physics_dataset_jsons."
                )
        else:
            physics_dataset_probs = [1.0 / len(physics_dataset_jsons)] * len(physics_dataset_jsons)
        prob_sum = float(sum(physics_dataset_probs))
        if prob_sum <= 0.0:
            raise ValueError("train.physics_dataset_probs must sum to a positive value.")
        physics_dataset_probs = [float(prob) / prob_sum for prob in physics_dataset_probs]

        for source_idx, (
            physics_dataset_json,
            source_prob,
            source_surface_num_points,
            source_num_spatial_parts,
            source_spatiotemporal_grid,
        ) in enumerate(
            zip(
                physics_dataset_jsons,
                physics_dataset_probs,
                physics_dataset_surface_num_points,
                physics_dataset_num_spatial_parts,
                physics_dataset_spatiotemporal_grid,
            )
        ):
            physics_dataset_path = Path(str(physics_dataset_json)).expanduser()
            if not physics_dataset_path.is_file():
                raise FileNotFoundError(
                    f"Physics dataset JSON does not exist: {physics_dataset_path}. "
                    "Set train.physics_dataset_json or train.physics_dataset_jsons to existing dataset_json/*.json files."
                )
            source_cfgs_physics = copy.deepcopy(cfgs_physics)
            source_cfgs_physics['dataset']['surface_num_points'] = int(source_surface_num_points)
            source_cfgs_physics['dataset']['config'] = [str(physics_dataset_path)]
            source_cfgs_physics['dataset']['spatiotemporal_grid'] = bool(source_spatiotemporal_grid)
            if source_spatiotemporal_grid:
                if source_num_spatial_parts is not None:
                    source_cfgs_physics['dataset']['num_spatial_parts'] = int(source_num_spatial_parts)
                else:
                    source_cfgs_physics['dataset'].pop('num_spatial_parts', None)
            else:
                source_cfgs_physics['dataset'].pop('num_spatial_parts', None)
            physics_dataset_specs.append(
                {
                    "index": source_idx,
                    "path": str(physics_dataset_path),
                    "prob": float(source_prob),
                    "spatiotemporal_grid": bool(source_spatiotemporal_grid),
                    "num_spatial_parts": source_num_spatial_parts,
                    "configs": source_cfgs_physics,
                    "surface_num_points": int(source_surface_num_points),
                }
            )

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers

    def prewarm_dataset_cache_streams(streams: list[tuple[str, torch.utils.data.Dataset]]) -> dict[str, int]:
        if not accelerator.is_main_process or not dataset_cache_enabled or args.dataset_cache_prefetch_window <= 0:
            return {}
        entries_per_stream = configs["train"]["batch_size_per_gpu"] + args.dataset_cache_prefetch_window
        prewarmed: dict[str, int] = {}
        for name, dataset in streams:
            data_configs = getattr(dataset, "data_configs", None)
            if not data_configs:
                continue
            count = min(len(data_configs), entries_per_stream)
            prefetch_data_configs(data_configs, 0, count)
            prewarmed[name] = count
        return prewarmed

    # Train/Val: 3D
    train_dataset_3d = BatchedObjaversePartDataset3D(
        configs=cfgs_3d,
        batch_size=configs["train"]["batch_size_per_gpu"],
        is_main_process=accelerator.is_main_process,
        shuffle=True,
        training=True,
    )
    val_dataset_3d = ObjaversePartDataset3D(
        configs=cfgs_3d,
        training=False
    )
    train_loader_3d = MultiEpochsDataLoader(
        train_dataset_3d,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset_3d.collate_fn,
        **loader_kwargs,
    )
    val_loader_3d = MultiEpochsDataLoader(
        val_dataset_3d,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        **loader_kwargs,
    )
    random_val_loader_3d = MultiEpochsDataLoader(
        val_dataset_3d,
        batch_size=configs["val"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        **loader_kwargs,
    )

    # Train/Val: 4D (frames as parts)
    train_dataset_4d = BatchedObjaversePartDataset4D(
        configs=cfgs_4d,
        batch_size=configs["train"]["batch_size_per_gpu"],
        is_main_process=accelerator.is_main_process,
        shuffle=True,
        training=True,
    )
    val_dataset_4d = ObjaversePartDataset4D(
        configs=cfgs_4d,
        training=False
    )
    train_loader_4d = MultiEpochsDataLoader(
        train_dataset_4d,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset_4d.collate_fn,
        **loader_kwargs,
    )
    val_loader_4d = MultiEpochsDataLoader(
        val_dataset_4d,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        **loader_kwargs,
    )
    random_val_loader_4d = MultiEpochsDataLoader(
        val_dataset_4d,
        batch_size=configs["val"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        **loader_kwargs,
    )

    train_dataset_physics = None
    train_loader_physics = None
    train_datasets_physics = []
    train_loaders_physics = []
    physics_data_prob = float(configs["train"].get("physics_data_prob", 0.0))
    if physics_dataset_specs and physics_data_prob > 0.0:
        for spec in physics_dataset_specs:
            train_dataset_physics_source = BatchedObjaversePartDataset4D(
                configs=spec["configs"],
                batch_size=configs["train"]["batch_size_per_gpu"],
                is_main_process=accelerator.is_main_process,
                shuffle=True,
                training=True,
            )
            train_loader_physics_source = MultiEpochsDataLoader(
                train_dataset_physics_source,
                batch_size=configs["train"]["batch_size_per_gpu"],
                num_workers=args.num_workers,
                drop_last=True,
                pin_memory=args.pin_memory,
                collate_fn=train_dataset_physics_source.collate_fn,
                **loader_kwargs,
            )
            if len(train_dataset_physics_source) == 0 or len(train_loader_physics_source) == 0:
                raise RuntimeError(
                    "Physics data mixing is enabled, but a batched physics dataset is empty. "
                    f"source={spec['path']}. Generate more physics sequences, lower "
                    "train.batch_size_per_gpu, or lower dataset_physics.min_num_parts/max_num_parts."
                )
            train_datasets_physics.append(train_dataset_physics_source)
            train_loaders_physics.append(train_loader_physics_source)
        train_dataset_physics = train_datasets_physics[0] if train_datasets_physics else None
        train_loader_physics = train_loaders_physics[0] if train_loaders_physics else None

    objaverse_dataset_configs = copy.deepcopy(configs)
    objaverse_dataset_configs['dataset'] = configs['dataset_objaverse']
    objaverse_dataset = BatchedObjaversePartDatasetOriginal(
        configs=objaverse_dataset_configs,
        batch_size=configs["train"]["batch_size_per_gpu"],
        is_main_process=accelerator.is_main_process,
        shuffle=False,
        training=True,
    )
    objaverse_train_loader = MultiEpochsDataLoader(
        objaverse_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=objaverse_dataset.collate_fn,
        **loader_kwargs,
    )

    cache_prewarm_streams = [
        ("3d", train_dataset_3d),
        ("4d", train_dataset_4d),
        ("objaverse", objaverse_dataset),
    ]
    for spec, dataset in zip(physics_dataset_specs, train_datasets_physics):
        cache_prewarm_streams.append((f"physics:{Path(spec['path']).stem}", dataset))
    cache_prewarmed = prewarm_dataset_cache_streams(cache_prewarm_streams)

    logger.info(
        f"Loaded 3D [{len(train_dataset_3d)}] train / [{len(val_dataset_3d)}] val; "
        f"4D [{len(train_dataset_4d)}] train / [{len(val_dataset_4d)}] val; "
        f"Physics [{sum(len(dataset) for dataset in train_datasets_physics)}] train; "
        f"Objaverse [{len(objaverse_dataset)}] train\n"
    )
    if accelerator.is_main_process:
        logger.info(
            f"Physics data mix: prob={physics_data_prob} "
            f"enabled={len(train_loaders_physics) > 0}\n"
        )
        if physics_dataset_specs:
            source_summary = [
                (
                    f"{Path(spec['path']).name}:{spec['prob']:.3f}:len={len(dataset)}:"
                    f"grid={spec['spatiotemporal_grid']}:spatial={spec['num_spatial_parts']}"
                )
                for spec, dataset in zip(physics_dataset_specs, train_datasets_physics)
            ]
            logger.info(f"Physics dataset sources: {source_summary}\n")
        logger.info(
            f"Dataset cache: enabled={dataset_cache_enabled} "
            f"source_root={args.dataset_source_root} cache_root={args.dataset_cache_root} "
            f"prefetch_window={args.dataset_cache_prefetch_window if dataset_cache_enabled else 0} "
            f"max_gb={args.dataset_cache_max_gb}\n"
        )
        if cache_prewarmed:
            logger.info(
                "Dataset cache prewarm: "
                + ", ".join(f"{name}={count}" for name, count in cache_prewarmed.items())
                + " upcoming entries staged per stream\n"
            )

    single_object_reg_prob = float(configs["train"].get("single_object_regularizer_prob", 0.0))
    single_object_configs = [
        cfg for cfg in getattr(objaverse_dataset, 'data_configs', [])
        if isinstance(cfg, dict) and cfg.get('num_parts', 0) == 1
    ]
    if accelerator.is_main_process:
        logger.info(
            f"Single-object regularizer: prob={single_object_reg_prob} with {len(single_object_configs)} Objaverse candidates\n"
        )

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * \
        accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]
    if accelerator.is_main_process:
        logger.info(
            "Runtime memory settings: "
            f"num_processes={accelerator.num_processes}, "
            f"mixed_precision={accelerator.mixed_precision}, "
            f"batch_size_per_gpu={configs['train']['batch_size_per_gpu']}, "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
            f"total_batch_size={total_batch_size}, "
            f"image_load_size={configs['train'].get('image_load_size', 'unset')}, "
            f"dino_preprocess_size={configs['train'].get('dino_preprocess_size', 'unset')}, "
            f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', 'unset')}, "
            f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'unset')}\n"
        )
    
    # Initialize the model
    logger.info("Initializing the model...")
    vae = TripoSGVAEModel.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="vae"
    )
    feature_extractor_dinov2 = BitImageProcessor.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="feature_extractor_dinov2"
    )
    image_encoder_dinov2 = Dinov2Model.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="image_encoder_dinov2"
    )
    target_image_size = configs["train"].get("dino_preprocess_size", None)
    target_image_size = int(target_image_size) if target_image_size is not None else None
    if accelerator.is_main_process:
        image_load_size = int(configs["train"].get("image_load_size", 512))
        logger.info(
            f"Image sizing - dataloader: {image_load_size}x{image_load_size}, "
            f"DINO preprocess: {target_image_size if target_image_size is not None else 'feature-extractor default'}\n"
        )

    def _normalize_range(raw_range, default_range):
        if raw_range is None:
            low, high = default_range
        else:
            values = list(raw_range)
            if len(values) != 2:
                low, high = default_range
            else:
                low, high = float(values[0]), float(values[1])
        if high < low:
            low, high = high, low
        return float(low), float(high)

    advanced_image_masking_cfg = configs["train"].get("advanced_image_masking", {})
    if not hasattr(advanced_image_masking_cfg, "get"):
        advanced_image_masking_cfg = {}

    advanced_image_masking_enabled = bool(advanced_image_masking_cfg.get("enabled", False))
    advanced_image_masking_mode_cfg = {}
    for mode_key in ["3d", "4d", "single"]:
        mode_cfg = advanced_image_masking_cfg.get(mode_key, {})
        if not hasattr(mode_cfg, "get"):
            mode_cfg = {}
        advanced_image_masking_mode_cfg[mode_key] = {
            "prob": float(mode_cfg.get("prob", 0.0)),
            "mask_size_ratio_range": _normalize_range(mode_cfg.get("mask_size_ratio_range", [0.1, 0.5]), (0.1, 0.5)),
            "mask_frame_propagate_range": _normalize_range(mode_cfg.get("mask_frame_propagate_range", [1.0, 1.0]), (1.0, 1.0)),
        }
        advanced_image_masking_mode_cfg[mode_key]["prob"] = max(
            0.0, min(1.0, advanced_image_masking_mode_cfg[mode_key]["prob"])
        )

    if accelerator.is_main_process and advanced_image_masking_enabled:
        logger.info(
            "Advanced image masking enabled: "
            f"3d={advanced_image_masking_mode_cfg['3d']}, "
            f"4d={advanced_image_masking_mode_cfg['4d']}, "
            f"single={advanced_image_masking_mode_cfg['single']}\n"
        )

    enable_part_embedding = configs["model"]["transformer"].get("enable_part_embedding", True)
    enable_frame_embedding = configs["model"]["transformer"].get("enable_frame_embedding", True)
    enable_local_cross_attn = configs["model"]["transformer"].get("enable_local_cross_attn", True)
    enable_global_cross_attn = configs["model"]["transformer"].get("enable_global_cross_attn", True)
    enable_static_embedding = configs["model"]["transformer"].get("enable_static_embedding", True)
    enable_dynamic_embedding = configs["model"]["transformer"].get("enable_dynamic_embedding", True)
    enable_static_embedding_per_block = configs["model"]["transformer"].get("enable_static_embedding_per_block", False)
    enable_dynamic_embedding_per_block = configs["model"]["transformer"].get("enable_dynamic_embedding_per_block", False)
    enable_instance_type_embedding = configs["model"]["transformer"].get("enable_instance_type_embedding", False)
    enable_object_id_embedding = configs["model"]["transformer"].get("enable_object_id_embedding", False)
    max_object_ids = int(configs["model"]["transformer"].get("max_object_ids", 32))
    enable_camera_time_conditioning = configs["model"]["transformer"].get("enable_camera_time_conditioning", False)
    camera_condition_dim = int(configs["model"]["transformer"].get("camera_condition_dim", 25))
    physics_condition_dim = int(configs["model"]["transformer"].get("physics_condition_dim", 8))
    mixing_mode = str(configs["model"]["transformer"].get("mixing_mode", "current"))
    flash_attention_cfg = configs["model"]["transformer"].get("flash_attention", {}) or {}
    transformer_sdpa_backend = str(flash_attention_cfg.get("backend", "auto")).lower()
    verify_flash_attention_once = bool(flash_attention_cfg.get("verify_once", False))
    valid_sdpa_backends = {"auto", "flash", "mem_efficient", "math"}
    if transformer_sdpa_backend not in valid_sdpa_backends:
        raise ValueError(
            "model.transformer.flash_attention.backend must be one of "
            f"{sorted(valid_sdpa_backends)}, got {transformer_sdpa_backend!r}."
        )
    sp_attention_cfg = configs["model"]["transformer"].get("sequence_parallel_attention", {}) or {}
    enable_sequence_parallel_attention = bool(sp_attention_cfg.get("enabled", False))
    sequence_parallel_replicated_batch = bool(sp_attention_cfg.get("replicated_batch", False))
    sequence_parallel_validate_replicated = bool(sp_attention_cfg.get("validate_replicated", True))
    sequence_parallel_replicate_inputs = bool(sp_attention_cfg.get("replicate_inputs", False))
    if enable_sequence_parallel_attention and not sequence_parallel_replicated_batch:
        raise ValueError(
            "model.transformer.sequence_parallel_attention.enabled=true currently requires "
            "replicated_batch=true. Ordinary DDP batches are not safe for SP attention."
        )
    if sequence_parallel_replicate_inputs and not enable_sequence_parallel_attention:
        raise ValueError("sequence_parallel_attention.replicate_inputs=true requires enabled=true.")

    def _sequence_parallel_active_for_mode(mode_name: str) -> bool:
        # Replicated sequence-parallel attention is only valid for physics/Humoto
        # steps, where grouped interactions are intentional and all ranks follow
        # the same collective path. 4D/3D-Front steps remain ordinary DDP.
        return enable_sequence_parallel_attention and mode_name == "physics"

    # Separate spatial and temporal global-attn block ids; fallback to global_attn_block_ids
    spatial_global_attn_block_ids = configs["model"]["transformer"].get("spatial_global_attn_block_ids", None)
    if spatial_global_attn_block_ids is not None:
        spatial_global_attn_block_ids = list(spatial_global_attn_block_ids)
    temporal_global_attn_block_ids = configs["model"]["transformer"].get("temporal_global_attn_block_ids", None)
    if temporal_global_attn_block_ids is not None:
        temporal_global_attn_block_ids = list(temporal_global_attn_block_ids)
    # Back-compat: if neither provided, use global_attn_block_ids/range
    fallback_global_ids = configs["model"]["transformer"].get("global_attn_block_ids", None)
    if fallback_global_ids is not None:
        fallback_global_ids = list(fallback_global_ids)
    global_attn_block_id_range = configs["model"]["transformer"].get("global_attn_block_id_range", None)
    if global_attn_block_id_range is not None:
        global_attn_block_id_range = list(global_attn_block_id_range)
        if fallback_global_ids is None:
            fallback_global_ids = list(range(global_attn_block_id_range[0], global_attn_block_id_range[1] + 1))
    if spatial_global_attn_block_ids is None:
        spatial_global_attn_block_ids = fallback_global_ids or []
    if temporal_global_attn_block_ids is None:
        temporal_global_attn_block_ids = fallback_global_ids or []
    transformer_init_source = "unknown"
    if args.from_scratch:
        transformer_init_source = (
            "from_scratch:"
            f"{os.path.join(configs['model']['pretrained_model_name_or_path'], 'transformer')}"
        )
        logger.info(f"Initialize PartFrameCrafterDiTModel from scratch\n")
        transformer = PartFrameCrafterDiTModel.from_config(
            os.path.join(
                configs["model"]["pretrained_model_name_or_path"],
                "transformer"
            ), 
            enable_part_embedding=enable_part_embedding,
            enable_frame_embedding=enable_frame_embedding,
            enable_static_embedding=enable_static_embedding,
            enable_dynamic_embedding=enable_dynamic_embedding,
            enable_static_embedding_per_block=enable_static_embedding_per_block,
            enable_dynamic_embedding_per_block=enable_dynamic_embedding_per_block,
            enable_instance_type_embedding=enable_instance_type_embedding,
            enable_object_id_embedding=enable_object_id_embedding,
            max_object_ids=max_object_ids,
            enable_camera_time_conditioning=enable_camera_time_conditioning,
            camera_condition_dim=camera_condition_dim,
            physics_condition_dim=physics_condition_dim,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=spatial_global_attn_block_ids,
            spatial_global_attn_block_ids=spatial_global_attn_block_ids,
            temporal_global_attn_block_ids=temporal_global_attn_block_ids,
            global_attn_block_id_range=None,
            mixing_mode=mixing_mode,
        )
    elif args.load_pretrained_model is None or args.load_pretrained_model_ckpt is None:
        direct_pretrained_dir = None
        if args.load_pretrained_model is not None:
            cand = os.path.join(args.load_pretrained_model, "transformer_ema")
            if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "diffusion_pytorch_model.safetensors")):
                direct_pretrained_dir = cand
            elif os.path.isdir(args.load_pretrained_model) and os.path.exists(
                os.path.join(args.load_pretrained_model, "diffusion_pytorch_model.safetensors")
            ):
                direct_pretrained_dir = args.load_pretrained_model
            elif os.path.isdir(os.path.join(args.load_pretrained_model, "transformer")):
                direct_pretrained_dir = os.path.join(args.load_pretrained_model, "transformer")

        if direct_pretrained_dir is not None:
            transformer_init_source = f"direct_pretrained_dir:{direct_pretrained_dir}"
            logger.info(
                f"Load PartFrameCrafterDiTModel weights directly from [{direct_pretrained_dir}] "
                "without a stage checkpoint iteration.\n"
            )
            transformer, loading_info = PartFrameCrafterDiTModel.from_pretrained(
                direct_pretrained_dir,
                low_cpu_mem_usage=False,
                output_loading_info=True,
                enable_part_embedding=enable_part_embedding,
                enable_frame_embedding=enable_frame_embedding,
                enable_static_embedding=enable_static_embedding,
                enable_dynamic_embedding=enable_dynamic_embedding,
                enable_static_embedding_per_block=enable_static_embedding_per_block,
                enable_dynamic_embedding_per_block=enable_dynamic_embedding_per_block,
                enable_instance_type_embedding=enable_instance_type_embedding,
                enable_object_id_embedding=enable_object_id_embedding,
                max_object_ids=max_object_ids,
                enable_camera_time_conditioning=enable_camera_time_conditioning,
                camera_condition_dim=camera_condition_dim,
                physics_condition_dim=physics_condition_dim,
                enable_local_cross_attn=enable_local_cross_attn,
                enable_global_cross_attn=enable_global_cross_attn,
                global_attn_block_ids=spatial_global_attn_block_ids,
                global_attn_block_id_range=None,
                mixing_mode=mixing_mode,
            )
        else:
            if args.load_pretrained_model is not None and args.load_pretrained_model_ckpt is None:
                logger.info(
                    "Ignoring `--load_pretrained_model` because `--load_pretrained_model_ckpt` < 0 "
                    "and the path does not point to a loadable weights directory. "
                    "Falling back to config pretrained transformer initialization.\n"
                )
            transformer_init_source = (
                "config_pretrained:"
                f"{configs['model']['pretrained_model_name_or_path']}/transformer"
            )
            logger.info(f"Load pretrained TripoSGDiTModel to initialize PartFrameCrafterDiTModel from [{configs['model']['pretrained_model_name_or_path']}]\n")
            transformer, loading_info = PartFrameCrafterDiTModel.from_pretrained(
                configs["model"]["pretrained_model_name_or_path"],
                subfolder="transformer",
                low_cpu_mem_usage=False, 
                output_loading_info=True, 
                enable_part_embedding=enable_part_embedding,
                enable_frame_embedding=enable_frame_embedding,
                enable_static_embedding=enable_static_embedding,
                enable_dynamic_embedding=enable_dynamic_embedding,
                enable_static_embedding_per_block=enable_static_embedding_per_block,
                enable_dynamic_embedding_per_block=enable_dynamic_embedding_per_block,
                enable_instance_type_embedding=enable_instance_type_embedding,
                enable_object_id_embedding=enable_object_id_embedding,
                max_object_ids=max_object_ids,
                enable_camera_time_conditioning=enable_camera_time_conditioning,
                camera_condition_dim=camera_condition_dim,
                physics_condition_dim=physics_condition_dim,
                enable_local_cross_attn=enable_local_cross_attn,
                enable_global_cross_attn=enable_global_cross_attn,
                global_attn_block_ids=spatial_global_attn_block_ids,
                global_attn_block_id_range=None,
                mixing_mode=mixing_mode,
            )
    else:
        transformer_init_source = f"checkpoint_ema:{args.load_pretrained_model}:{args.load_pretrained_model_ckpt:06d}"
        logger.info(f"Load PartFrameCrafterDiTModel EMA checkpoint from [{args.load_pretrained_model}] iteration [{args.load_pretrained_model_ckpt:06d}]\n")
        path = os.path.join(
            args.output_dir,
            args.load_pretrained_model, 
            "checkpoints", 
            f"{args.load_pretrained_model_ckpt:06d}"
        )

        print(f"Loading EMA checkpoint from path: {path}", os.path.exists(path))

        transformer, loading_info = PartFrameCrafterDiTModel.from_pretrained(
            path, 
            subfolder="transformer_ema",
            low_cpu_mem_usage=False, 
            output_loading_info=True, 
            enable_part_embedding=enable_part_embedding,
            enable_frame_embedding=enable_frame_embedding,
            enable_static_embedding=enable_static_embedding,
            enable_dynamic_embedding=enable_dynamic_embedding,
            enable_static_embedding_per_block=enable_static_embedding_per_block,
            enable_dynamic_embedding_per_block=enable_dynamic_embedding_per_block,
            enable_instance_type_embedding=enable_instance_type_embedding,
            enable_object_id_embedding=enable_object_id_embedding,
            max_object_ids=max_object_ids,
            enable_camera_time_conditioning=enable_camera_time_conditioning,
            camera_condition_dim=camera_condition_dim,
            physics_condition_dim=physics_condition_dim,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=spatial_global_attn_block_ids,
            global_attn_block_id_range=None,
            mixing_mode=mixing_mode,
        )
    if not args.from_scratch:
        for v in loading_info.values():
            if v and len(v) > 0:
                logger.info(f"Loading info of PartFrameCrafterDiTModel: {loading_info}\n")
                break
    if accelerator.is_main_process:
        logger.info(
            f"Transformer initialization source: [{transformer_init_source}]\n"
        )

    # Transformer parameters
    transformer.enable_part_embedding = enable_part_embedding
    transformer.enable_frame_embedding = enable_frame_embedding
    transformer.enable_static_embedding = enable_static_embedding
    transformer.enable_dynamic_embedding = enable_dynamic_embedding
    transformer.enable_static_embedding_per_block = enable_static_embedding_per_block
    transformer.enable_dynamic_embedding_per_block = enable_dynamic_embedding_per_block
    transformer.enable_instance_type_embedding = enable_instance_type_embedding
    transformer.enable_object_id_embedding = enable_object_id_embedding
    transformer.max_object_ids = max_object_ids
    transformer.enable_camera_time_conditioning = enable_camera_time_conditioning
    transformer.enable_local_cross_attn = enable_local_cross_attn
    transformer.enable_global_cross_attn = enable_global_cross_attn
    transformer.spatial_global_attn_block_ids = list(spatial_global_attn_block_ids)
    transformer.temporal_global_attn_block_ids = list(temporal_global_attn_block_ids)
    transformer.mixing_mode = mixing_mode

    layout_pose_cfg = configs["train"].get("layout_pose_auxiliary", {})
    if not hasattr(layout_pose_cfg, "get"):
        layout_pose_cfg = {}
    layout_pose_aux_enabled = bool(layout_pose_cfg.get("enabled", False))
    layout_pose_aux_weight = float(layout_pose_cfg.get("weight", 0.0))
    layout_pose_aux_hidden_dim = int(layout_pose_cfg.get("hidden_dim", 256))
    layout_pose_aux_head = None
    if layout_pose_aux_enabled and layout_pose_aux_weight > 0.0:
        layout_pose_aux_head = LayoutPoseAuxiliaryHead(
            in_channels=int(transformer.config.in_channels),
            hidden_dim=layout_pose_aux_hidden_dim,
            out_dim=6,
        )
        logger.info(
            "Layout/pose auxiliary supervision enabled: "
            f"weight={layout_pose_aux_weight}, hidden_dim={layout_pose_aux_hidden_dim}, "
            "target=[relative_part_center(3), log_part_size(3)]\n"
        )
    elif accelerator.is_main_process:
        logger.info("Layout/pose auxiliary supervision disabled.\n")

    room_layout_cfg = configs["train"].get("room_layout_auxiliary", {})
    if not hasattr(room_layout_cfg, "get"):
        room_layout_cfg = {}
    room_layout_aux_enabled = bool(room_layout_cfg.get("enabled", False))
    room_layout_aux_weight = float(room_layout_cfg.get("weight", 0.0))
    room_layout_aux_hidden_dim = int(room_layout_cfg.get("hidden_dim", 256))
    room_layout_aux_head = None
    if room_layout_aux_enabled and room_layout_aux_weight > 0.0:
        room_layout_aux_head = RoomLayoutAuxiliaryHead(
            in_channels=int(transformer.config.in_channels),
            hidden_dim=room_layout_aux_hidden_dim,
            out_dim=12,
        )
        logger.info(
            "Room-layout auxiliary supervision enabled: "
            f"weight={room_layout_aux_weight}, hidden_dim={room_layout_aux_hidden_dim}, "
            "target=[relative_room_center(3), log_room_extent(3), shell_presence(6)]\n"
        )
    elif accelerator.is_main_process:
        logger.info("Room-layout auxiliary supervision disabled.\n")

    geometry_aux_cfg = configs["train"].get("geometry_field_auxiliary", {})
    if not hasattr(geometry_aux_cfg, "get"):
        geometry_aux_cfg = {}
    geometry_aux_enabled = bool(geometry_aux_cfg.get("enabled", False))
    geometry_aux_weight = float(geometry_aux_cfg.get("weight", 0.0))
    geometry_aux_num_points = int(geometry_aux_cfg.get("num_surface_points", 128))
    geometry_aux_sdf_weight = float(geometry_aux_cfg.get("sdf_weight", 1.0))
    geometry_aux_normal_weight = float(geometry_aux_cfg.get("normal_weight", 10.0))
    geometry_aux_eikonal_weight = float(geometry_aux_cfg.get("eikonal_weight", 0.1))
    geometry_aux_second_order = bool(geometry_aux_cfg.get("second_order", False))
    geometry_aux_eikonal_grad_norm_clip = float(geometry_aux_cfg.get("eikonal_grad_norm_clip", 10.0))
    geometry_aux_decoder_num_chunks = int(geometry_aux_cfg.get("decoder_num_chunks", 8192))
    geometry_aux_modes = set(str(mode) for mode in geometry_aux_cfg.get("modes", ["3d", "single"]))
    geometry_aux_active = geometry_aux_enabled and geometry_aux_weight > 0.0
    if geometry_aux_active:
        logger.warning(
            "Geometry-field auxiliary supervision enabled. This is surface-only supervision, "
            "not true TripoSG-style SDF/TSDF supervision: the training .npy files provide "
            "surface_points/surface_normals only, with no off-surface signed-distance targets. "
            f"weight={geometry_aux_weight}, modes={sorted(geometry_aux_modes)}, "
            f"num_surface_points={geometry_aux_num_points}, "
            f"surface_zero(sdf_weight)={geometry_aux_sdf_weight}, normal={geometry_aux_normal_weight}, "
            f"eikonal={geometry_aux_eikonal_weight}, second_order={geometry_aux_second_order}, "
            f"eikonal_grad_norm_clip={geometry_aux_eikonal_grad_norm_clip}. "
            "Recommended default: keep this auxiliary disabled while testing depth supervision; "
            "if used, prefer surface_zero only, with normal/eikonal off. "
            "Normal/eikonal require second-order decoder gradients and are only active when second_order=true.\n"
        )
    elif accelerator.is_main_process:
        logger.info("Geometry-field auxiliary supervision disabled.\n")

    geometry_image_cfg = configs["train"].get("geometry_image_auxiliary", {})
    if not hasattr(geometry_image_cfg, "get"):
        geometry_image_cfg = {}
    geometry_image_aux_enabled = bool(geometry_image_cfg.get("enabled", False))
    geometry_image_aux_weight = float(geometry_image_cfg.get("weight", 0.0))
    geometry_image_aux_modes = set(str(mode) for mode in geometry_image_cfg.get("modes", ["3d"]))
    geometry_image_aux_num_surface_pixels = int(geometry_image_cfg.get("num_surface_pixels", 32))
    geometry_image_aux_num_empty_pixels = int(geometry_image_cfg.get("num_empty_pixels", 32))
    geometry_image_aux_depth_weight = float(geometry_image_cfg.get("depth_weight", 1.0))
    geometry_image_aux_normal_weight = float(geometry_image_cfg.get("normal_weight", 0.0))
    geometry_image_aux_mask_weight = float(geometry_image_cfg.get("mask_weight", 0.0))
    geometry_image_aux_empty_weight = float(geometry_image_cfg.get("empty_weight", 0.0))
    geometry_image_aux_surface_margin = float(geometry_image_cfg.get("surface_margin", 0.0))
    geometry_image_aux_empty_margin = float(geometry_image_cfg.get("empty_margin", 0.05))
    geometry_image_aux_max_empty_backprop_pixels = int(geometry_image_cfg.get("max_empty_backprop_pixels", 8))
    geometry_image_aux_sync_step_only = bool(geometry_image_cfg.get("sync_step_only", True))
    geometry_image_aux_ray_sign_weight = float(geometry_image_cfg.get("ray_sign_weight", 0.0))
    geometry_image_aux_ray_sign_epsilon = float(geometry_image_cfg.get("ray_sign_epsilon", 0.02))
    geometry_image_aux_decoder_num_chunks = int(geometry_image_cfg.get("decoder_num_chunks", 8192))
    geometry_image_aux_active = geometry_image_aux_enabled and geometry_image_aux_weight > 0.0
    if geometry_image_aux_active:
        logger.info(
            "Geometry-image auxiliary supervision enabled: "
            f"weight={geometry_image_aux_weight}, modes={sorted(geometry_image_aux_modes)}, "
            f"surface_pixels={geometry_image_aux_num_surface_pixels}, empty_pixels={geometry_image_aux_num_empty_pixels}, "
            f"depth={geometry_image_aux_depth_weight}, normal={geometry_image_aux_normal_weight}, "
            f"mask={geometry_image_aux_mask_weight}, empty={geometry_image_aux_empty_weight}, "
            f"surface_margin={geometry_image_aux_surface_margin}, empty_margin={geometry_image_aux_empty_margin}, "
            f"max_empty_backprop_pixels={geometry_image_aux_max_empty_backprop_pixels}, "
            f"sync_step_only={geometry_image_aux_sync_step_only}. "
            "This uses 3D-FRONT depth/normal/semantic render targets as differentiable SDF queries; "
            "the pyrender mesh renderer remains validation-only.\n"
        )
    elif accelerator.is_main_process:
        logger.info("Geometry-image auxiliary supervision disabled.\n")

    temporal_geometry_aux_cfg = configs["train"].get("temporal_geometry_auxiliary", {})
    if not hasattr(temporal_geometry_aux_cfg, "get"):
        temporal_geometry_aux_cfg = {}
    temporal_geometry_aux_enabled = bool(temporal_geometry_aux_cfg.get("enabled", False))
    temporal_geometry_aux_modes = set(str(mode) for mode in temporal_geometry_aux_cfg.get("modes", ["physics"]))
    temporal_geometry_aux_latent_weight = float(temporal_geometry_aux_cfg.get("latent_weight", 0.0))
    temporal_geometry_aux_surface_weight = float(temporal_geometry_aux_cfg.get("surface_weight", 0.0))
    temporal_geometry_aux_scale_weight = float(temporal_geometry_aux_cfg.get("scale_weight", 0.0))
    temporal_geometry_aux_accel_weight = float(temporal_geometry_aux_cfg.get("accel_weight", 0.0))
    temporal_geometry_aux_num_surface_points = int(temporal_geometry_aux_cfg.get("num_surface_points", 32))
    temporal_geometry_aux_decoder_num_chunks = int(temporal_geometry_aux_cfg.get("decoder_num_chunks", 8192))
    temporal_geometry_aux_fallback_consecutive = bool(temporal_geometry_aux_cfg.get("fallback_consecutive", False))
    temporal_geometry_aux_skip_first_spatial_parts = int(temporal_geometry_aux_cfg.get("skip_first_spatial_parts", 0))
    temporal_geometry_aux_physics_skip_first_spatial_parts = None
    physics_skip_first_spatial_parts_cfg = temporal_geometry_aux_cfg.get("physics_skip_first_spatial_parts", None)
    if physics_skip_first_spatial_parts_cfg is not None:
        def _int_config_list(value):
            if isinstance(value, (list, tuple, ListConfig)):
                return [int(item) for item in value]
            text = str(value).strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            if "," in text:
                return [int(item.strip()) for item in text.split(",") if item.strip()]
            return [int(text)]

        temporal_geometry_aux_physics_skip_first_spatial_parts = _int_config_list(
            physics_skip_first_spatial_parts_cfg
        )
        if physics_dataset_specs and len(temporal_geometry_aux_physics_skip_first_spatial_parts) != len(physics_dataset_specs):
            raise ValueError(
                "train.temporal_geometry_auxiliary.physics_skip_first_spatial_parts must have the same length as "
                "train.physics_dataset_jsons."
            )
    temporal_geometry_aux_active = temporal_geometry_aux_enabled and (
        temporal_geometry_aux_latent_weight > 0.0
        or temporal_geometry_aux_surface_weight > 0.0
        or temporal_geometry_aux_scale_weight > 0.0
        or temporal_geometry_aux_accel_weight > 0.0
    )
    if temporal_geometry_aux_active:
        logger.info(
            "Temporal-geometry auxiliary supervision enabled: "
            f"modes={sorted(temporal_geometry_aux_modes)}, "
            f"latent_weight={temporal_geometry_aux_latent_weight}, "
            f"surface_weight={temporal_geometry_aux_surface_weight}, "
            f"scale_weight={temporal_geometry_aux_scale_weight}, "
            f"accel_weight={temporal_geometry_aux_accel_weight}, "
            f"num_surface_points={temporal_geometry_aux_num_surface_points}, "
            f"fallback_consecutive={temporal_geometry_aux_fallback_consecutive}, "
            f"skip_first_spatial_parts={temporal_geometry_aux_skip_first_spatial_parts}, "
            f"physics_skip_first_spatial_parts={temporal_geometry_aux_physics_skip_first_spatial_parts}. "
            "Pairs are adjacent same spatial-part frames for frame-major physics grids; "
            "the surface term uses bbox-aligned implicit SDF cross-consistency, so global motion is not penalized.\n"
        )
    elif accelerator.is_main_process:
        logger.info("Temporal-geometry auxiliary supervision disabled.\n")

    layout_pose_aux_init_path = args.load_layout_pose_aux_head or _checkpoint_aux_head_path(
        args.load_pretrained_model,
        args.load_pretrained_model_ckpt,
        "layout_pose_aux_head.pt",
    )
    room_layout_aux_init_path = args.load_room_layout_aux_head or _checkpoint_aux_head_path(
        args.load_pretrained_model,
        args.load_pretrained_model_ckpt,
        "room_layout_aux_head.pt",
    )
    _load_aux_head_state_if_available(layout_pose_aux_head, layout_pose_aux_init_path, "layout/pose", logger)
    _load_aux_head_state_if_available(room_layout_aux_head, room_layout_aux_init_path, "room-layout", logger)

    freeze_transformer_value = configs["train"].get("freeze_transformer", False)
    if isinstance(freeze_transformer_value, str):
        freeze_transformer = freeze_transformer_value.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        freeze_transformer = bool(freeze_transformer_value)
    use_ema_for_transformer = bool(args.use_ema and not freeze_transformer)
    if args.use_ema and freeze_transformer and accelerator.is_main_process:
        logger.info("Transformer EMA disabled because train.freeze_transformer=true.\n")

    noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )

    if use_ema_for_transformer:
        ema_transformer = MyEMAModel(
            transformer.parameters(),
            model_cls=PartFrameCrafterDiTModel,
            model_config=transformer.config,
            **configs["train"]["ema_kwargs"]
        )

    # Freeze VAE and image encoder
    vae.requires_grad_(False)
    image_encoder_dinov2.requires_grad_(False)
    vae.eval()
    image_encoder_dinov2.eval()

    trainable_modules = configs["train"].get("trainable_modules", None)
    if freeze_transformer:
        transformer.requires_grad_(False)
        transformer.eval()
        logger.info("Transformer backbone frozen; training only enabled auxiliary heads/probes.\n")
    elif trainable_modules is None:
        transformer.requires_grad_(True)
    else:
        trainable_module_names = []
        transformer.requires_grad_(False)
        for name, module in transformer.named_modules():
            for module_name in tuple(trainable_modules.split(",")):
                if module_name in name:
                    for params in module.parameters():
                        params.requires_grad = True
                    trainable_module_names.append(name)
        logger.info(f"Trainable parameter names: {trainable_module_names}\n")

    # transformer.enable_xformers_memory_efficient_attention()  # use `tF.scaled_dot_product_attention` instead

    logger.info("Model initialized.\n%s", transformer)

    # Build processor maps and a switch for spatial (3D) vs temporal (4D) global attention
    def _build_attn_processor_map(active_ids: List[int]) -> Dict[str, Any]:
        attn_processor_dict: Dict[str, Any] = {}
        for layer_id in range(transformer.config.num_layers):
            for attn_id in [1, 2]:
                key = f'blocks.{layer_id}.attn{attn_id}.processor'
                if layer_id in (active_ids or []):
                    attn_processor_dict[key] = PartFrameCrafterAttnProcessor()
                else:
                    attn_processor_dict[key] = TripoSGAttnProcessor2_0()
        return attn_processor_dict

    attn_map_spatial = _build_attn_processor_map(spatial_global_attn_block_ids)
    attn_map_temporal = _build_attn_processor_map(temporal_global_attn_block_ids)
    mixed_global_attn_block_ids = sorted(set(spatial_global_attn_block_ids) | set(temporal_global_attn_block_ids))
    attn_map_mixed = _build_attn_processor_map(mixed_global_attn_block_ids)

    def _unwrap_transformer_for_attn() -> PartFrameCrafterDiTModel:
        # Prefer accelerator.unwrap_model (handles DDP/FSDP); otherwise fall back to .module if present.
        try:
            return accelerator.unwrap_model(transformer)
        except Exception:
            return transformer.module if hasattr(transformer, "module") else transformer

    def switch_to_mode(mode: str):
        base_transformer = _unwrap_transformer_for_attn()
        if mode == "3d":
            # set_attn_processor mutates the dict (pops entries); pass a fresh copy
            base_transformer.set_attn_processor(attn_map_spatial.copy())
            base_transformer.global_attn_block_ids = list(spatial_global_attn_block_ids)
        elif mode == "physics":
            base_transformer.set_attn_processor(attn_map_mixed.copy())
            base_transformer.global_attn_block_ids = list(mixed_global_attn_block_ids)
        else:
            base_transformer.set_attn_processor(attn_map_temporal.copy())
            base_transformer.global_attn_block_ids = list(temporal_global_attn_block_ids)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if use_ema_for_transformer:
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                # Save models with explicit subfolders
                for i, model in enumerate(models):
                    unwrapped = accelerator.unwrap_model(model)
                    if isinstance(unwrapped, PartFrameCrafterDiTModel):
                        if not freeze_transformer:
                            unwrapped.save_pretrained(os.path.join(output_dir, "transformer"))
                    elif isinstance(unwrapped, LayoutPoseAuxiliaryHead):
                        torch.save(
                            unwrapped.state_dict(),
                            os.path.join(output_dir, "layout_pose_aux_head.pt"),
                        )
                    elif isinstance(unwrapped, RoomLayoutAuxiliaryHead):
                        torch.save(
                            unwrapped.state_dict(),
                            os.path.join(output_dir, "room_layout_aux_head.pt"),
                        )
                    else:
                        save_pretrained = getattr(unwrapped, "save_pretrained", None)
                        if callable(save_pretrained):
                            save_pretrained(os.path.join(output_dir, f"model_{i}"))

                    # Make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if use_ema_for_transformer:
                load_model = MyEMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), PartFrameCrafterDiTModel)
                ema_transformer.load_state_dict(load_model.state_dict())
                ema_transformer.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # Pop models so that they are not loaded again
                model = models.pop()
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, PartFrameCrafterDiTModel):
                    transformer_dir = os.path.join(input_dir, "transformer")
                    if os.path.isdir(transformer_dir):
                        load_model = PartFrameCrafterDiTModel.from_pretrained(
                            input_dir,
                            subfolder="transformer",
                            enable_part_embedding=enable_part_embedding,
                            enable_frame_embedding=enable_frame_embedding,
                            enable_static_embedding=enable_static_embedding,
                            enable_dynamic_embedding=enable_dynamic_embedding,
                            enable_static_embedding_per_block=enable_static_embedding_per_block,
                            enable_dynamic_embedding_per_block=enable_dynamic_embedding_per_block,
                            enable_instance_type_embedding=enable_instance_type_embedding,
                            enable_object_id_embedding=enable_object_id_embedding,
                            max_object_ids=max_object_ids,
                            enable_camera_time_conditioning=enable_camera_time_conditioning,
                            camera_condition_dim=camera_condition_dim,
                            physics_condition_dim=physics_condition_dim,
                            enable_local_cross_attn=enable_local_cross_attn,
                            enable_global_cross_attn=enable_global_cross_attn,
                            global_attn_block_ids=spatial_global_attn_block_ids,
                            global_attn_block_id_range=None,
                            mixing_mode=mixing_mode,
                        )
                        model.register_to_config(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                elif isinstance(unwrapped, LayoutPoseAuxiliaryHead):
                    aux_path = os.path.join(input_dir, "layout_pose_aux_head.pt")
                    if os.path.exists(aux_path):
                        model.load_state_dict(torch.load(aux_path, map_location="cpu"))
                elif isinstance(unwrapped, RoomLayoutAuxiliaryHead):
                    aux_path = os.path.join(input_dir, "room_layout_aux_head.pt")
                    if os.path.exists(aux_path):
                        model.load_state_dict(torch.load(aux_path, map_location="cpu"))
                else:
                    # Unknown model; skip
                    pass

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if configs["train"]["grad_checkpoint"] and not freeze_transformer:
        transformer.enable_gradient_checkpointing()

    # Initialize the optimizer and learning rate scheduler
    logger.info("Initializing the optimizer and learning rate scheduler...\n")
    name_lr_mult = configs["train"].get("name_lr_mult", None)
    lr_mult = configs["train"].get("lr_mult", 1.0)
    params, params_lr_mult, names_lr_mult = [], [], []
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue
        if name_lr_mult is not None:
            matched_lr_mult = False
            for k in name_lr_mult.split(","):
                if k in name:
                    params_lr_mult.append(param)
                    names_lr_mult.append(name)
                    matched_lr_mult = True
                    break
            if not matched_lr_mult:
                params.append(param)
        else:
            params.append(param)

    if layout_pose_aux_head is not None:
        aux_params = [param for param in layout_pose_aux_head.parameters() if param.requires_grad]
        if aux_params:
            params.extend(aux_params)
    if room_layout_aux_head is not None:
        aux_params = [param for param in room_layout_aux_head.parameters() if param.requires_grad]
        if aux_params:
            params.extend(aux_params)

    total_trainable_tensors = len(params) + len(params_lr_mult)
    if total_trainable_tensors == 0:
        raise RuntimeError(
            "No trainable parameters were found before optimizer creation. "
            "This would make DDP fail because one or more ranks present zero trainable tensors."
        )
    optimizer = get_optimizer(
        params=[
            {"params": params, "lr": configs["optimizer"]["lr"]},
            {"params": params_lr_mult, "lr": configs["optimizer"]["lr"] * lr_mult}
        ],
        **configs["optimizer"]
    )
    if name_lr_mult is not None:
        logger.info(f"Learning rate x [{lr_mult}] parameter names: {names_lr_mult}\n")

    def _summarize_model(module: torch.nn.Module) -> Dict[str, Any]:
        param_tensors = 0
        total_params = 0
        trainable_tensors = 0
        trainable_params = 0
        first_trainable_names = []

        for name, param in module.named_parameters():
            param_tensors += 1
            total_params += param.numel()
            if param.requires_grad:
                trainable_tensors += 1
                trainable_params += param.numel()
                if len(first_trainable_names) < 8:
                    first_trainable_names.append(name)

        return {
            "rank": int(accelerator.process_index),
            "type": type(module).__name__,
            "param_tensors": int(param_tensors),
            "total_params": int(total_params),
            "trainable_tensors": int(trainable_tensors),
            "trainable_params": int(trainable_params),
            "first_trainable_names": first_trainable_names,
        }

    def _summarize_optimizer(optimizer_obj: torch.optim.Optimizer) -> Dict[str, int]:
        opt_param_tensors = 0
        opt_total_params = 0
        for group in optimizer_obj.param_groups:
            opt_param_tensors += len(group["params"])
            opt_total_params += sum(param.numel() for param in group["params"])
        return {
            "opt_param_tensors": int(opt_param_tensors),
            "opt_total_params": int(opt_total_params),
        }

    model_debug = _summarize_model(transformer)
    optimizer_debug = _summarize_optimizer(optimizer)
    print(
        f"[DEBUG before prepare] rank={model_debug['rank']} "
        f"type={model_debug['type']} "
        f"param_tensors={model_debug['param_tensors']} "
        f"total_params={model_debug['total_params']} "
        f"trainable_tensors={model_debug['trainable_tensors']} "
        f"trainable_params={model_debug['trainable_params']}",
        flush=True,
    )
    print(
        f"[DEBUG optimizer] rank={accelerator.process_index} "
        f"opt_param_tensors={optimizer_debug['opt_param_tensors']} "
        f"opt_total_params={optimizer_debug['opt_total_params']}",
        flush=True,
    )

    if (model_debug["trainable_tensors"] == 0 and not freeze_transformer) or optimizer_debug["opt_param_tensors"] == 0:
        raise RuntimeError(
            "Detected an empty trainable model or optimizer parameter list before accelerator.prepare(). "
            f"Rank {accelerator.process_index} summary: model={model_debug}, optimizer={optimizer_debug}"
        )

    # Derive total steps; prefer existing value in configs if provided
    if "total_steps" not in configs["lr_scheduler"] or int(configs["lr_scheduler"]["total_steps"]) <= 0:
        loader_lengths = [len(train_loader_3d), len(train_loader_4d)]
        loader_lengths.extend(len(loader) for loader in train_loaders_physics)
        approx_len = max(1, max(loader_lengths))
        configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
            approx_len // max(1, accelerator.num_processes) / max(1, args.gradient_accumulation_steps)
        )  # only account updated steps
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes  # for lr scheduler setting
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes  # reset for multi-gpu
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes  # reset for multi-gpu

    # Prepare everything with `accelerator`.
    print(
        f"[DEBUG entering prepare] rank={accelerator.process_index}",
        flush=True,
    )
    prepare_start_time = time.perf_counter()
    prepare_items = [
        transformer,
        optimizer,
        lr_scheduler,
        train_loader_3d,
        train_loader_4d,
        val_loader_3d,
        val_loader_4d,
        random_val_loader_3d,
        random_val_loader_4d,
    ]
    prepare_items.extend(train_loaders_physics)
    if layout_pose_aux_head is not None:
        prepare_items.append(layout_pose_aux_head)
    if room_layout_aux_head is not None:
        prepare_items.append(room_layout_aux_head)
    prepared_items = accelerator.prepare(*prepare_items)
    (
        transformer,
        optimizer,
        lr_scheduler,
        train_loader_3d,
        train_loader_4d,
        val_loader_3d,
        val_loader_4d,
        random_val_loader_3d,
        random_val_loader_4d,
        *optional_prepared_loaders,
    ) = prepared_items
    if train_loaders_physics:
        train_loaders_physics = list(optional_prepared_loaders[:len(train_loaders_physics)])
        train_loader_physics = train_loaders_physics[0]
        optional_prepared_loaders = optional_prepared_loaders[len(train_loaders_physics):]
    if layout_pose_aux_head is not None:
        layout_pose_aux_head = optional_prepared_loaders[0]
        optional_prepared_loaders = optional_prepared_loaders[1:]
    if room_layout_aux_head is not None:
        room_layout_aux_head = optional_prepared_loaders[0]
    print(
        f"[DEBUG prepare done] rank={accelerator.process_index} "
        f"elapsed={time.perf_counter() - prepare_start_time:.2f}s",
        flush=True,
    )

    if use_ema_for_transformer:
        ema_transformer.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move `vae`, frozen image encoder to gpu and cast to `weight_dtype`
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder_dinov2.to(accelerator.device, dtype=weight_dtype)

    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = max(1, (configs["lr_scheduler"]["total_steps"] // max(1, int(configs["train"]["epochs"]))))
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    display_total_steps = min(total_updated_steps, int(args.max_train_steps))
    # In mixed setup, allow non-exact divisibility between total steps and epochs
    if accelerator.num_processes > 1 and accelerator.is_main_process:
        print()
    accelerator.wait_for_everyone()
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: 3D[{len(val_loader_3d)}], 4D[{len(val_loader_4d)}]\n")

    # (Optional) Load checkpoint
    global_update_step = 0
    if args.resume_from_iter is not None:
        if args.resume_from_iter < 0:
            args.resume_from_iter = int(sorted(os.listdir(ckpt_dir))[-1])
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        # Load everything
        if version.parse(torch.__version__) >= version.parse("2.4.0"):
            torch.serialization.add_safe_globals([
                int, list, dict, 
                defaultdict,
                Any,
                DictConfig, ListConfig, Metadata, ContainerMetadata, AnyNode
            ]) # avoid deserialization error when loading optimizer state
        accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"))  # torch < 2.4.0 here for `weights_only=False`
        global_update_step = int(args.resume_from_iter)

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = save_experiment_params(args, configs, exp_dir)
        save_model_architecture(accelerator.unwrap_model(transformer), exp_dir)

    # WandB logger
    if accelerator.is_main_process:
        if args.offline_wandb:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=PROJECT_NAME, name=args.tag,
            config=exp_params, dir=exp_dir,
            reinit=True
        )
        # Wandb artifact for logging experiment information
        arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
        arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
        arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
        arti_exp_info.add_file(os.path.join(exp_dir, "log.txt"))  # only save the log before training
        wandb.log_artifact(arti_exp_info)

    def get_sigmas(timesteps: Tensor, n_dim: int, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(dtype=dtype, device=accelerator.device)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero()[0].item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ===== Optional fp32 upcasting for sigma / noisy-latent / target computation =====
    force_fp32 = configs["train"].get("force_fp32", False)

    # ===== Diffusion-Forcing (per-part noise + target-only loss) helpers =====
    df_enabled = configs["train"].get("df_enabled", False)
    df_context_mode = configs["train"].get("df_context_mode", "prefix_k")  # ["prefix_k", "bernoulli_p"]
    df_context_k = int(configs["train"].get("df_context_k", 1))              # used if prefix_k
    df_context_p = float(configs["train"].get("df_context_p", 0.5))          # used if bernoulli_p (prob a token is HISTORY)

    # choose the cleanest timestep (min sigma) for "history" tokens
    _sigmas_all = noise_scheduler.sigmas.to(device=accelerator.device)
    _timesteps_all = noise_scheduler.timesteps.to(device=accelerator.device)
    _clean_idx = torch.argmin(_sigmas_all)
    _clean_timestep = _timesteps_all[_clean_idx]

    def build_df_context_mask(num_parts_vec: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask over the flattened [N] tokens: True = HISTORY (context), False = TARGET.
        num_parts_vec: shape [M], number of parts per object.
        """
        npv = num_parts_vec.to(accelerator.device)
        starts = torch.cat([torch.zeros(1, device=npv.device, dtype=torch.long), torch.cumsum(npv[:-1], dim=0)])
        N = int(npv.sum().item())
        context_mask = torch.zeros(N, device=npv.device, dtype=torch.bool)
        for m in range(npv.shape[0]):
            s, e = int(starts[m].item()), int((starts[m] + npv[m]).item())
            if df_context_mode == "prefix_k":
                k = min(df_context_k, e - s)
                if k > 0:
                    context_mask[s:s+k] = True
            elif df_context_mode == "bernoulli_p":
                context_mask[s:e] = (torch.rand(e - s, device=npv.device) < df_context_p)
            else:
                raise ValueError(f"Unknown df_context_mode: {df_context_mode}")
        # ensure at least one TARGET exists per object
        for m in range(npv.shape[0]):
            s, e = int(starts[m].item()), int((starts[m] + npv[m]).item())
            if context_mask[s:e].all():
                context_mask[e-1] = False
        return context_mask

    def sample_square_occlusion_box(height: int, width: int, ratio_range: tuple[float, float]) -> tuple[int, int, int, int]:
        ratio = random.uniform(ratio_range[0], ratio_range[1])
        side = int(math.floor(min(height, width) * ratio))
        side = max(1, min(side, height, width))
        max_y0 = height - side
        max_x0 = width - side
        y0 = random.randint(0, max_y0) if max_y0 > 0 else 0
        x0 = random.randint(0, max_x0) if max_x0 > 0 else 0
        return y0, y0 + side, x0, x0 + side

    def sample_propagation_span(num_frames: int, ratio_range: tuple[float, float]) -> tuple[int, int]:
        ratio = random.uniform(ratio_range[0], ratio_range[1])
        span = int(math.floor(num_frames * ratio))
        span = max(0, min(span, num_frames))
        if span == 0:
            return 0, 0
        max_start = num_frames - span
        start = random.randint(0, max_start) if max_start > 0 else 0
        return start, start + span

    def apply_advanced_image_masking(images_hw3: torch.Tensor, num_parts_vec: torch.Tensor, mode_key: str) -> torch.Tensor:
        if (not advanced_image_masking_enabled) or mode_key not in advanced_image_masking_mode_cfg:
            return images_hw3
        mode_cfg = advanced_image_masking_mode_cfg[mode_key]
        prob = float(mode_cfg["prob"])
        if prob <= 0.0:
            return images_hw3
        if (not torch.is_tensor(images_hw3)) or images_hw3.ndim != 4 or images_hw3.shape[-1] != 3:
            return images_hw3

        masked_images = images_hw3.clone()
        _, image_h, image_w, _ = masked_images.shape
        mask_size_ratio_range = mode_cfg["mask_size_ratio_range"]
        frame_propagate_ratio_range = mode_cfg["mask_frame_propagate_range"]

        if torch.is_tensor(num_parts_vec):
            num_parts_list = [int(v) for v in num_parts_vec.detach().cpu().tolist()]
        else:
            num_parts_list = [int(v) for v in num_parts_vec]

        start_idx = 0
        for num_parts_item in num_parts_list:
            end_idx = start_idx + num_parts_item
            if num_parts_item <= 0:
                start_idx = end_idx
                continue

            if random.random() < prob:
                y0, y1, x0, x1 = sample_square_occlusion_box(image_h, image_w, mask_size_ratio_range)
                if mode_key == "4d":
                    span_start, span_end = sample_propagation_span(num_parts_item, frame_propagate_ratio_range)
                    if span_end > span_start:
                        frame_slice = slice(start_idx + span_start, start_idx + span_end)
                        # Use each frame's top-left pixel as the fill color for masked regions in 4D mode.
                        bg_colors = masked_images[frame_slice, 0, 0, :].unsqueeze(1).unsqueeze(1)
                        masked_images[frame_slice, y0:y1, x0:x1, :] = bg_colors
                else:
                    masked_images[start_idx:end_idx, y0:y1, x0:x1, :] = 0

            start_idx = end_idx

        return masked_images

    # Start training
    if accelerator.is_main_process:
        print()
    logger.info(f"Start training into {exp_dir}\n")
    logger.logger.propagate = False  # not propagate to the root logger (console)
    progress_bar = tqdm(
        range(display_total_steps),
        initial=global_update_step,
        desc="Training",
        ncols=175,
        disable=not accelerator.is_main_process
    )

    def _replication_collectives_ready(active: bool) -> bool:
        return (
            active
            and sequence_parallel_replicate_inputs
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    def _replicate_batch_from_main_process(batch_obj, *, active: bool):
        if not _replication_collectives_ready(active):
            return batch_obj
        payload = [_move_tensors_to_local_device(batch_obj) if accelerator.is_main_process else None]
        dist.broadcast_object_list(payload, src=0)
        return _move_tensors_to_local_device(payload[0])

    def _move_tensors_to_local_device(obj):
        if torch.is_tensor(obj):
            return obj.to(accelerator.device)
        if isinstance(obj, dict):
            return {key: _move_tensors_to_local_device(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [_move_tensors_to_local_device(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(_move_tensors_to_local_device(value) for value in obj)
        return obj

    def _replicate_tensor_from_main_process(tensor, *, active: bool):
        if tensor is None or not _replication_collectives_ready(active):
            return tensor
        tensor = tensor.to(accelerator.device).contiguous()
        dist.broadcast(tensor, src=0)
        return tensor

    def _replicate_mode_choice_from_main_process(mode_choice):
        if not _replication_collectives_ready(enable_sequence_parallel_attention):
            return mode_choice
        payload = [mode_choice if accelerator.is_main_process else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _transformer_sdpa_context():
        if transformer_sdpa_backend == "auto" or not torch.cuda.is_available():
            return nullcontext()
        return torch.backends.cuda.sdp_kernel(
            enable_flash=transformer_sdpa_backend == "flash",
            enable_mem_efficient=transformer_sdpa_backend == "mem_efficient",
            enable_math=transformer_sdpa_backend == "math",
        )

    flash_attention_profiled = False

    def _profiled_transformer_forward(**forward_kwargs):
        nonlocal flash_attention_profiled
        should_profile = (
            verify_flash_attention_once
            and not flash_attention_profiled
            and torch.cuda.is_available()
            and accelerator.is_main_process
        )
        with _transformer_sdpa_context():
            if not should_profile:
                return transformer(**forward_kwargs)
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities) as prof:
                output = transformer(**forward_kwargs)
                torch.cuda.synchronize(accelerator.device)
            flash_attention_profiled = True
            event_names = {event.key for event in prof.key_averages()}
            flash_hits = sorted(
                name for name in event_names
                if "scaled_dot_product_flash_attention" in name or "flash_attention" in name
            )
            if flash_hits:
                logger.info("FlashAttention SDPA verified for transformer forward: %s", flash_hits)
            else:
                logger.warning(
                    "FlashAttention SDPA was not observed in the profiled transformer forward. "
                    "backend=%s dtype=%s device=%s",
                    transformer_sdpa_backend,
                    forward_kwargs["hidden_states"].dtype,
                    forward_kwargs["hidden_states"].device,
                )
            return output

    def _discard_pending_update() -> None:
        # `AcceleratedOptimizer.zero_grad()` is a no-op when sync_gradients is False,
        # so force a real clear before resetting the accumulation window.
        previous_sync_gradients = accelerator.sync_gradients
        try:
            accelerator.sync_gradients = True
            optimizer.zero_grad()
        finally:
            accelerator.sync_gradients = previous_sync_gradients
        transformer.zero_grad(set_to_none=True)
        accelerator.step = 0
        accelerator.gradient_state._set_sync_gradients(False)

    # Mixed training iterators
    train_iter_3d = yield_forever(train_loader_3d)
    train_iter_4d = yield_forever(train_loader_4d)
    train_iters_physics = [yield_forever(loader) for loader in train_loaders_physics]
    train_iter_physics = train_iters_physics[0] if train_iters_physics else None
    physics_source_probs = [spec["prob"] for spec in physics_dataset_specs]
    physics_source_names = [Path(spec["path"]).stem for spec in physics_dataset_specs]

    def _choose_physics_source_index() -> int:
        if not train_iters_physics:
            return 0
        return int(random.choices(range(len(train_iters_physics)), weights=physics_source_probs, k=1)[0])
    objaverse_train_iter = yield_forever(objaverse_train_loader)
    # Probability to pick 4D at each iter (default 0.5)
    p_4d = float(configs["train"].get("prob_4d", 0.5))
    debug_step_timing = bool(configs["train"].get("debug_step_timing", False))
    max_nonfinite_retries = int(configs["train"].get("max_nonfinite_retries", 20))
    nonfinite_retry_count = 0
    micro_step = 0
    for _ in range(10**12):  # effectively infinite, controlled by max_train_steps

        if global_update_step == args.max_train_steps:
            progress_bar.close()
            logger.logger.propagate = True  # propagate to the root logger (console)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.finish()
            accelerator.end_training()
            logger.info("Training finished!\n")
            return

        if freeze_transformer:
            transformer.eval()
        else:
            transformer.train()

        with accelerator.accumulate(transformer):
            micro_step += 1
            timing_start = time.perf_counter()
            last_timing = timing_start
            if debug_step_timing and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(accelerator.device)

            def _debug_timing(stage: str) -> None:
                nonlocal last_timing
                if not debug_step_timing or not accelerator.is_main_process:
                    return
                now = time.perf_counter()
                try:
                    current_mode = mode
                except (NameError, UnboundLocalError):
                    current_mode = "pending"
                memory_suffix = ""
                if torch.cuda.is_available():
                    allocated_gib = torch.cuda.memory_allocated(accelerator.device) / (1024 ** 3)
                    reserved_gib = torch.cuda.memory_reserved(accelerator.device) / (1024 ** 3)
                    peak_gib = torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3)
                    memory_suffix = (
                        f" mem_alloc_gib={allocated_gib:.3f}"
                        f" mem_reserved_gib={reserved_gib:.3f}"
                        f" mem_peak_gib={peak_gib:.3f}"
                    )
                logger.info(
                    f"[timing] update={global_update_step:06d} micro={micro_step:06d} "
                    f"stage={stage} mode={current_mode} "
                    f"dt={now - last_timing:.3f}s total={now - timing_start:.3f}s "
                    f"sync={accelerator.sync_gradients}"
                    f"{memory_suffix}"
                )
                last_timing = now

            # Randomly choose dataset for this iteration. With replicated SP
            # inputs, every SP rank must enter the same attention backend and
            # collectives, so rank 0 owns this stochastic choice.
            _debug_timing("mode_rng")
            if accelerator.is_main_process or not sequence_parallel_replicate_inputs:
                chosen_use_physics = bool(train_iters_physics) and random.random() < physics_data_prob
                if chosen_use_physics:
                    physics_source_index = _choose_physics_source_index()
                    mode_choice = {
                        "use_physics": True,
                        "is_single_object_step": False,
                        "use_4d": False,
                        "mode": "physics",
                        "physics_source_index": physics_source_index,
                        "physics_source_name": physics_source_names[physics_source_index],
                    }
                else:
                    chosen_single_object = random.random() < single_object_reg_prob and single_object_configs != []
                    if chosen_single_object:
                        mode_choice = {
                            "use_physics": False,
                            "is_single_object_step": True,
                            "use_4d": False,
                            "mode": "single",
                        }
                    else:
                        chosen_use_4d = random.random() < p_4d
                        mode_choice = {
                            "use_physics": False,
                            "is_single_object_step": False,
                            "use_4d": chosen_use_4d,
                            "mode": "4d" if chosen_use_4d else "3d",
                        }
            else:
                mode_choice = None

            mode_choice = _replicate_mode_choice_from_main_process(mode_choice)
            use_physics = bool(mode_choice["use_physics"])
            is_single_object_step = bool(mode_choice["is_single_object_step"])
            use_4d = bool(mode_choice["use_4d"])
            mode = mode_choice["mode"]
            sequence_parallel_active = _sequence_parallel_active_for_mode(mode)

            fetch_batch_on_this_rank = accelerator.is_main_process or not (
                sequence_parallel_active and sequence_parallel_replicate_inputs
            )
            if use_physics:
                physics_source_index = int(mode_choice.get("physics_source_index", 0))
                physics_source_name = str(mode_choice.get("physics_source_name", "physics"))
                _debug_timing(f"batch_wait:physics:{physics_source_name}")
                batch = next(train_iters_physics[physics_source_index]) if fetch_batch_on_this_rank else None
            else:
                if is_single_object_step:
                    _debug_timing("batch_wait:single")
                    batch = next(objaverse_train_iter) if fetch_batch_on_this_rank else None
                else:
                    _debug_timing(f"batch_wait:{mode}")
                    batch = (
                        next(train_iter_4d) if use_4d else next(train_iter_3d)
                    ) if fetch_batch_on_this_rank else None
            trace_sequence_parallel_event("train.before_switch_mode", active=sequence_parallel_active, mode=mode)
            switch_to_mode(mode)
            trace_sequence_parallel_event("train.after_switch_mode", active=sequence_parallel_active, mode=mode)
            trace_sequence_parallel_event("train.before_batch_broadcast", active=sequence_parallel_active, has_batch=batch is not None)
            batch = _replicate_batch_from_main_process(batch, active=sequence_parallel_active)
            trace_sequence_parallel_event("train.after_batch_broadcast", active=sequence_parallel_active, batch_keys=sorted(batch.keys()))
            _debug_timing("batch")

            images_hw3 = batch["images"] # [N, H, W, 3]
            images_hw3 = apply_advanced_image_masking(images_hw3, batch["num_parts"], mode)
            with torch.no_grad():
                preprocess_kwargs = {
                    "images": images_hw3,
                    "return_tensors": "pt",
                }
                if target_image_size is not None:
                    target_size = {"height": target_image_size, "width": target_image_size}
                    preprocess_kwargs["size"] = target_size
                    preprocess_kwargs["crop_size"] = target_size
                    preprocess_kwargs["do_resize"] = True
                    preprocess_kwargs["do_center_crop"] = False
                trace_sequence_parallel_event("train.before_feature_extractor", active=sequence_parallel_active)
                pixel_values = feature_extractor_dinov2(**preprocess_kwargs).pixel_values
                trace_sequence_parallel_event("train.after_feature_extractor", pixel_values, active=sequence_parallel_active)
            pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype) # [N, 3, Hf, Wf]
            trace_sequence_parallel_event("train.after_pixel_to_device", pixel_values, active=sequence_parallel_active)
            _debug_timing("preprocess")
            # Original single-image tokens (frozen DINO)
            with torch.no_grad():
                trace_sequence_parallel_event("train.before_dino", pixel_values, active=sequence_parallel_active)
                single_tokens = image_encoder_dinov2(pixel_values).last_hidden_state  # [N, T, D]
                trace_sequence_parallel_event("train.after_dino", single_tokens, active=sequence_parallel_active)
            _debug_timing("dino")

            # Group indices by objects using num_parts
            num_parts = batch["num_parts"].to(accelerator.device) # [M]
            num_objects = num_parts.shape[0]
            # Use only per-part single-image DINO tokens
            image_embeds = single_tokens  # [N, Ts, D]

            negative_image_embeds = torch.zeros_like(image_embeds)

            if configs["train"]["cfg_dropout_prob"] > 0:
                # Drop entire conditions per part/frame so multi-frame objects do not lose all context at once
                dropout_mask = torch.rand(image_embeds.shape[0], device=accelerator.device) < configs["train"]["cfg_dropout_prob"]  # [N]
                if dropout_mask.any():
                    image_embeds[dropout_mask] = negative_image_embeds[dropout_mask]

            if configs["train"]["occlusion_dropout_prob"] > 0:
                dropout_mask = torch.rand_like(image_embeds, device=accelerator.device) < configs["train"]["occlusion_dropout_prob"]
                if dropout_mask.any():
                    image_embeds[dropout_mask] = negative_image_embeds[dropout_mask]
            trace_sequence_parallel_event("train.before_image_embeds_broadcast", image_embeds, active=sequence_parallel_active)
            image_embeds = _replicate_tensor_from_main_process(image_embeds, active=sequence_parallel_active)
            trace_sequence_parallel_event("train.after_image_embeds_broadcast", image_embeds, active=sequence_parallel_active)

            part_surfaces = batch["part_surfaces"] # [N, P, 6]
            part_surfaces = part_surfaces.to(device=accelerator.device, dtype=weight_dtype)

            with torch.no_grad():
                trace_sequence_parallel_event("train.before_vae_encode", part_surfaces, active=sequence_parallel_active)
                latents = vae.encode(
                    part_surfaces, 
                    **configs["model"]["vae"]
                ).latent_dist.sample()
                trace_sequence_parallel_event("train.after_vae_encode", latents, active=sequence_parallel_active)
            trace_sequence_parallel_event("train.before_latents_broadcast", latents, active=sequence_parallel_active)
            latents = _replicate_tensor_from_main_process(latents, active=sequence_parallel_active)
            trace_sequence_parallel_event("train.after_latents_broadcast", latents, active=sequence_parallel_active)
            _debug_timing("vae_encode")

            noise = torch.randn_like(latents)
            trace_sequence_parallel_event("train.before_noise_broadcast", noise, active=sequence_parallel_active)
            noise = _replicate_tensor_from_main_process(noise, active=sequence_parallel_active)
            trace_sequence_parallel_event("train.after_noise_broadcast", noise, active=sequence_parallel_active)
            # For weighting schemes where we sample timesteps non-uniformly
            # ---- DF-aware per-token timestep sampling ----
            if df_enabled:
                # total token count N across all objects/parts
                N_tokens = latents.shape[0]
                # independent per-token density sampling
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=configs["train"]["weighting_scheme"],
                    batch_size=N_tokens,
                    logit_mean=configs["train"]["logit_mean"],
                    logit_std=configs["train"]["logit_std"],
                    mode_scale=configs["train"]["mode_scale"],
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)  # [N]

                # Build history/target split per object and push HISTORY tokens to the clean timestep
                context_mask = build_df_context_mask(num_parts)                        # [N] bool, True = history
                timesteps = timesteps.clone()
                timesteps[context_mask] = _clean_timestep
            else:
                # original: one timestep per object, then repeat for each part
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=configs["train"]["weighting_scheme"],
                    batch_size=num_objects,
                    logit_mean=configs["train"]["logit_mean"],
                    logit_std=configs["train"]["logit_std"],
                    mode_scale=configs["train"]["mode_scale"],
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(accelerator.device) # [M, ]
                # Repeat the timesteps for each part
                timesteps = timesteps.repeat_interleave(num_parts) # [N, ]
                context_mask = None

            # When force_fp32 is enabled, compute sigmas/noisy-latents/target in fp32
            # to avoid fp16 overflow (e.g. 1/sigma → Inf at small sigma values)
            if force_fp32:
                sigmas = get_sigmas(timesteps, len(latents.shape), torch.float32)
                latents = latents.float()
                noise = noise.float()
            else:
                sigmas = get_sigmas(timesteps, len(latents.shape), weight_dtype)
            trace_sequence_parallel_event("train.before_timesteps_broadcast", timesteps, active=sequence_parallel_active)
            timesteps = _replicate_tensor_from_main_process(timesteps, active=sequence_parallel_active)
            trace_sequence_parallel_event("train.after_timesteps_broadcast", timesteps, active=sequence_parallel_active)
            sigmas = get_sigmas(timesteps, len(latents.shape), torch.float32 if force_fp32 else weight_dtype)
            noisy_latents = (1. - sigmas) * latents + sigmas * noise
            latent_model_input = noisy_latents.to(weight_dtype)
            trace_sequence_parallel_event("train.before_latent_model_input_broadcast", latent_model_input, active=sequence_parallel_active)
            latent_model_input = _replicate_tensor_from_main_process(latent_model_input, active=sequence_parallel_active)
            trace_sequence_parallel_event("train.after_latent_model_input_broadcast", latent_model_input, active=sequence_parallel_active)

            # Note: CFG dropout is applied via gating above, so no in-place
            # replacement here. This preserves the autograd path even when
            # the effective contribution is zero.

            # print(f"latent_model_input.shape = {latent_model_input.shape}, "
            #       f"timesteps.shape = {timesteps.shape}, "
            #       f"sigmas.shape = {sigmas.shape}, "
            #       f"timesteps = {timesteps[:10]}, "
            #       f"num_parts = {num_parts}, "
            #       f"surfaces.shape = {part_surfaces.shape}, "
            #       f"images.shape = {images_hw3.shape}, "
            #       f"pixel_values.shape = {pixel_values.shape}, "
            #       f"image_embeds.shape = {image_embeds.shape}"
            # )

            # Compose attention kwargs for 3D/4D
            ones = torch.ones_like(num_parts).to(device=accelerator.device)
            
            attn_kwargs = {"num_parts": num_parts, "num_frames": ones}
            if mode == "physics" and "num_frames" in batch and "num_spatial_parts" in batch:
                attn_kwargs = {
                    "num_frames": batch["num_frames"].to(accelerator.device),
                    "num_parts": batch["num_spatial_parts"].to(accelerator.device),
                    "layout": "frame_major",
                    "mixing_mode": mixing_mode,
                }
            elif use_4d:
                attn_kwargs = {"num_frames": num_parts, "num_parts": ones}
            elif is_single_object_step:
                attn_kwargs = {"num_parts": ones, "num_frames": ones}

            if sequence_parallel_active:
                attn_kwargs.update({
                    "sequence_parallel_attention": True,
                    "sequence_parallel_replicated_batch": sequence_parallel_replicated_batch,
                    "sequence_parallel_validate_replicated": sequence_parallel_validate_replicated,
                })

            trace_sequence_parallel_event("train.before_condition_broadcasts", active=sequence_parallel_active)
            camera_params = _replicate_tensor_from_main_process(batch.get("camera_params", None), active=sequence_parallel_active)
            frame_time = _replicate_tensor_from_main_process(batch.get("frame_time", None), active=sequence_parallel_active)
            has_camera = _replicate_tensor_from_main_process(batch.get("has_camera", None), active=sequence_parallel_active)
            physics_context = _replicate_tensor_from_main_process(batch.get("physics_context", None), active=sequence_parallel_active)
            trace_sequence_parallel_event("train.after_condition_broadcasts", active=sequence_parallel_active)

            trace_sequence_parallel_event("train.before_transformer", latent_model_input, active=sequence_parallel_active, mode=mode)
            model_pred = _profiled_transformer_forward(
                hidden_states=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=image_embeds,
                attention_kwargs=attn_kwargs,
                camera_params=camera_params,
                frame_time=frame_time,
                has_camera=has_camera,
                physics_context=physics_context,
            ).sample
            trace_sequence_parallel_event("train.after_transformer", model_pred, active=sequence_parallel_active, mode=mode)
            _debug_timing("transformer")

            # Keep a copy of the raw model predictions for consistency loss
            raw_model_pred = model_pred
            clean_latents_for_aux = None

            layout_pose_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            layout_pose_valid_count = torch.tensor(0, device=accelerator.device, dtype=torch.long)
            room_layout_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            room_layout_valid_count = torch.tensor(0, device=accelerator.device, dtype=torch.long)
            geometry_field_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            geometry_field_sdf_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            geometry_field_normal_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            geometry_field_eikonal_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            geometry_image_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            geometry_image_depth_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            geometry_image_normal_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            geometry_image_mask_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            geometry_image_empty_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            geometry_image_valid_count = torch.tensor(0, device=accelerator.device, dtype=torch.long)
            temporal_geometry_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            temporal_geometry_latent_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            temporal_geometry_surface_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            temporal_geometry_scale_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            temporal_geometry_accel_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
            temporal_geometry_pair_count = torch.tensor(0, device=accelerator.device, dtype=torch.long)
            temporal_geometry_object_count = torch.tensor(0, device=accelerator.device, dtype=torch.long)
            layout_latents = None
            layout_pose_pred = None
            use_room_aux = mode == "3d"
            layout_pose_step_active = use_room_aux and layout_pose_aux_head is not None
            room_layout_step_active = use_room_aux and room_layout_aux_head is not None
            geometry_field_step_active = geometry_aux_active and mode in geometry_aux_modes
            if use_room_aux and (layout_pose_aux_head is not None or room_layout_aux_head is not None):
                clean_latents_for_aux = _predict_clean_latents(
                    raw_model_pred=raw_model_pred,
                    noisy_latents=noisy_latents,
                    sigmas=sigmas,
                    objective=configs["train"]["training_objective"],
                )
                layout_latents = clean_latents_for_aux

            if use_room_aux and layout_pose_aux_head is not None:
                layout_pose_pred = layout_pose_aux_head(layout_latents)
                layout_pose_target, layout_pose_mask = _build_layout_pose_targets(
                    part_surfaces=part_surfaces,
                    num_parts=num_parts,
                    room_geometries=batch.get("room_geometry", []),
                )
                layout_pose_valid_count = layout_pose_mask.long().sum()
                if layout_pose_mask.any():
                    layout_pose_loss = tF.mse_loss(
                        layout_pose_pred[layout_pose_mask].float(),
                        layout_pose_target[layout_pose_mask].float(),
                    )
                else:
                    layout_pose_loss = layout_pose_pred.sum() * 0.0

            if use_room_aux and room_layout_aux_head is not None:
                room_layout_pred = room_layout_aux_head(layout_latents, num_parts)
                room_layout_target, room_layout_mask = _build_room_layout_targets(
                    part_surfaces=part_surfaces,
                    num_parts=num_parts,
                    room_geometries=batch.get("room_geometry", []),
                )
                room_layout_valid_count = room_layout_mask.long().sum()
                if room_layout_mask.any():
                    room_layout_reg_loss = tF.mse_loss(
                        room_layout_pred[room_layout_mask, :6].float(),
                        room_layout_target[room_layout_mask, :6].float(),
                    )
                    room_layout_presence_loss = tF.binary_cross_entropy_with_logits(
                        room_layout_pred[room_layout_mask, 6:].float(),
                        room_layout_target[room_layout_mask, 6:].float(),
                    )
                    room_layout_loss = room_layout_reg_loss + room_layout_presence_loss
                else:
                    room_layout_loss = room_layout_pred.sum() * 0.0

            if geometry_field_step_active:
                if clean_latents_for_aux is None:
                    clean_latents_for_aux = _predict_clean_latents(
                        raw_model_pred=raw_model_pred,
                        noisy_latents=noisy_latents,
                        sigmas=sigmas,
                        objective=configs["train"]["training_objective"],
                    )
                geometry_field_loss, geometry_field_terms = _compute_geometry_field_auxiliary_loss(
                    vae=vae,
                    clean_latents=clean_latents_for_aux,
                    part_surfaces=part_surfaces,
                    num_points=geometry_aux_num_points,
                    sdf_weight=geometry_aux_sdf_weight,
                    normal_weight=geometry_aux_normal_weight,
                    eikonal_weight=geometry_aux_eikonal_weight,
                    second_order=geometry_aux_second_order,
                    eikonal_grad_norm_clip=geometry_aux_eikonal_grad_norm_clip,
                    decoder_num_chunks=geometry_aux_decoder_num_chunks,
                    decoder_dtype=weight_dtype,
                )
                geometry_field_sdf_loss = geometry_field_terms["sdf"]
                geometry_field_normal_loss = geometry_field_terms["normal"]
                geometry_field_eikonal_loss = geometry_field_terms["eikonal"]

            geometry_image_aux_step_active = (
                geometry_image_aux_active
                and mode in geometry_image_aux_modes
                and (not geometry_image_aux_sync_step_only or accelerator.sync_gradients)
            )
            if geometry_image_aux_step_active:
                if clean_latents_for_aux is None:
                    clean_latents_for_aux = _predict_clean_latents(
                        raw_model_pred=raw_model_pred,
                        noisy_latents=noisy_latents,
                        sigmas=sigmas,
                        objective=configs["train"]["training_objective"],
                    )
                geometry_image_loss, geometry_image_terms = _compute_geometry_image_auxiliary_loss(
                    vae=vae,
                    clean_latents=clean_latents_for_aux,
                    geometry_image=batch.get("geometry_image"),
                    num_parts=num_parts,
                    num_surface_pixels=geometry_image_aux_num_surface_pixels,
                    num_empty_pixels=geometry_image_aux_num_empty_pixels,
                    depth_weight=geometry_image_aux_depth_weight,
                    normal_weight=geometry_image_aux_normal_weight,
                    mask_weight=geometry_image_aux_mask_weight,
                    empty_weight=geometry_image_aux_empty_weight,
                    surface_margin=geometry_image_aux_surface_margin,
                    empty_margin=geometry_image_aux_empty_margin,
                    max_empty_backprop_pixels=geometry_image_aux_max_empty_backprop_pixels,
                    ray_sign_weight=geometry_image_aux_ray_sign_weight,
                    ray_sign_epsilon=geometry_image_aux_ray_sign_epsilon,
                    decoder_num_chunks=geometry_image_aux_decoder_num_chunks,
                    decoder_dtype=weight_dtype,
                )
                geometry_image_depth_loss = geometry_image_terms["depth"]
                geometry_image_normal_loss = geometry_image_terms["normal"]
                geometry_image_mask_loss = geometry_image_terms["mask"]
                geometry_image_empty_loss = geometry_image_terms["empty"]
                geometry_image_valid_count = geometry_image_terms["valid"]

            temporal_geometry_step_active = temporal_geometry_aux_active and mode in temporal_geometry_aux_modes
            if temporal_geometry_step_active:
                if clean_latents_for_aux is None:
                    clean_latents_for_aux = _predict_clean_latents(
                        raw_model_pred=raw_model_pred,
                        noisy_latents=noisy_latents,
                        sigmas=sigmas,
                        objective=configs["train"]["training_objective"],
                    )
                temporal_layout_pose_pred = None
                if temporal_geometry_aux_scale_weight > 0.0 and layout_pose_aux_head is not None:
                    temporal_layout_pose_pred = layout_pose_pred
                    if temporal_layout_pose_pred is None:
                        # Use the layout/pose head as a frozen scale probe here. The
                        # temporal term should shape the predicted clean latents, not
                        # teach the probe to output constant scales.
                        aux_param_requires_grad = [param.requires_grad for param in layout_pose_aux_head.parameters()]
                        try:
                            for param in layout_pose_aux_head.parameters():
                                param.requires_grad_(False)
                            temporal_layout_pose_pred = layout_pose_aux_head(clean_latents_for_aux)
                        finally:
                            for param, requires_grad in zip(layout_pose_aux_head.parameters(), aux_param_requires_grad):
                                param.requires_grad_(requires_grad)
                temporal_geometry_skip_first_spatial_parts = temporal_geometry_aux_skip_first_spatial_parts
                if (
                    mode == "physics"
                    and temporal_geometry_aux_physics_skip_first_spatial_parts is not None
                    and "physics_source_index" in mode_choice
                ):
                    temporal_geometry_skip_first_spatial_parts = temporal_geometry_aux_physics_skip_first_spatial_parts[
                        int(mode_choice["physics_source_index"])
                    ]
                temporal_geometry_loss, temporal_geometry_terms = _compute_temporal_geometry_auxiliary_loss(
                    vae=vae,
                    clean_latents=clean_latents_for_aux,
                    part_surfaces=part_surfaces,
                    num_parts=num_parts,
                    num_frames=batch.get("num_frames"),
                    num_spatial_parts=batch.get("num_spatial_parts"),
                    layout_pose_pred=temporal_layout_pose_pred,
                    latent_weight=temporal_geometry_aux_latent_weight,
                    surface_weight=temporal_geometry_aux_surface_weight,
                    scale_weight=temporal_geometry_aux_scale_weight,
                    accel_weight=temporal_geometry_aux_accel_weight,
                    num_surface_points=temporal_geometry_aux_num_surface_points,
                    decoder_num_chunks=temporal_geometry_aux_decoder_num_chunks,
                    decoder_dtype=weight_dtype,
                    fallback_consecutive=temporal_geometry_aux_fallback_consecutive,
                    skip_first_spatial_parts=temporal_geometry_skip_first_spatial_parts,
                )
                temporal_geometry_latent_loss = temporal_geometry_terms["latent"]
                temporal_geometry_surface_loss = temporal_geometry_terms["surface"]
                temporal_geometry_scale_loss = temporal_geometry_terms["scale"]
                temporal_geometry_accel_loss = temporal_geometry_terms["accel"]
                temporal_geometry_pair_count = temporal_geometry_terms["pairs"]
                temporal_geometry_object_count = temporal_geometry_terms["objects"]
            _debug_timing("aux")

            if configs["train"]["training_objective"] == "x0":  # Section 5 of https://arxiv.org/abs/2206.00364
                model_pred = model_pred.float() * (-sigmas) + noisy_latents  # predicted x_0
                target = latents
            elif configs["train"]["training_objective"] == 'v':  # flow matching
                target = noise - latents
            elif configs["train"]["training_objective"] == '-v':  # reverse flow matching
                # The training objective for TripoSG is the reverse of the flow matching objective. 
                # It uses "different directions", i.e., the negative velocity. 
                # This is probably a mistake in engineering, not very harmful. 
                # In TripoSG's rectified flow scheduler, prev_sample = sample + (sigma - sigma_next) * model_output
                # See TripoSG's scheduler https://github.com/VAST-AI-Research/TripoSG/blob/main/triposg/schedulers/scheduling_rectified_flow.py#L296
                # While in diffusers's flow matching scheduler, prev_sample = sample + (sigma_next - sigma) * model_output
                # See https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L454
                target = latents - noise
            else:
                raise ValueError(f"Unknown training objective [{configs['train']['training_objective']}]")

            # For these weighting schemes use a uniform timestep sampling, so post-weight the loss
            weighting = compute_loss_weighting_for_sd3(
                configs["train"]["weighting_scheme"],
                sigmas
            )

            diff_loss = weighting * tF.mse_loss(model_pred.float(), target.float(), reduction="none")
            diff_loss = diff_loss.mean(dim=list(range(1, len(diff_loss.shape))))  # [N]

            # Intra-object consecutive consistency loss on raw model predictions
            cons_weight = float(configs["train"].get("consistency_loss_weight", 0.0))
            consistency_loss = torch.tensor(0.0, device=accelerator.device, dtype=diff_loss.dtype)
            if cons_weight > 0.0 and mode == "4d":
                pred_flat = raw_model_pred.float().view(raw_model_pred.shape[0], -1)  # [N, *]
                parts_ptr = 0
                all_pairs = []
                for m in range(num_objects):
                    k = int(num_parts[m].item())
                    if k > 1:
                        seq = pred_flat[parts_ptr:parts_ptr + k]  # [k, *]
                        diffs = seq[1:] - seq[:-1]               # [k-1, *]
                        pair_loss = (diffs.pow(2).mean(dim=1))   # [k-1]
                        all_pairs.append(pair_loss)
                    parts_ptr += k
                if len(all_pairs) > 0:
                    consistency_loss = torch.cat(all_pairs, dim=0).mean()

            if df_enabled and context_mask is not None:
                # target-only reduction for diffusion loss
                target_mask = (~context_mask).to(diff_loss.dtype)
                denom = target_mask.sum().clamp_min(1.0)
                base_loss = (diff_loss * target_mask).sum() / denom
            else:
                base_loss = diff_loss.mean()

            loss = base_loss + cons_weight * consistency_loss
            if layout_pose_step_active:
                loss = loss + layout_pose_aux_weight * layout_pose_loss
            if room_layout_step_active:
                loss = loss + room_layout_aux_weight * room_layout_loss
            if geometry_field_step_active:
                loss = loss + geometry_aux_weight * geometry_field_loss
            if geometry_image_aux_step_active:
                loss = loss + geometry_image_aux_weight * geometry_image_loss
            if temporal_geometry_step_active:
                loss = loss + temporal_geometry_loss

            # Ensure optional embedding parameters participate every step to keep DDP reductions consistent.
            base_transformer = _unwrap_transformer_for_attn()
            zero_refs = []
            for attr in (
                "frame_embedding",
                "dynamic_embedding",
                "dynamic_embedding_per_block",
                "static_embedding",
                "static_embedding_per_block",
                "part_embedding",
                "instance_type_embedding",
                "object_id_embedding",
                "camera_condition_proj",
                "frame_time_proj",
                "physics_condition_proj",
            ):
                module = getattr(base_transformer, attr, None)
                if module is None:
                    continue
                if hasattr(module, "parameters"):
                    zero_refs.extend(param for param in module.parameters() if param.requires_grad)
                elif hasattr(module, "weight") and getattr(module.weight, "requires_grad", False):
                    zero_refs.append(module.weight)
                elif torch.is_tensor(module) and getattr(module, "requires_grad", False):
                    zero_refs.append(module)
            if zero_refs:
                zero_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
                for ref in zero_refs:
                    zero_loss = zero_loss + ref.sum() * 0.0
                loss = loss + zero_loss

            aux_zero_refs = []
            for aux_head in (layout_pose_aux_head, room_layout_aux_head):
                if aux_head is None:
                    continue
                aux_zero_refs.extend(param for param in aux_head.parameters() if param.requires_grad)
            if aux_zero_refs:
                aux_zero_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
                for ref in aux_zero_refs:
                    aux_zero_loss = aux_zero_loss + ref.sum() * 0.0
                loss = loss + aux_zero_loss

            # Skip this batch if loss is NaN/Inf on any rank.
            # We clear gradients to avoid carrying partial accumulation forward.
            trace_sequence_parallel_event("train.before_finite_check", loss.detach())
            finite_int = torch.isfinite(loss.detach()).to(device=accelerator.device, dtype=torch.int32)
            trace_sequence_parallel_event("train.before_finite_reduce", finite_int)
            if accelerator.num_processes > 1:
                finite_int = accelerator.reduce(finite_int, reduction="min")
            trace_sequence_parallel_event("train.after_finite_reduce", finite_int)
            if finite_int.item() == 0:
                nonfinite_retry_count += 1
                _discard_pending_update()
                if nonfinite_retry_count >= max_nonfinite_retries:
                    raise RuntimeError(
                        f"Non-finite loss persisted for {nonfinite_retry_count} consecutive micro-steps "
                        f"at update [{global_update_step:06d}] in mode [{mode}]. "
                        "Stopping instead of retrying forever. Disable newly added conditioning paths or lower precision-sensitive settings."
                    )
                logger.warning(
                    f"Non-finite loss detected at update [{global_update_step:06d}] in mode [{mode}]; "
                    "discarded accumulated gradients and restarting this optimizer step."
                )
                continue

            nonfinite_retry_count = 0

            # Backpropagate
            trace_sequence_parallel_event("train.before_backward", loss.detach())
            accelerator.backward(loss)
            trace_sequence_parallel_event("train.after_backward", loss.detach())
            _debug_timing("backward")
            if accelerator.sync_gradients:
                clip_params = [param for group in optimizer.param_groups for param in group["params"]]
                accelerator.clip_grad_norm_(clip_params, args.max_grad_norm)
                _debug_timing("clip")

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            _debug_timing("optim")

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            # Gather the losses across all processes for logging (if we use distributed training)
            loss = accelerator.gather(loss.detach()).mean()
            base_loss_for_log = accelerator.gather(base_loss.detach()).mean()
            consistency_loss_for_log = accelerator.gather(consistency_loss.detach()).mean()

            logs = {
                "loss": loss.item(),
                "loss_base": base_loss_for_log.item(),
                "lr": lr_scheduler.get_last_lr()[0],
                f"{mode}_loss": loss.item(),
                f"{mode}_loss_base": base_loss_for_log.item(),
            }
            # Log DF stats if enabled
            if df_enabled and ("context_mask" in locals()) and (context_mask is not None):
                with torch.no_grad():
                    tgt_ratio = ((~context_mask).float().mean()).item()
                logs.update({"df/target_ratio": tgt_ratio})
            # Log consistency and total loss if enabled
            if configs["train"].get("consistency_loss_weight", 0.0) > 0.0:
                # Mirror the actual base + consistency loss composition in logs.
                base_loss_log = base_loss_for_log
                logs.update({
                    "loss_consistency": consistency_loss_for_log.item(),
                    "loss_total": (base_loss_log + configs["train"]["consistency_loss_weight"] * consistency_loss_for_log).item(),
                    f"{mode}_loss_consistency": consistency_loss_for_log.item(),
                    f"{mode}_loss_total": (base_loss_log + configs["train"]["consistency_loss_weight"] * consistency_loss_for_log).item(),
                })
            if layout_pose_step_active:
                valid_count = accelerator.gather(layout_pose_valid_count.detach()).sum()
                if int(valid_count.item()) > 0:
                    logs.update({
                        "loss_layout_pose": layout_pose_loss.item(),
                        "layout_pose_valid": int(valid_count.item()),
                    })
            if room_layout_step_active:
                valid_count = accelerator.gather(room_layout_valid_count.detach()).sum()
                if int(valid_count.item()) > 0:
                    logs.update({
                        "loss_room_layout": room_layout_loss.item(),
                        "room_layout_valid": int(valid_count.item()),
                    })
            if geometry_field_step_active:
                geometry_field_logs = {"loss_geometry_field": geometry_field_loss.item()}
                if geometry_aux_sdf_weight > 0.0:
                    geometry_field_logs["loss_geometry_sdf"] = geometry_field_sdf_loss.item()
                if geometry_aux_normal_weight > 0.0 and geometry_aux_second_order:
                    geometry_field_logs["loss_geometry_normal"] = geometry_field_normal_loss.item()
                if geometry_aux_eikonal_weight > 0.0 and geometry_aux_second_order:
                    geometry_field_logs["loss_geometry_eikonal"] = geometry_field_eikonal_loss.item()
                logs.update(geometry_field_logs)
            if geometry_image_aux_step_active:
                valid_count = accelerator.gather(geometry_image_valid_count.detach()).sum()
                if int(valid_count.item()) > 0:
                    geometry_image_logs = {
                        "loss_geometry_image": geometry_image_loss.item(),
                        "geometry_image_valid": int(valid_count.item()),
                    }
                    if geometry_image_aux_depth_weight > 0.0:
                        geometry_image_logs["loss_geometry_image_depth"] = geometry_image_depth_loss.item()
                    if geometry_image_aux_normal_weight > 0.0:
                        geometry_image_logs["loss_geometry_image_normal"] = geometry_image_normal_loss.item()
                    if geometry_image_aux_mask_weight > 0.0:
                        geometry_image_logs["loss_geometry_image_mask"] = geometry_image_mask_loss.item()
                    if geometry_image_aux_empty_weight > 0.0:
                        geometry_image_logs["loss_geometry_image_empty"] = geometry_image_empty_loss.item()
                    logs.update(geometry_image_logs)
            if temporal_geometry_step_active:
                pair_count = accelerator.gather(temporal_geometry_pair_count.detach()).sum()
                object_count = accelerator.gather(temporal_geometry_object_count.detach()).sum()
                temporal_logs = {
                    "loss_temporal_geometry": temporal_geometry_loss.item(),
                    "temporal_geometry_pairs": int(pair_count.item()),
                    "temporal_geometry_objects": int(object_count.item()),
                }
                if temporal_geometry_aux_latent_weight > 0.0:
                    temporal_logs["loss_temporal_geometry_latent"] = temporal_geometry_latent_loss.item()
                if temporal_geometry_aux_surface_weight > 0.0:
                    temporal_logs["loss_temporal_geometry_surface"] = temporal_geometry_surface_loss.item()
                if temporal_geometry_aux_scale_weight > 0.0:
                    temporal_logs["loss_temporal_geometry_scale"] = temporal_geometry_scale_loss.item()
                if temporal_geometry_aux_accel_weight > 0.0:
                    temporal_logs["loss_temporal_geometry_accel"] = temporal_geometry_accel_loss.item()
                logs.update(temporal_logs)
            if use_ema_for_transformer:
                ema_transformer.step(transformer.parameters())
                logs.update({"ema": ema_transformer.cur_decay_value})

            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            global_update_step += 1

            msg = (
                f"[{global_update_step:06d} / {display_total_steps:06d}] "
                f"loss: {logs['loss']:.4f}, loss_base: {logs['loss_base']:.4f}, lr: {logs['lr']:.2e}, mode: {mode}"
            )
            if mode == "physics":
                msg += f", physics_loss: {logs['physics_loss']:.4f}, physics_loss_base: {logs['physics_loss_base']:.4f}"
            if 'loss_consistency' in logs and 'loss_total' in logs:
                msg += f", loss_consistency: {logs['loss_consistency']:.4f}, loss_total: {logs['loss_total']:.4f}"
            if 'loss_layout_pose' in logs:
                msg += f", loss_layout_pose: {logs['loss_layout_pose']:.4f}, layout_pose_valid: {logs['layout_pose_valid']}"
            if 'loss_room_layout' in logs:
                msg += f", loss_room_layout: {logs['loss_room_layout']:.4f}, room_layout_valid: {logs['room_layout_valid']}"
            if 'loss_geometry_field' in logs:
                field_parts = []
                if "loss_geometry_sdf" in logs:
                    field_parts.append(f"sdf: {logs['loss_geometry_sdf']:.4f}")
                if "loss_geometry_normal" in logs:
                    field_parts.append(f"normal: {logs['loss_geometry_normal']:.4f}")
                if "loss_geometry_eikonal" in logs:
                    field_parts.append(f"eik: {logs['loss_geometry_eikonal']:.4f}")
                suffix = f" ({', '.join(field_parts)})" if field_parts else ""
                msg += f", loss_geometry_field: {logs['loss_geometry_field']:.4f}{suffix}"
            if 'loss_geometry_image' in logs:
                image_parts = []
                if "loss_geometry_image_depth" in logs:
                    image_parts.append(f"depth: {logs['loss_geometry_image_depth']:.4f}")
                if "loss_geometry_image_normal" in logs:
                    image_parts.append(f"normal: {logs['loss_geometry_image_normal']:.4f}")
                if "loss_geometry_image_mask" in logs:
                    image_parts.append(f"mask: {logs['loss_geometry_image_mask']:.4f}")
                if "loss_geometry_image_empty" in logs:
                    image_parts.append(f"empty: {logs['loss_geometry_image_empty']:.4f}")
                image_parts.append(f"valid: {logs['geometry_image_valid']}")
                msg += f", loss_geometry_image: {logs['loss_geometry_image']:.4f} ({', '.join(image_parts)})"
            if 'loss_temporal_geometry' in logs:
                temporal_parts = []
                if "loss_temporal_geometry_latent" in logs:
                    temporal_parts.append(f"latent: {logs['loss_temporal_geometry_latent']:.4f}")
                if "loss_temporal_geometry_surface" in logs:
                    temporal_parts.append(f"surface: {logs['loss_temporal_geometry_surface']:.4f}")
                if "loss_temporal_geometry_scale" in logs:
                    temporal_parts.append(f"scale: {logs['loss_temporal_geometry_scale']:.4f}")
                if "loss_temporal_geometry_accel" in logs:
                    temporal_parts.append(f"accel: {logs['loss_temporal_geometry_accel']:.4f}")
                temporal_parts.append(f"pairs: {logs['temporal_geometry_pairs']}")
                msg += f", loss_temporal_geometry: {logs['loss_temporal_geometry']:.4f} ({', '.join(temporal_parts)})"
            if use_ema_for_transformer and 'ema' in logs:
                msg += f", ema: {logs['ema']:.4f}"
            logger.info(msg)

            # Log the training progress
            if (
                global_update_step % configs["train"]["log_freq"] == 0 
                or global_update_step == 1
                or global_update_step % updated_steps_per_epoch == 0 # last step of an epoch
            ):  
                if accelerator.is_main_process:
                    to_log = {
                        "training/loss": logs["loss"],
                        "training/loss_base": logs["loss_base"],
                        "training/lr": logs["lr"],
                        f"training_{mode}/loss": logs[f"{mode}_loss"],
                        f"training_{mode}/loss_base": logs[f"{mode}_loss_base"],
                    }
                    if mode == "physics":
                        to_log.update({
                            "training/physics_loss": logs["physics_loss"],
                            "training/physics_loss_base": logs["physics_loss_base"],
                            "training_physics/physics_loss": logs["physics_loss"],
                            "training_physics/physics_loss_base": logs["physics_loss_base"],
                        })
                    if "loss_consistency" in logs:
                        to_log.update({
                            "training/loss_consistency": logs["loss_consistency"],
                            "training/loss_total": logs["loss_total"],
                            f"training_{mode}/loss_consistency": logs[f"{mode}_loss_consistency"],
                            f"training_{mode}/loss_total": logs[f"{mode}_loss_total"],
                        })
                    if "loss_layout_pose" in logs:
                        to_log.update({
                            "training/loss_layout_pose": logs["loss_layout_pose"],
                            "training/layout_pose_valid": logs["layout_pose_valid"],
                            f"training_{mode}/loss_layout_pose": logs["loss_layout_pose"],
                        })
                    if "loss_room_layout" in logs:
                        to_log.update({
                            "training/loss_room_layout": logs["loss_room_layout"],
                            "training/room_layout_valid": logs["room_layout_valid"],
                            f"training_{mode}/loss_room_layout": logs["loss_room_layout"],
                        })
                    if "loss_geometry_field" in logs:
                        to_log.update({
                            "training/loss_geometry_field": logs["loss_geometry_field"],
                            f"training_{mode}/loss_geometry_field": logs["loss_geometry_field"],
                        })
                        if "loss_geometry_sdf" in logs:
                            to_log["training/loss_geometry_sdf"] = logs["loss_geometry_sdf"]
                        if "loss_geometry_normal" in logs:
                            to_log["training/loss_geometry_normal"] = logs["loss_geometry_normal"]
                        if "loss_geometry_eikonal" in logs:
                            to_log["training/loss_geometry_eikonal"] = logs["loss_geometry_eikonal"]
                    if "loss_geometry_image" in logs:
                        to_log.update({
                            "training/loss_geometry_image": logs["loss_geometry_image"],
                            "training/geometry_image_valid": logs["geometry_image_valid"],
                            f"training_{mode}/loss_geometry_image": logs["loss_geometry_image"],
                        })
                        if "loss_geometry_image_depth" in logs:
                            to_log["training/loss_geometry_image_depth"] = logs["loss_geometry_image_depth"]
                        if "loss_geometry_image_normal" in logs:
                            to_log["training/loss_geometry_image_normal"] = logs["loss_geometry_image_normal"]
                        if "loss_geometry_image_mask" in logs:
                            to_log["training/loss_geometry_image_mask"] = logs["loss_geometry_image_mask"]
                        if "loss_geometry_image_empty" in logs:
                            to_log["training/loss_geometry_image_empty"] = logs["loss_geometry_image_empty"]
                    if "loss_temporal_geometry" in logs:
                        to_log.update({
                            "training/loss_temporal_geometry": logs["loss_temporal_geometry"],
                            "training/temporal_geometry_pairs": logs["temporal_geometry_pairs"],
                            "training/temporal_geometry_objects": logs["temporal_geometry_objects"],
                            f"training_{mode}/loss_temporal_geometry": logs["loss_temporal_geometry"],
                        })
                        if "loss_temporal_geometry_latent" in logs:
                            to_log["training/loss_temporal_geometry_latent"] = logs["loss_temporal_geometry_latent"]
                        if "loss_temporal_geometry_surface" in logs:
                            to_log["training/loss_temporal_geometry_surface"] = logs["loss_temporal_geometry_surface"]
                        if "loss_temporal_geometry_scale" in logs:
                            to_log["training/loss_temporal_geometry_scale"] = logs["loss_temporal_geometry_scale"]
                        if "loss_temporal_geometry_accel" in logs:
                            to_log["training/loss_temporal_geometry_accel"] = logs["loss_temporal_geometry_accel"]
                    wandb.log(to_log, step=global_update_step)
                    if use_ema_for_transformer:
                        wandb.log({
                            "training/ema": logs["ema"]
                        }, step=global_update_step)

            # Save checkpoint
            if (
                global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                or global_update_step == total_updated_steps # 3. last step of an epoch
                # or global_update_step == 1 # 4. first step
            ): 

                gc.collect()
                if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                elif accelerator.is_main_process:
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                accelerator.wait_for_everyone()  # ensure all processes have finished saving
                gc.collect()

            # Evaluate on the validation set
            if args.max_val_steps > 0 and (
                (global_update_step % configs["train"]["early_eval_freq"] == 0 and global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                or global_update_step == total_updated_steps # 4. last step of an epoch
                or (global_update_step == 1 and not args.offline_wandb) # 5. first step
            ):  

                # Use EMA parameters for evaluation
                if use_ema_for_transformer:
                    # Store the Transformer parameters temporarily and load the EMA parameters to perform inference
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())

                transformer.eval()

                try:
                    log_validation(
                        val_loader_3d, random_val_loader_3d,
                        val_loader_4d, random_val_loader_4d,
                        feature_extractor_dinov2, image_encoder_dinov2,
                        vae, transformer,
                        global_update_step, eval_dir,
                        accelerator, logger,
                        args, configs
                    )
                except Exception:
                    # Keep training alive even if validation unexpectedly fails on some rank.
                    logger.exception("Validation failed unexpectedly; skipping this validation cycle.")
                    accelerator.wait_for_everyone()
                finally:
                    if use_ema_for_transformer:
                        # Switch back to the original Transformer parameters
                        ema_transformer.restore(transformer.parameters())
                    torch.cuda.empty_cache()
                    gc.collect()

@torch.no_grad()
def log_validation(
    dataloader_3d, random_dataloader_3d,
    dataloader_4d, random_dataloader_4d,
    feature_extractor_dinov2, image_encoder_dinov2,
    vae, transformer, 
    global_step, eval_dir,
    accelerator, logger,  
    args, configs
):  

    if args.val_only_rank0 and not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    val_noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )
    target_image_size = configs["train"].get("dino_preprocess_size", None)
    target_image_size = int(target_image_size) if target_image_size is not None else None

    # Build both pipelines sharing the same underlying transformer
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    pipeline_3d = PartCrafterPipeline(
        vae=vae,
        transformer=unwrapped_transformer,
        scheduler=val_noise_scheduler,
        feature_extractor_dinov2=feature_extractor_dinov2,
        image_encoder_dinov2=image_encoder_dinov2,
    )
    pipeline_4d = FourDCrafterPipeline(
        vae=vae,
        transformer=unwrapped_transformer,
        scheduler=val_noise_scheduler,
        feature_extractor_dinov2=feature_extractor_dinov2,
        image_encoder_dinov2=image_encoder_dinov2,
    )

    pipeline_3d.set_progress_bar_config(disable=True)
    pipeline_4d.set_progress_bar_config(disable=True)
    # pipeline.enable_xformers_memory_efficient_attention()

    if args.seed >= 0:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None
        

    val_progress_bar = tqdm(
        range((len(dataloader_3d) + len(dataloader_4d)) // 2) if args.max_val_steps is None else range(args.max_val_steps),
        desc=f"Validation [{global_step:06d}]",
        ncols=125,
        disable=not accelerator.is_main_process
    )

    medias_dictlist, metrics_dictlist = defaultdict(list), defaultdict(list)

    val_dataloader_3d, random_val_dataloader_3d = yield_forever(dataloader_3d), yield_forever(random_dataloader_3d)
    val_dataloader_4d, random_val_dataloader_4d = yield_forever(dataloader_4d), yield_forever(random_dataloader_4d)
    sample_timeout_sec = float(configs["val"].get("sample_timeout_seconds", 90.0))
    requested_signal_timeout = bool(configs["val"].get("use_signal_sample_timeout", False))
    enable_signal_timeout = (
        requested_signal_timeout
        and sample_timeout_sec > 0
        and hasattr(signal, "setitimer")
        and hasattr(signal, "ITIMER_REAL")
        and os.name != "nt"
    )
    if sample_timeout_sec > 0 and requested_signal_timeout and accelerator.is_main_process:
        if enable_signal_timeout:
            logger.warning(
                "Validation sample timeout is using SIGALRM interruption. "
                "Set val.use_signal_sample_timeout=false to use safe elapsed-time checks."
            )
        else:
            logger.warning(
                "Validation sample timeout requested SIGALRM interruption, but this runtime does not support it. "
                "Falling back to safe elapsed-time checks."
            )

    def _raise_if_sample_timed_out(sample_start_time: float, stage_name: str) -> None:
        if sample_timeout_sec <= 0:
            return
        elapsed = time.perf_counter() - sample_start_time
        if elapsed > sample_timeout_sec:
            raise ValidationSampleTimeout(
                f"Validation sample exceeded {sample_timeout_sec:.1f} seconds during {stage_name} "
                f"(elapsed {elapsed:.2f}s)."
            )

    def _start_sample_timeout() -> Optional[Any]:
        if not enable_signal_timeout:
            return None

        def _timeout_handler(signum, frame):
            raise ValidationSampleTimeout(
                f"Validation sample exceeded {sample_timeout_sec:.1f} seconds."
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, sample_timeout_sec)
        return previous_handler

    def _clear_sample_timeout(previous_handler: Optional[Any]) -> None:
        if previous_handler is None:
            return
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)

    val_step = 0
    while val_step < args.max_val_steps:
        sample_start_time = time.perf_counter()
        sample_timed_out = False
        sample_skip_reason = ""
        timeout_prev_handler = None
        N: Optional[int] = None
        use_4d = False
        sample_medias_dictlist, sample_metrics_dictlist = defaultdict(list), defaultdict(list)
        parts_chamfer_distances = torch.zeros(1, device=accelerator.device)
        parts_f_scores = torch.zeros(1, device=accelerator.device)

        try:
            timeout_prev_handler = _start_sample_timeout()

            # randomly select between 3D and 4D validation batches
            use_4d = (torch.rand(1).item() < 0.5)
            if val_step < args.max_val_steps // 2:
                # deterministic half
                batch = next(val_dataloader_4d if use_4d else val_dataloader_3d)
            else:
                batch = next(random_val_dataloader_4d if use_4d else random_val_dataloader_3d)

            images = batch["images"]
            if len(images.shape) == 5:
                images = images[0] # (1, N, H, W, 3) -> (N, H, W, 3)
            images = [Image.fromarray(image) for image in images.cpu().numpy()]
            part_surfaces = batch["part_surfaces"].cpu().numpy()
            if len(part_surfaces.shape) == 4:
                part_surfaces = part_surfaces[0] # (1, N, P, 6) -> (N, P, 6)

            N = len(images)
            preprocess_kwargs = {
                "images": images,
                "return_tensors": "pt",
            }
            if target_image_size is not None:
                target_size = {"height": target_image_size, "width": target_image_size}
                preprocess_kwargs["size"] = target_size
                preprocess_kwargs["crop_size"] = target_size
                preprocess_kwargs["do_resize"] = True
                preprocess_kwargs["do_center_crop"] = False
            pixel_values = feature_extractor_dinov2(**preprocess_kwargs).pixel_values
            if target_image_size is not None and pixel_values.shape[-2:] != (target_image_size, target_image_size):
                pixel_values = tF.interpolate(
                    pixel_values,
                    size=(target_image_size, target_image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            _raise_if_sample_timed_out(sample_start_time, "input preprocessing")

            val_progress_bar.set_postfix(
                {"num_parts": N}
            )

            # Build attention maps for switching inside validation
            spatial_ids = configs["model"]["transformer"].get("spatial_global_attn_block_ids", configs["model"]["transformer"].get("global_attn_block_ids", []))
            spatial_ids = list(spatial_ids) if spatial_ids is not None else []
            temporal_ids = configs["model"]["transformer"].get("temporal_global_attn_block_ids", configs["model"]["transformer"].get("global_attn_block_ids", []))
            temporal_ids = list(temporal_ids) if temporal_ids is not None else []
            def _build_attn_processor_map_local(model, active_ids):
                d = {}
                for layer_id in range(model.config.num_layers):
                    for attn_id in [1, 2]:
                        key = f'blocks.{layer_id}.attn{attn_id}.processor'
                        d[key] = PartFrameCrafterAttnProcessor() if layer_id in (active_ids or []) else TripoSGAttnProcessor2_0()
                return d

            with torch.autocast("cuda", torch.float16):
                for guidance_scale in sorted(args.val_guidance_scales):
                    _raise_if_sample_timed_out(sample_start_time, f"guidance_scale={guidance_scale:.1f} setup")

                    # Switch attention processors and ids for this mode
                    if use_4d:
                        unwrapped_transformer.set_attn_processor(_build_attn_processor_map_local(unwrapped_transformer, temporal_ids))
                        unwrapped_transformer.global_attn_block_ids = list(temporal_ids)
                        pred_part_meshes = pipeline_4d(
                            pixel_values,
                            num_inference_steps=configs['val']['num_inference_steps'],
                            num_tokens=configs['model']['vae']['num_tokens'],
                            guidance_scale=guidance_scale, 
                            attention_kwargs={"num_frames": N, "num_parts": 1},
                            generator=generator,
                            max_num_expanded_coords=configs['val']['max_num_expanded_coords'],
                            use_flash_decoder=configs['val']['use_flash_decoder'],
                        ).meshes
                    else:
                        unwrapped_transformer.set_attn_processor(_build_attn_processor_map_local(unwrapped_transformer, spatial_ids))
                        unwrapped_transformer.global_attn_block_ids = list(spatial_ids)
                        pred_part_meshes = pipeline_3d(
                            pixel_values,
                            num_inference_steps=configs['val']['num_inference_steps'],
                            num_tokens=configs['model']['vae']['num_tokens'],
                            guidance_scale=guidance_scale, 
                            attention_kwargs={"num_parts": N, "num_frames": 1},
                            generator=generator,
                            max_num_expanded_coords=configs['val']['max_num_expanded_coords'],
                            use_flash_decoder=configs['val']['use_flash_decoder'],
                        ).meshes
                    _raise_if_sample_timed_out(sample_start_time, f"guidance_scale={guidance_scale:.1f} generation")

                    # Save the generated meshes
                    if accelerator.is_main_process:
                        mode_str = "4d" if use_4d else "3d"
                        local_eval_dir = os.path.join(eval_dir, f"{global_step:06d}", mode_str, f"guidance_scale_{guidance_scale:.1f}")
                        os.makedirs(local_eval_dir, exist_ok=True)
                        rendered_images_list, rendered_normals_list = [], []
                        # 1. save the gt image
                        images[0].save(os.path.join(local_eval_dir, f"{val_step:04d}.png"))
                        # 2. save the generated part meshes
                        for n in range(N):
                            if pred_part_meshes[n] is None:
                                # If the generated mesh is None (decoing error), use a dummy mesh
                                pred_part_meshes[n] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                            pred_part_meshes[n].export(os.path.join(local_eval_dir, f"{val_step:04d}_{n:02d}.glb"))

                        # 3. render the generated mesh and save the rendered images
                        pred_mesh = get_colored_mesh_composition(pred_part_meshes)
                        rendered_images: List[Image.Image] = render_views_around_mesh(
                            pred_mesh, 
                            num_views=configs['val']['rendering']['num_views'],
                            radius=configs['val']['rendering']['radius'],
                        )
                        rendered_normals: List[Image.Image] = render_normal_views_around_mesh(
                            pred_mesh,
                            num_views=configs['val']['rendering']['num_views'],
                            radius=configs['val']['rendering']['radius'],
                        )
                        export_renderings(
                            rendered_images,
                            os.path.join(local_eval_dir, f"{val_step:04d}.gif"),
                            fps=configs['val']['rendering']['fps']
                        )
                        export_renderings(
                            rendered_normals,
                            os.path.join(local_eval_dir, f"{val_step:04d}_normals.gif"),
                            fps=configs['val']['rendering']['fps']
                        )
                        rendered_images_list.append(rendered_images)
                        rendered_normals_list.append(rendered_normals)
                        # Build a paired GIF cycling each condition image with the rotating render
                        paired_frames = []
                        for cond_img in images:
                            for frame in rendered_images:
                                w = cond_img.width + frame.width
                                h = max(cond_img.height, frame.height)
                                canvas = Image.new("RGB", (w, h), (255, 255, 255))
                                canvas.paste(cond_img, (0, 0))
                                canvas.paste(frame, (cond_img.width, 0))
                                paired_frames.append(canvas)
                        export_renderings(
                            paired_frames,
                            os.path.join(local_eval_dir, f"{val_step:04d}_paired.gif"),
                            fps=configs['val']['rendering']['fps']
                        )
                        _raise_if_sample_timed_out(sample_start_time, f"guidance_scale={guidance_scale:.1f} rendering")

                        sample_medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/gt_image"] += [images[0]] # List[Image.Image] TODO: support batch size > 1
                        sample_medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_images"] += rendered_images_list # List[List[Image.Image]]
                        sample_medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_normals"] += rendered_normals_list # List[List[Image.Image]]

                    ################################ Compute generation metrics ################################

                    parts_chamfer_distances, parts_f_scores = [], []

                    for n in range(N):
                        # gt_part_surface = part_surfaces[n]
                        # pred_part_mesh = pred_part_meshes[n]
                        # if pred_part_mesh is None:
                        #     # If the generated mesh is None (decoing error), use a dummy mesh
                        #     pred_part_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                        # part_cd, part_f = compute_cd_and_f_score_in_training(
                        #     gt_part_surface, pred_part_mesh,
                        #     num_samples=configs['val']['metric']['cd_num_samples'],
                        #     threshold=configs['val']['metric']['f1_score_threshold'],
                        #     metric=configs['val']['metric']['cd_metric']
                        # )
                        # # avoid nan
                        # part_cd = configs['val']['metric']['default_cd'] if np.isnan(part_cd) else part_cd
                        # part_f = configs['val']['metric']['default_f1'] if np.isnan(part_f) else part_f
                        # parts_chamfer_distances.append(part_cd)
                        # parts_f_scores.append(part_f)

                        # TODO: Fix this
                        # Disable chamfer distance and F1 score for now
                        parts_chamfer_distances.append(0.0)
                        parts_f_scores.append(0.0)

                    parts_chamfer_distances = torch.tensor(parts_chamfer_distances, device=accelerator.device)
                    parts_f_scores = torch.tensor(parts_f_scores, device=accelerator.device)

                    sample_metrics_dictlist[f"parts_chamfer_distance_cfg{guidance_scale:.1f}"].append(parts_chamfer_distances.mean())
                    sample_metrics_dictlist[f"parts_f_score_cfg{guidance_scale:.1f}"].append(parts_f_scores.mean())
                    _raise_if_sample_timed_out(sample_start_time, f"guidance_scale={guidance_scale:.1f} metrics")
        except ValidationSampleTimeout as e:
            sample_timed_out = True
            sample_skip_reason = str(e)
        except Exception:
            sample_timed_out = True
            sample_skip_reason = "validation sample failed unexpectedly"
            logger.exception(
                f"Validation [{val_step:02d}/{args.max_val_steps:02d}] failed unexpectedly. Skipping sample."
            )
        finally:
            _clear_sample_timeout(timeout_prev_handler)

        sample_elapsed = time.perf_counter() - sample_start_time
        if sample_timeout_sec > 0 and (sample_elapsed > sample_timeout_sec) and (not sample_timed_out):
            sample_timed_out = True
            sample_skip_reason = f"sample exceeded {sample_timeout_sec:.1f}s"

        if sample_timed_out:
            if accelerator.is_main_process:
                skipped_dir = os.path.join(eval_dir, f"{global_step:06d}", "skipped")
                os.makedirs(skipped_dir, exist_ok=True)
                dummy_mesh = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                mode_str = "4d" if use_4d else "3d"
                dummy_mesh.export(os.path.join(skipped_dir, f"{val_step:04d}_{mode_str}.glb"))
            logger.warning(
                f"Validation [{val_step:02d}/{args.max_val_steps:02d}] skipped: "
                f"{sample_skip_reason or 'timed out'} (elapsed {sample_elapsed:.2f}s)."
            )
            val_progress_bar.set_postfix({"num_parts": N if N is not None else 0, "skipped": 1})
            val_step += 1
            val_progress_bar.update(1)
            continue

        for key, value in sample_medias_dictlist.items():
            medias_dictlist[key] += value
        for key, value in sample_metrics_dictlist.items():
            metrics_dictlist[key] += value

        # Only log the last (biggest) cfg metrics in the progress bar
        val_logs = {
            "parts_chamfer_distance": parts_chamfer_distances.mean().item(),
            "parts_f_score": parts_f_scores.mean().item(),
        }
        val_progress_bar.set_postfix(**val_logs)
        logger.info(
            f"Validation [{val_step:02d}/{args.max_val_steps:02d}] " +
            f"parts_chamfer_distance: {val_logs['parts_chamfer_distance']:.4f}, parts_f_score: {val_logs['parts_f_score']:.4f}"
        )
        logger.info(
            f"parts_chamfer_distances: {[f'{x:.4f}' for x in parts_chamfer_distances.tolist()]}"
        )
        logger.info(
            f"parts_f_scores: {[f'{x:.4f}' for x in parts_f_scores.tolist()]}"
        )
        val_step += 1
        val_progress_bar.update(1)

    val_progress_bar.close()

    if accelerator.is_main_process:
        try:
            for key, value in medias_dictlist.items():
                if len(value) == 0:
                    continue
                # Ensure nested directory exists for keys like "3d/guidance_scale_7.0/gt_image"
                nested_dir = os.path.join(eval_dir, f"{global_step:06d}", os.path.dirname(key))
                os.makedirs(nested_dir, exist_ok=True)
                if isinstance(value[0], Image.Image): # assuming gt_image
                    image_grid = make_grid_for_images_or_videos(
                        value, 
                        nrow=configs['val']['nrow'],
                        return_type='pil', 
                    )
                    image_grid.save(os.path.join(eval_dir, f"{global_step:06d}", f"{key}.png"))
                    wandb.log({f"validation/{key}": wandb.Image(image_grid)}, step=global_step)
                else: # assuming pred_rendered_images or pred_rendered_normals
                    image_grids = make_grid_for_images_or_videos(
                        value, 
                        nrow=configs['val']['nrow'],
                        return_type='ndarray',
                    )
                    wandb.log({
                        f"validation/{key}": wandb.Video(
                            image_grids, 
                            fps=configs['val']['rendering']['fps'], 
                            format="gif"
                    )}, step=global_step)
                    image_grids = [Image.fromarray(image_grid.transpose(1, 2, 0)) for image_grid in image_grids]
                    export_renderings(
                        image_grids, 
                        os.path.join(eval_dir, f"{global_step:06d}", f"{key}.gif"), 
                        fps=configs['val']['rendering']['fps']
                    )

            for k, v in metrics_dictlist.items():
                wandb.log({f"validation/{k}": torch.tensor(v).mean().item()}, step=global_step)
        except Exception:
            logger.exception("Validation media/metrics logging failed; continuing training.")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()

from src.utils.typing_utils import *

import json
import os
import random

import accelerate
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_surface, load_surfaces
from src.datasets.local_cache import prefetch_data_configs, resolve_path

# --- Robust image loader for WebP/alpha ---
def _load_rgb_image(image_path: str, size: tuple[int, int]) -> torch.Tensor:
    """
    Open an image (incl. WebP with alpha), composite on white, resize, and return
    a uint8 tensor shaped [H, W, 3]. Handles animated WebP by taking frame 0.
    """
    im = Image.open(image_path)  # do NOT pass custom mode
    # If animated (e.g., WebP), use first frame
    if getattr(im, "is_animated", False):
        try:
            im.seek(0)
        except Exception:
            pass
    # Ensure it is fully loaded before further ops (avoids lazy-loading issues)
    im.load()

    # Handle transparency correctly: composite onto white before dropping alpha
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB")
    else:
        im = im.convert("RGB")

    # Resize with a stable resampler
    im = im.resize(size, Image.Resampling.BILINEAR)

    # Convert to tensor HWC uint8
    arr = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(arr)

def _extract_single_surface_array(surface_data: dict, num_points: int) -> torch.Tensor:
    """Return a single [P,6] surface tensor from a loaded npy dict.

    Supports three shapes:
    - Top-level dict with keys 'surface_points' & 'surface_normals'
    - Dict with key 'object' containing the above
    - Dict with key 'parts' (expects length 1); takes the first part
    """
    if surface_data is None:
        raise ValueError("surface_data is None")
    if 'surface_points' in surface_data and 'surface_normals' in surface_data:
        return load_surface(surface_data, num_pc=num_points)
    if 'object' in surface_data:
        return load_surface(surface_data['object'], num_pc=num_points)
    if 'parts' in surface_data:
        parts = surface_data['parts']
        if isinstance(parts, list) and len(parts) > 0:
            return load_surface(parts[0], num_pc=num_points)
    raise KeyError("Unrecognized surface data format: expected 'surface_points'+'surface_normals', or 'object', or non-empty 'parts'.")


def _extract_part_surface_arrays(surface_data: dict, num_points: int, max_parts: Optional[int] = None) -> torch.Tensor:
    """Return [num_parts, P, 6] surface tensors, preserving explicit parts when present."""
    if surface_data is None:
        raise ValueError("surface_data is None")
    if 'parts' in surface_data and isinstance(surface_data['parts'], list) and len(surface_data['parts']) > 0:
        parts = surface_data['parts'][:max_parts] if max_parts is not None else surface_data['parts']
        return load_surfaces(parts, num_pc=num_points)
    return _extract_single_surface_array(surface_data, num_points).unsqueeze(0)


def _count_surface_parts(surface_path: str) -> int:
    surface_data = np.load(resolve_path(surface_path), allow_pickle=True).item()
    parts = surface_data.get('parts', None) if isinstance(surface_data, dict) else None
    if isinstance(parts, list) and len(parts) > 0:
        return len(parts)
    return 1


_PART_ALIGNED_FRAME_FIELDS = (
    "object_names", "object_translation", "object_quaternion_xyzw",
    "object_linear_velocity", "object_angular_velocity", "visibility",
    "visibility_valid", "visible_mask_paths", "amodal_mask_paths",
    "visible_mask_area", "amodal_mask_area", "mask_touches_border",
    "sensor_coverage_proxy", "observation_quality",
)


def _selected_part_indices(frame: dict, include_name_prefixes: tuple[str, ...],
                           exclude_names: frozenset[str],
                           expected_parts: Optional[int] = None) -> list[int]:
    """Select ordered frame-part indices using manifest object names."""
    names = frame.get("object_names")
    filtering = bool(include_name_prefixes or exclude_names)
    if names is None:
        if filtering:
            raise ValueError("Part-name filtering requires object_names in every frame")
        if expected_parts is None:
            raise ValueError("expected_parts is required when object_names are absent")
        return list(range(expected_parts))
    if not isinstance(names, (list, tuple)):
        raise ValueError("object_names must be a list")
    if expected_parts is not None and len(names) != expected_parts:
        raise ValueError(
            f"object_names has {len(names)} entries, but the surface has {expected_parts} parts"
        )
    selected = [
        index for index, raw_name in enumerate(names)
        if ((not include_name_prefixes or str(raw_name).startswith(include_name_prefixes))
            and str(raw_name) not in exclude_names)
    ]
    if not selected:
        raise ValueError(
            "Part-name filtering removed every object from frame "
            f"{frame.get('surface_path', '<unknown>')}"
        )
    return selected


def _filter_frame_parts(frame: dict, surface_data: dict,
                        include_name_prefixes: tuple[str, ...],
                        exclude_names: frozenset[str]) -> tuple[dict, dict]:
    """Filter surface parts and all manifest fields aligned with those parts."""
    parts = surface_data.get("parts") if isinstance(surface_data, dict) else None
    if not isinstance(parts, list) or not parts:
        if include_name_prefixes or exclude_names:
            raise ValueError("Part-name filtering requires a non-empty surface parts list")
        return frame, surface_data
    indices = _selected_part_indices(
        frame, include_name_prefixes, exclude_names, expected_parts=len(parts)
    )
    if len(indices) == len(parts):
        return frame, surface_data
    filtered_frame = dict(frame)
    for field in _PART_ALIGNED_FRAME_FIELDS:
        if field not in frame:
            continue
        values = frame[field]
        if not isinstance(values, (list, tuple)) or len(values) != len(parts):
            raise ValueError(
                f"{field} must contain one value per surface part before filtering; "
                f"expected {len(parts)} values"
            )
        filtered_frame[field] = [values[index] for index in indices]
    filtered_surface = dict(surface_data)
    filtered_surface["parts"] = [parts[index] for index in indices]
    return filtered_frame, filtered_surface


def _is_spatiotemporal_part_count_error(exc: Exception) -> bool:
    return (
        isinstance(exc, ValueError)
        and "Physics/spatio-temporal grid expected" in str(exc)
    )


def _data_config_key(data_config: dict) -> str:
    if not data_config:
        return ""
    if "object_key" in data_config:
        return str(data_config["object_key"])
    if "frames" in data_config and len(data_config["frames"]) > 0:
        return str(data_config["frames"][0].get("surface_path", ""))
    return str(data_config.get("surface_path", ""))


def _as_float_tensor(value, shape: tuple[int, ...]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == shape:
        return torch.from_numpy(arr)
    flat = arr.reshape(-1)
    expected = int(np.prod(shape))
    if flat.size == expected:
        return torch.from_numpy(flat.reshape(shape))
    return None


def _camera_condition_from_frame(frame: dict) -> tuple[torch.Tensor, torch.Tensor]:
    intr = _as_float_tensor(
        frame.get("intrinsics", frame.get("K", frame.get("camera_intrinsics"))),
        (3, 3),
    )
    c2w = _as_float_tensor(
        frame.get(
            "camera_to_world",
            frame.get("c2w", frame.get("world_from_camera", frame.get("transform_matrix"))),
        ),
        (4, 4),
    )
    w2c = _as_float_tensor(
        frame.get(
            "world_to_camera",
            frame.get("w2c", frame.get("camera_from_world", frame.get("extrinsics_camera_from_world"))),
        ),
        (4, 4),
    )
    if c2w is None and w2c is not None:
        c2w = torch.linalg.inv(w2c.float())
    if intr is None or c2w is None:
        return torch.zeros(25, dtype=torch.float32), torch.tensor(False)
    return torch.cat([intr.float().reshape(-1), c2w.float().reshape(-1)], dim=0), torch.tensor(True)


def _frame_time_from_frame(frame: dict, fallback_index: int, sequence_len: int) -> torch.Tensor:
    for key in ("tau", "frame_time", "time", "timestamp"):
        if key in frame:
            try:
                return torch.tensor(float(frame[key]), dtype=torch.float32)
            except (TypeError, ValueError):
                pass
    frame_idx = frame.get("frame_index", frame.get("frame", fallback_index))
    try:
        frame_idx = float(frame_idx)
    except (TypeError, ValueError):
        frame_idx = float(fallback_index)
    denom = max(float(sequence_len - 1), 1.0)
    return torch.tensor(frame_idx / denom, dtype=torch.float32)



class ObjaversePartDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs: DictConfig, 
        training: bool = True, 
    ):
        super().__init__()
        self.configs = configs
        self.training = training

        self.min_num_parts = configs['dataset']['min_num_parts']
        self.max_num_parts = configs['dataset']['max_num_parts']
        # For new frame-format datasets, min/max_num_parts historically mean
        # sampled frame count. Prefer explicit aliases when provided.
        self.min_num_frames = int(configs['dataset'].get('min_num_frames', self.min_num_parts))
        self.max_num_frames = int(configs['dataset'].get('max_num_frames', self.max_num_parts))
        self.val_min_num_parts = configs['val']['min_num_parts']
        self.val_max_num_parts = configs['val']['max_num_parts']

        self.max_iou_mean = configs['dataset'].get('max_iou_mean', None)
        self.max_iou_max = configs['dataset'].get('max_iou_max', None)

        self.shuffle_parts = configs['dataset']['shuffle_parts']
        self.training_ratio = configs['dataset']['training_ratio']
        self.balance_object_and_parts = configs['dataset'].get('balance_object_and_parts', False)
        self.spatiotemporal_grid = bool(configs['dataset'].get('spatiotemporal_grid', False))
        self.include_object_name_prefixes = tuple(
            str(value) for value in configs['dataset'].get('include_object_name_prefixes', [])
        )
        self.exclude_object_names = frozenset(
            str(value) for value in configs['dataset'].get('exclude_object_names', [])
        )
        configured_num_spatial_parts = configs['dataset'].get('num_spatial_parts', None)
        self.num_spatial_parts = int(configured_num_spatial_parts) if configured_num_spatial_parts is not None else None
        configured_max_spatial_parts = configs['dataset'].get('max_num_spatial_parts', None)
        self.max_num_spatial_parts = int(configured_max_spatial_parts) if configured_max_spatial_parts is not None else None
        self.max_spatiotemporal_grid_retries = int(configs['dataset'].get('max_spatiotemporal_grid_retries', 64))
        self._bad_spatiotemporal_configs = set()

        self.rotating_ratio = configs['dataset'].get('rotating_ratio', 0.0)
        self.rotating_degree = configs['dataset'].get('rotating_degree', 10.0)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-self.rotating_degree, self.rotating_degree), fill=(255, 255, 255)),
        ])

        # Load dataset configs. Support two formats:
        # 1) Old list format: a list of dicts with keys including 'num_parts', 'valid', etc.
        # 2) New frame format: a dict mapping object_key -> list of frame dicts ({surface_path, image_path, iou_*}).
        def _load_one(path):
            with open(path, 'r') as f:
                return json.load(f)

        raw_entries = []
        if isinstance(configs['dataset']['config'], ListConfig):
            for config_path in configs['dataset']['config']:
                raw_entries.append(_load_one(config_path))
        else:
            raw_entries.append(_load_one(configs['dataset']['config']))

        # Detect format by inspecting the first top-level element
        data_configs: list[dict] = []
        new_format_objects: list[dict] = []
        any_new_format = False
        for entry in raw_entries:
            if isinstance(entry, list):
                # Old flat list format
                data_configs += entry
            elif isinstance(entry, dict):
                # New format: object_key -> list[frame]
                any_new_format = True
                for object_key, frames in entry.items():
                    if not isinstance(frames, list) or len(frames) == 0:
                        continue
                    # Normalize frames: only keep those with both surface_path & image_path
                    norm_frames = []
                    for fr in frames:
                        sp = fr.get('surface_path', None)
                        ip = fr.get('image_path', None)
                        if sp is None or ip is None:
                            continue
                        norm_frame = dict(fr)
                        norm_frame['surface_path'] = sp
                        norm_frame['image_path'] = ip
                        norm_frames.append(norm_frame)
                    if len(norm_frames) == 0:
                        continue
                    new_format_objects.append({
                        'object_key': object_key,
                        'frames': norm_frames,
                        'num_frames': len(norm_frames),
                        'valid': True,
                    })
            else:
                raise ValueError("Unsupported dataset JSON root type; expected list or dict")

        if any_new_format:
            # We keep dataset min/max parts from the config. Do not override with global frame stats.
            if len(new_format_objects) == 0:
                raise ValueError("No valid objects found in the new frame-format dataset JSON")

            # Split train/val by ratio if not balancing objects/parts
            objects = new_format_objects
            if not self.balance_object_and_parts:
                split = int(len(objects) * self.training_ratio)
                if self.training:
                    objects = objects[:split]
                else:
                    objects = objects[split:]

            # Assign a fixed sampled-frame count to each object. For new frame-format
            # datasets, min/max_num_frames are the clear names; min/max_num_parts are
            # kept as backward-compatible aliases.
            data_configs = []
            for obj in objects:
                n_frames = obj['num_frames']
                # Limit upper bound by object's frames with stride-2 feasibility (0,2,4,...)
                feasible_max = (n_frames + 1) // 2  # maximum K with step=2 contiguous selection
                upper = min(self.max_num_frames, feasible_max)
                # If object has fewer frames than dataset min, lower falls back to what's available
                lower = min(max(1, self.min_num_frames), upper)
                if lower <= 0:
                    continue
                num_frames_sample = random.randint(lower, upper)
                if self.spatiotemporal_grid:
                    if self.include_object_name_prefixes or self.exclude_object_names:
                        selected_signatures = {
                            tuple(
                                str(frame["object_names"][index])
                                for index in _selected_part_indices(
                                    frame,
                                    self.include_object_name_prefixes,
                                    self.exclude_object_names,
                                    expected_parts=len(frame.get("object_names", [])),
                                )
                            )
                            for frame in obj["frames"]
                        }
                        if len(selected_signatures) != 1:
                            raise ValueError(
                                "Part-name filtering must select one stable ordered object set "
                                f"per sequence; {obj['object_key']} produced "
                                f"{sorted(selected_signatures)}"
                            )
                        spatial_parts = len(next(iter(selected_signatures)))
                    elif self.num_spatial_parts is None:
                        spatial_parts = _count_surface_parts(obj['frames'][0]['surface_path'])
                    else:
                        spatial_parts = self.num_spatial_parts
                    if self.max_num_spatial_parts is not None and spatial_parts > self.max_num_spatial_parts:
                        continue
                    num_parts = num_frames_sample * spatial_parts
                else:
                    spatial_parts = 1
                    num_parts = num_frames_sample
                data_configs.append({
                    'object_key': obj['object_key'],
                    'frames': obj['frames'],
                    'num_frames': n_frames,
                    'num_sampled_frames': num_frames_sample,
                    'num_spatial_parts': spatial_parts,
                    'num_parts': num_parts,
                    'valid': True,
                    # Placeholders for compatibility
                    'iou_mean': 0.0,
                    'iou_max': 0.0,
                })

        # Filter and finalize data_configs
        if len(data_configs) > 0 and 'surface_path' in data_configs[0] or (len(data_configs) > 0 and 'surface_paths' in data_configs[0]):
            # Old format path: apply old filters
            data_configs = [config for config in data_configs if config.get('valid', True)]
            data_configs = [config for config in data_configs if self.min_num_parts <= config['num_parts'] <= self.max_num_parts]
            if self.max_iou_mean is not None and self.max_iou_max is not None:
                data_configs = [config for config in data_configs if config.get('iou_mean', 0.0) <= self.max_iou_mean]
                data_configs = [config for config in data_configs if config.get('iou_max', 0.0) <= self.max_iou_max]
            if not self.balance_object_and_parts:
                if self.training:
                    data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
                else:
                    data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
                    data_configs = [config for config in data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
        else:
            # New format: we already assigned num_parts and split by ratio above; optional extra filter by val min/max when not training
            if not self.training:
                data_configs = [config for config in data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]

        self.data_configs = data_configs
        image_load_size = int(configs["train"].get("image_load_size", 512))
        self.image_size = (image_load_size, image_load_size)
        self.surface_num_points = int(
            configs.get("dataset", {}).get(
                "surface_num_points",
                configs["train"].get("surface_num_points", 204800),
            )
        )

    def __len__(self) -> int:
        return len(self.data_configs)
    
    def _get_data_by_config(self, data_config):
        # Support three cases:
        # 1) Old format with a single 'surface_path' + one 'image_path'
        # 2) Old format with 'surface_paths' list + one 'image_path' (replicate image)
        # 3) New format with 'frames' list and fixed 'num_parts' -> sample that many frames
        if 'frames' in data_config:
            # New frame-based format
            frames = data_config['frames']
            k = int(data_config.get('num_sampled_frames', data_config['num_parts']))
            # Select k frames in even-step consecutive order: s, s+2, s+4, ... (no reordering)
            F = len(frames)
            max_start = (F - 1) - 2 * (k - 1)
            max_start = max_start if max_start >= 0 else 0
            # Enforce even starts (0,2,4,...) to get 0-2-4 style
            valid_starts = list(range(0, max_start + 1, 2))
            if len(valid_starts) == 0:
                # Fallback: allow any start, still step by 2 to preserve no-jump guarantee
                valid_starts = list(range(0, max_start + 1))
            stride = int(self.configs['dataset'].get('temporal_stride', 2))
            stride_choices = self.configs['dataset'].get('temporal_stride_choices', None)
            if stride_choices:
                stride = int(random.choice(list(stride_choices)))
            stride = max(stride, 1)
            max_start = max((F - 1) - stride * (k - 1), 0)
            valid_starts = list(range(0, max_start + 1, stride)) or [0]
            s = random.choice(valid_starts)
            chosen = [frames[min(s + stride * i, F - 1)] for i in range(k)]
            # Load surfaces per chosen frame
            part_surfaces = []
            images_list = []
            camera_list = []
            has_camera_list = []
            frame_time_list = []
            visibility_list = []
            visibility_valid_list = []
            observation_quality_list = []
            non_border_list = []
            translation_list = []
            quaternion_list = []
            pose_valid_list = []
            for local_idx, fr in enumerate(chosen):
                surface_path = resolve_path(fr['surface_path'])
                image_path = resolve_path(fr['image_path'])
                surface_data = np.load(surface_path, allow_pickle=True).item()
                fr, surface_data = _filter_frame_parts(
                    fr,
                    surface_data,
                    self.include_object_name_prefixes,
                    self.exclude_object_names,
                )
                camera_condition, has_camera = _camera_condition_from_frame(fr)
                frame_time = _frame_time_from_frame(fr, s + stride * local_idx, F)
                if self.spatiotemporal_grid:
                    expected_parts = int(data_config.get('num_spatial_parts', self.num_spatial_parts or 1))
                    frame_surfaces = _extract_part_surface_arrays(
                        surface_data,
                        self.surface_num_points,
                        max_parts=expected_parts,
                    )
                    if frame_surfaces.shape[0] != expected_parts:
                        raise ValueError(
                            f"Physics/spatio-temporal grid expected {expected_parts} explicit parts in {surface_path}, "
                            f"but found {frame_surfaces.shape[0]}. Regenerate/preprocess physics data with --include-parts "
                            "or set dataset_physics.num_spatial_parts to match the data."
                        )
                    part_surfaces.append(frame_surfaces)
                    repeat_count = frame_surfaces.shape[0]
                else:
                    part_surfaces.append(_extract_single_surface_array(surface_data, self.surface_num_points))
                    repeat_count = 1
                camera_list.extend([camera_condition] * repeat_count)
                has_camera_list.extend([has_camera] * repeat_count)
                frame_time_list.extend([frame_time] * repeat_count)
                raw_visibility = fr.get("visibility", 0.0)
                if isinstance(raw_visibility, (list, tuple)):
                    if len(raw_visibility) != repeat_count:
                        raise ValueError(
                            f"visibility must have {repeat_count} values for {surface_path}"
                        )
                    visibility_list.extend(float(value) for value in raw_visibility)
                else:
                    visibility_list.extend([float(raw_visibility)] * repeat_count)
                raw_visibility_valid = fr.get("visibility_valid", False)
                if isinstance(raw_visibility_valid, (list, tuple)):
                    if len(raw_visibility_valid) != repeat_count:
                        raise ValueError(
                            f"visibility_valid must have {repeat_count} values for {surface_path}"
                        )
                    visibility_valid_list.extend(bool(value) for value in raw_visibility_valid)
                else:
                    visibility_valid_list.extend([bool(raw_visibility_valid)] * repeat_count)

                raw_quality = fr.get("observation_quality", raw_visibility)
                if isinstance(raw_quality, (list, tuple)):
                    if len(raw_quality) != repeat_count:
                        raise ValueError(
                            f"observation_quality must have {repeat_count} values for {surface_path}"
                        )
                    observation_quality_list.extend(float(value) for value in raw_quality)
                else:
                    observation_quality_list.extend([float(raw_quality)] * repeat_count)

                raw_border = fr.get("mask_touches_border", False)
                if isinstance(raw_border, (list, tuple)):
                    if len(raw_border) != repeat_count:
                        raise ValueError(
                            f"mask_touches_border must have {repeat_count} values for {surface_path}"
                        )
                    non_border_list.extend(not bool(value) for value in raw_border)
                else:
                    non_border_list.extend([not bool(raw_border)] * repeat_count)

                translations = fr.get("object_translation")
                quaternions = fr.get("object_quaternion_xyzw")
                pose_valid = (
                    isinstance(translations, (list, tuple))
                    and isinstance(quaternions, (list, tuple))
                    and len(translations) == repeat_count
                    and len(quaternions) == repeat_count
                )
                if pose_valid:
                    translation_list.extend(translations)
                    quaternion_list.extend(quaternions)
                    pose_valid_list.extend([True] * repeat_count)
                else:
                    translation_list.extend([[0.0, 0.0, 0.0]] * repeat_count)
                    quaternion_list.extend([[0.0, 0.0, 0.0, 1.0]] * repeat_count)
                    pose_valid_list.extend([False] * repeat_count)
                # Load image per frame
                pil_image = Image.open(image_path)
                if getattr(pil_image, "is_animated", False):
                    try:
                        pil_image.seek(0)
                    except Exception:
                        pass
                pil_image.load()
                if pil_image.mode in ("RGBA", "LA"):
                    bg = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
                    pil_image = Image.alpha_composite(bg, pil_image.convert("RGBA")).convert("RGB")
                else:
                    pil_image = pil_image.convert("RGB")
                if random.random() < self.rotating_ratio:
                    pil_image = self.transform(pil_image)
                pil_image = pil_image.resize(self.image_size, Image.Resampling.BILINEAR)
                img = np.asarray(pil_image, dtype=np.uint8).copy()
                image_tensor = torch.from_numpy(img).to(torch.uint8)
                if self.spatiotemporal_grid:
                    images_list.extend([image_tensor] * part_surfaces[-1].shape[0])
                else:
                    images_list.append(image_tensor)
            if self.spatiotemporal_grid:
                part_surfaces = torch.cat(part_surfaces, dim=0)  # frame-major [F*num_parts, P, 6]
            else:
                part_surfaces = torch.stack(part_surfaces, dim=0)  # [N, P, 6]
            images = torch.stack(images_list, dim=0)  # [N, H, W, 3]
            out = {
                "images": images,
                "part_surfaces": part_surfaces,
                "camera_params": torch.stack(camera_list, dim=0),
                "has_camera": torch.stack(has_camera_list, dim=0).bool(),
                "frame_time": torch.stack(frame_time_list, dim=0),
                "visibility": torch.tensor(visibility_list, dtype=torch.float32).clamp_(0, 1),
                "visibility_valid": torch.tensor(visibility_valid_list, dtype=torch.bool),
                "observation_quality": torch.tensor(observation_quality_list, dtype=torch.float32).clamp_(0, 1),
                "non_border": torch.tensor(non_border_list, dtype=torch.bool),
                "object_translation": torch.tensor(translation_list, dtype=torch.float32),
                "object_quaternion_xyzw": torch.tensor(quaternion_list, dtype=torch.float32),
                "object_pose_valid": torch.tensor(pose_valid_list, dtype=torch.bool),
            }
            if self.spatiotemporal_grid:
                out["num_frames"] = torch.LongTensor([k])
                out["num_spatial_parts"] = torch.LongTensor([int(data_config.get('num_spatial_parts', self.num_spatial_parts or 1))])
            return out
        elif 'surface_path' in data_config:
            surface_path = resolve_path(data_config['surface_path'])
            surface_data = np.load(surface_path, allow_pickle=True).item()
            # If parts is empty, the object is the only part
            part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
            if self.shuffle_parts:
                random.shuffle(part_surfaces)
            part_surfaces = load_surfaces(part_surfaces, num_pc=self.surface_num_points) # [N, P, 6]
            image_path = resolve_path(data_config['image_path'])
            # Robustly load WebP/alpha images and resize
            pil_image = Image.open(image_path)
            # Apply optional rotation on the PIL image first (needs RGB mode)
            if getattr(pil_image, "is_animated", False):
                try:
                    pil_image.seek(0)
                except Exception:
                    pass
            pil_image.load()
            if pil_image.mode in ("RGBA", "LA"):
                bg = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
                pil_image = Image.alpha_composite(bg, pil_image.convert("RGBA")).convert("RGB")
            else:
                pil_image = pil_image.convert("RGB")
            if random.random() < self.rotating_ratio:
                pil_image = self.transform(pil_image)
            pil_image = pil_image.resize(self.image_size, Image.Resampling.BILINEAR)
            image = np.asarray(pil_image, dtype=np.uint8)
            image = torch.from_numpy(image).to(torch.uint8)  # [H, W, 3]
            images = torch.stack([image] * part_surfaces.shape[0], dim=0) # [N, H, W, 3]
            out = {
                "images": images,
                "part_surfaces": part_surfaces,
            }
            return out
        else:
            part_surfaces = []
            for surface_path in data_config['surface_paths']:
                surface_path = resolve_path(surface_path)
                surface_data = np.load(surface_path, allow_pickle=True).item()
                part_surfaces.append(_extract_single_surface_array(surface_data, self.surface_num_points))
            part_surfaces = torch.stack(part_surfaces, dim=0) # [N, P, 6]
            image_path = resolve_path(data_config['image_path'])
            # Robustly load WebP/alpha images and resize
            pil_image = Image.open(image_path)
            # Apply optional rotation on the PIL image first (needs RGB mode)
            if getattr(pil_image, "is_animated", False):
                try:
                    pil_image.seek(0)
                except Exception:
                    pass
            pil_image.load()
            if pil_image.mode in ("RGBA", "LA"):
                bg = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
                pil_image = Image.alpha_composite(bg, pil_image.convert("RGBA")).convert("RGB")
            else:
                pil_image = pil_image.convert("RGB")
            if random.random() < self.rotating_ratio:
                pil_image = self.transform(pil_image)
            pil_image = pil_image.resize(self.image_size, Image.Resampling.BILINEAR)
            image = np.asarray(pil_image, dtype=np.uint8)
            image = torch.from_numpy(image).to(torch.uint8)  # [H, W, 3]
            images = torch.stack([image] * part_surfaces.shape[0], dim=0) # [N, H, W, 3]
            out = {
                "images": images,
                "part_surfaces": part_surfaces,
            }
            return out
    
    def __getitem__(self, idx: int):
        # The dataset can only support batchsize == 1 training. 
        # Because the number of parts is not fixed.
        # Please see BatchedObjaversePartDataset for batched training.
        data_config = self.data_configs[idx]
        cache = getattr(self, "configs", {}).get("dataset_cache", {})
        prefetch_data_configs(self.data_configs, idx + 1, int(cache.get("prefetch_window", 0)))
        try:
            return self._get_data_by_config(data_config)
        except Exception as exc:
            if not (self.spatiotemporal_grid and _is_spatiotemporal_part_count_error(exc)):
                raise

            target_num_parts = data_config.get("num_parts", None)
            self._bad_spatiotemporal_configs.add(_data_config_key(data_config))
            for _ in range(max(self.max_spatiotemporal_grid_retries, 0)):
                replacement = random.choice(self.data_configs)
                if not replacement or replacement.get("num_parts", None) != target_num_parts:
                    continue
                if _data_config_key(replacement) in self._bad_spatiotemporal_configs:
                    continue
                try:
                    return self._get_data_by_config(replacement)
                except Exception as replacement_exc:
                    if not _is_spatiotemporal_part_count_error(replacement_exc):
                        raise
                    self._bad_spatiotemporal_configs.add(_data_config_key(replacement))

            raise RuntimeError(
                f"Could not replace malformed spatio-temporal physics sample after "
                f"{self.max_spatiotemporal_grid_retries} retries; num_parts={target_num_parts}. "
                "The physics data likely needs preprocessing with explicit per-part surfaces."
            ) from exc
        
class BatchedObjaversePartDataset(ObjaversePartDataset):
    def __init__(
        self,
        configs: DictConfig,
        batch_size: int,
        is_main_process: bool = False,
        shuffle: bool = True,
        training: bool = True,
    ):
        assert training
        assert batch_size > 1
        super().__init__(configs, training)
        self.batch_size = batch_size
        self.is_main_process = is_main_process
        # Exclude any object whose sampled parts equal or exceed the batch size.
        # This guarantees we never form a single-object batch with batch_size parts.
        self.data_configs = [config for config in self.data_configs if config['num_parts'] < batch_size]
        
        if shuffle:
            random.shuffle(self.data_configs)

        self.object_configs = [config for config in self.data_configs if config['num_parts'] == 1]
        self.parts_configs = [config for config in self.data_configs if config['num_parts'] > 1]
        
        self.object_ratio = configs['dataset']['object_ratio']
        # Here we keep the ratio of object to parts
        self.object_configs = self.object_configs[:int(len(self.parts_configs) * self.object_ratio)]

        dropped_data_configs = self.parts_configs + self.object_configs
        if shuffle:
            random.shuffle(dropped_data_configs)

        self.data_configs = self._get_batched_configs(dropped_data_configs, batch_size)
    
    def _get_batched_configs(self, data_configs, batch_size):
        batched_data_configs = []
        num_data_configs = len(data_configs)
        progress_bar = tqdm(
            range(len(data_configs)),
            desc="Batching Dataset",
            ncols=125,
            disable=not self.is_main_process,
        )
        while len(data_configs) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen_configs = []
            while temp_num_parts < batch_size and len(data_configs) > 0:
                config = data_configs.pop() # pop the last config
                num_parts = config['num_parts']
                if temp_num_parts + num_parts <= batch_size:
                    temp_batch.append(config)
                    temp_num_parts += num_parts
                    progress_bar.update(1)
                else:
                    unchosen_configs.append(config) # add back to the end
            data_configs = data_configs + unchosen_configs # concat the unchosen configs
            if temp_num_parts == batch_size:
                # Successfully get a batch
                if len(temp_batch) < batch_size:
                    # pad the batch
                    temp_batch += [{}] * (batch_size - len(temp_batch))
                batched_data_configs += temp_batch
                # Else, the code enters here because len(data_configs) == 0
                # which means in the left data_configs, there are no enough 
                # "suitable" configs to form a batch. 
                # Thus, drop the uncompleted batch.
        progress_bar.close()
        return batched_data_configs
        
    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        if len(data_config) == 0:
            # placeholder
            return {}
        cache = getattr(self, "configs", {}).get("dataset_cache", {})
        prefetch_data_configs(self.data_configs, idx + 1, int(cache.get("prefetch_window", 0)))
        try:
            return self._get_data_by_config(data_config)
        except Exception as exc:
            if not (self.spatiotemporal_grid and _is_spatiotemporal_part_count_error(exc)):
                raise

            target_num_parts = data_config.get("num_parts", None)
            self._bad_spatiotemporal_configs.add(_data_config_key(data_config))
            for _ in range(max(self.max_spatiotemporal_grid_retries, 0)):
                replacement = random.choice(self.data_configs)
                if not replacement or replacement.get("num_parts", None) != target_num_parts:
                    continue
                if _data_config_key(replacement) in self._bad_spatiotemporal_configs:
                    continue
                try:
                    return self._get_data_by_config(replacement)
                except Exception as replacement_exc:
                    if not _is_spatiotemporal_part_count_error(replacement_exc):
                        raise
                    self._bad_spatiotemporal_configs.add(_data_config_key(replacement))

            raise RuntimeError(
                f"Could not replace malformed spatio-temporal physics sample after "
                f"{self.max_spatiotemporal_grid_retries} retries; num_parts={target_num_parts}. "
                "The physics data likely needs preprocessing with explicit per-part surfaces."
            ) from exc
    
    def collate_fn(self, batch):
        batch = [data for data in batch if len(data) > 0]
        images = torch.cat([data['images'] for data in batch], dim=0) # [N, H, W, 3]
        surfaces = torch.cat([data['part_surfaces'] for data in batch], dim=0) # [N, P, 6]
        num_parts = torch.LongTensor([data['part_surfaces'].shape[0] for data in batch])
        assert images.shape[0] == surfaces.shape[0] == num_parts.sum() == self.batch_size, \
            f"Batch size mismatch: {images.shape[0]} != {surfaces.shape[0]} != {num_parts.sum()} != {self.batch_size}"
        
        out = {
            "images": images,
            "part_surfaces": surfaces,
            "num_parts": num_parts,
        }
        if all("camera_params" in data for data in batch):
            out["camera_params"] = torch.cat([data["camera_params"] for data in batch], dim=0)
            out["has_camera"] = torch.cat([data["has_camera"] for data in batch], dim=0)
            out["frame_time"] = torch.cat([data["frame_time"] for data in batch], dim=0)
        if all("visibility" in data for data in batch):
            out["visibility"] = torch.cat([data["visibility"] for data in batch], dim=0)
            out["visibility_valid"] = torch.cat([data["visibility_valid"] for data in batch], dim=0)
            out["observation_quality"] = torch.cat([data["observation_quality"] for data in batch], dim=0)
            out["non_border"] = torch.cat([data["non_border"] for data in batch], dim=0)
        if all("object_translation" in data for data in batch):
            out["object_translation"] = torch.cat([data["object_translation"] for data in batch], dim=0)
            out["object_quaternion_xyzw"] = torch.cat([data["object_quaternion_xyzw"] for data in batch], dim=0)
            out["object_pose_valid"] = torch.cat([data["object_pose_valid"] for data in batch], dim=0)
        if all("num_frames" in data and "num_spatial_parts" in data for data in batch):
            out["num_frames"] = torch.cat([data["num_frames"] for data in batch], dim=0)
            out["num_spatial_parts"] = torch.cat([data["num_spatial_parts"] for data in batch], dim=0)
        return out

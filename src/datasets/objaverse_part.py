from src.utils.typing_utils import *

import json
import os
import random
import re

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import accelerate
import torch
import torch.nn.functional as tF
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_surface, load_surfaces
from src.datasets.local_cache import prefetch_data_configs, resolve_path

ROOM_GEOMETRY_KEYS = (
    "house_id",
    "room_id",
    "room_scene_dir",
    "mesh_path",
    "floor_path",
    "wall_path",
    "ceiling_path",
    "depth_path",
    "normal_path",
    "semantic_path",
    "render_meta_path",
    "geometry_metadata_path",
    "num_object_glbs",
)

IMAGE_INDEX_RE = re.compile(r"(?:re)?render_(\d+)\.[^.]+$")
_MESH_NORMALIZATION_CACHE: dict[str, Optional[tuple[torch.Tensor, torch.Tensor]]] = {}


def _image_index(path: str) -> Optional[int]:
    match = IMAGE_INDEX_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else None


def _load_rgb_image(
    image_path: str,
    image_size: tuple[int, int],
    transform=None,
    rotating_ratio: float = 0.0,
) -> torch.Tensor:
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
    if transform is not None and random.random() < rotating_ratio:
        pil_image = transform(pil_image)
    pil_image = pil_image.resize(image_size, Image.Resampling.BILINEAR)
    image = np.array(pil_image, dtype=np.uint8, copy=True)
    return torch.from_numpy(image).to(torch.uint8)


def _resize_tensor_image(tensor: torch.Tensor, image_size: tuple[int, int], mode: str) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor[None, None]
        out = tF.interpolate(tensor.float(), size=image_size, mode=mode)
        return out[0, 0]
    if tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)[None]
        kwargs = {"mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        out = tF.interpolate(tensor.float(), size=image_size, **kwargs)
        return out[0].permute(1, 2, 0)
    raise ValueError(f"Unsupported image tensor shape: {tuple(tensor.shape)}")


def _load_depth_exr(path: str, image_size: tuple[int, int]) -> Optional[torch.Tensor]:
    if not path or not os.path.isfile(path):
        return None
    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2
        depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    except Exception:
        depth = None
    if depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = torch.from_numpy(depth.astype(np.float32, copy=False))
    return _resize_tensor_image(depth, image_size, mode="bilinear")


def _load_normal_image(path: str, image_size: tuple[int, int]) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not path or not os.path.isfile(path):
        return None, None
    arr = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32)
    normal = torch.from_numpy(arr[..., :3] / 255.0 * 2.0 - 1.0)
    normal = _resize_tensor_image(normal, image_size, mode="bilinear")
    normal = tF.normalize(normal, dim=-1, eps=1e-6)
    alpha = torch.from_numpy((arr[..., 3] > 16.0).astype(np.float32))
    alpha = _resize_tensor_image(alpha, image_size, mode="nearest") > 0.5
    return normal, alpha


def _load_semantic_mask(path: str, image_size: tuple[int, int]) -> Optional[torch.Tensor]:
    if not path or not os.path.isfile(path):
        return None
    mask = torch.from_numpy((np.asarray(Image.open(path)) > 0).astype(np.float32))
    return _resize_tensor_image(mask, image_size, mode="nearest") > 0.5


def _mesh_world_to_sdf_normalization(path: str) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    path = resolve_path(path) if path else ""
    if not path or not os.path.isfile(path):
        return None
    if path in _MESH_NORMALIZATION_CACHE:
        return _MESH_NORMALIZATION_CACHE[path]
    try:
        import trimesh

        geom = trimesh.load(path, process=False)
        bounds = np.asarray(geom.bounds, dtype=np.float32)
    except Exception:
        _MESH_NORMALIZATION_CACHE[path] = None
        return None
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        _MESH_NORMALIZATION_CACHE[path] = None
        return None
    extent = bounds[1] - bounds[0]
    max_extent = float(np.max(extent))
    if max_extent <= 1e-6:
        _MESH_NORMALIZATION_CACHE[path] = None
        return None
    center = torch.from_numpy(0.5 * (bounds[0] + bounds[1])).float()
    scale = torch.tensor(2.0 / max_extent, dtype=torch.float32)
    result = (center, scale)
    _MESH_NORMALIZATION_CACHE[path] = result
    return result


def _refine_depth_mask(depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask & torch.isfinite(depth) & (depth > 0.0) & (depth < 1.0e4)
    if not bool(mask.any()):
        return mask
    values = depth[mask]
    median = values.median()
    q90 = torch.quantile(values, 0.90)
    cutoff = torch.maximum(median * 8.0, q90 * 2.0).clamp(max=1000.0)
    return mask & (depth <= cutoff)


def _as_matrix(value, shape: tuple[int, int]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == shape:
        return torch.from_numpy(arr)
    flat = arr.reshape(-1)
    if flat.size == shape[0] * shape[1]:
        return torch.from_numpy(flat.reshape(shape))
    return None


def _first_matrix(values, shape: tuple[int, int]) -> Optional[torch.Tensor]:
    for value in values:
        matrix = _as_matrix(value, shape)
        if matrix is not None:
            return matrix
    return None


def _camera_from_meta(meta: dict, image_path: str, image_size: tuple[int, int], depth_path: str = "") -> Optional[dict[str, torch.Tensor]]:
    if not isinstance(meta, dict):
        return None
    idx = _image_index(image_path)
    image_name = os.path.basename(image_path)
    depth_name = os.path.basename(depth_path)
    candidate_names = {image_name, image_name.replace("rerender_", "render_")}
    if depth_name:
        candidate_names.add(depth_name)
    candidates = []
    if isinstance(meta.get("frames"), list):
        candidates.extend(meta["frames"])
    if isinstance(meta.get("cameras"), list):
        candidates.extend(meta["cameras"])
    if isinstance(meta.get("locations"), list):
        candidates.extend(meta["locations"])
    selected = meta
    if candidates:
        selected = None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            frame_names = {
                os.path.basename(str(frame.get("name", "")))
                for frame in candidate.get("frames", [])
                if isinstance(frame, dict)
            }
            if candidate_names & frame_names:
                selected = candidate
                break
        if selected is None:
            selected = candidates[min(idx, len(candidates) - 1)] if idx is not None else candidates[0]
        if not isinstance(selected, dict):
            selected = meta

    h, w = image_size
    intr = _first_matrix(
        [selected.get("intrinsics"), selected.get("K"), selected.get("camera_intrinsics")],
        (3, 3),
    )
    if intr is None:
        angle_x = selected.get("camera_angle_x", meta.get("camera_angle_x"))
        if angle_x is None:
            return None
        fx = 0.5 * float(w) / max(np.tan(0.5 * float(angle_x)), 1e-6)
        fy = fx
        cx = 0.5 * float(w)
        cy = 0.5 * float(h)
        intr = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    intr = intr.float().clone()
    if intr[0, 2].abs() <= 2.0 and intr[1, 2].abs() <= 2.0 and intr[0, 0].abs() <= 10.0:
        intr[0, :] *= float(w)
        intr[1, :] *= float(h)

    c2w = _first_matrix(
        [
            selected.get("world_from_camera"),
            selected.get("c2w"),
            selected.get("camera_to_world"),
            selected.get("transform_matrix"),
        ],
        (4, 4),
    )
    w2c = _first_matrix(
        [
            selected.get("extrinsics_camera_from_world"),
            selected.get("camera_from_world"),
            selected.get("w2c"),
            selected.get("world_to_camera"),
        ],
        (4, 4),
    )
    if c2w is None and w2c is None:
        return None
    if c2w is None:
        c2w = torch.linalg.inv(w2c.float())
    return {"intrinsics": intr, "camera_to_world": c2w.float()}


def _load_geometry_image_supervision(data_config: dict, image_size: tuple[int, int]) -> Optional[dict[str, torch.Tensor]]:
    depth = _load_depth_exr(resolve_path(data_config.get("depth_path", "")), image_size)
    normal, alpha = _load_normal_image(resolve_path(data_config.get("normal_path", "")), image_size)
    semantic = _load_semantic_mask(resolve_path(data_config.get("semantic_path", "")), image_size)
    if depth is None and normal is None and alpha is None and semantic is None:
        return None

    camera = None
    meta_path = data_config.get("render_meta_path")
    if meta_path:
        try:
            with open(resolve_path(meta_path), "r") as f:
                camera = _camera_from_meta(
                    json.load(f),
                    data_config.get("image_path", ""),
                    image_size,
                    data_config.get("depth_path", ""),
                )
        except Exception:
            camera = None
    if camera is None:
        return None

    mask = torch.ones(image_size, dtype=torch.bool)
    if depth is not None:
        mask = _refine_depth_mask(depth, mask)
    if alpha is not None:
        mask &= alpha
    if semantic is not None:
        mask &= semantic
    if depth is not None:
        mask = _refine_depth_mask(depth, mask)

    normalization = _mesh_world_to_sdf_normalization(data_config.get("mesh_path", ""))
    if normalization is None:
        return None

    payload = {
        "valid": torch.tensor(True),
        "mask": mask,
        "intrinsics": camera["intrinsics"],
        "camera_to_world": camera["camera_to_world"],
        "world_to_sdf_center": normalization[0],
        "world_to_sdf_scale": normalization[1],
    }
    if depth is not None:
        payload["depth"] = depth
    if normal is not None:
        payload["normal"] = normal
    if semantic is not None:
        payload["semantic_mask"] = semantic
    return payload


def _extract_room_geometry(data_config: dict) -> dict:
    geometry = {}
    for key in ROOM_GEOMETRY_KEYS:
        if key in data_config:
            geometry[key] = data_config[key]
    return geometry


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
        self.val_min_num_parts = configs['val']['min_num_parts']
        self.val_max_num_parts = configs['val']['max_num_parts']

        self.max_iou_mean = configs['dataset'].get('max_iou_mean', None)
        self.max_iou_max = configs['dataset'].get('max_iou_max', None)

        self.shuffle_parts = configs['dataset']['shuffle_parts']
        self.training_ratio = configs['dataset']['training_ratio']
        self.balance_object_and_parts = configs['dataset'].get('balance_object_and_parts', False)

        self.rotating_ratio = configs['dataset'].get('rotating_ratio', 0.0)
        self.rotating_degree = configs['dataset'].get('rotating_degree', 10.0)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-self.rotating_degree, self.rotating_degree), fill=(255, 255, 255)),
        ])

        if isinstance(configs['dataset']['config'], ListConfig):
            data_configs = []
            for config in configs['dataset']['config']:
                local_data_configs = json.load(open(config))
                if self.balance_object_and_parts:
                    if self.training:
                        local_data_configs = local_data_configs[:int(len(local_data_configs) * self.training_ratio)]
                    else:
                        local_data_configs = local_data_configs[int(len(local_data_configs) * self.training_ratio):]
                        local_data_configs = [config for config in local_data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
                data_configs += local_data_configs
        else:
            data_configs = json.load(open(configs['dataset']['config']))
        data_configs = [config for config in data_configs if config['valid']]
        data_configs = [config for config in data_configs if self.min_num_parts <= config['num_parts'] <= self.max_num_parts]
        if self.max_iou_mean is not None and self.max_iou_max is not None:
            data_configs = [config for config in data_configs if config['iou_mean'] <= self.max_iou_mean]
            data_configs = [config for config in data_configs if config['iou_max'] <= self.max_iou_max]
        if not self.balance_object_and_parts:
            if self.training:
                data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
            else:
                data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
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
        geometry_image_cfg = configs["train"].get("geometry_image_auxiliary", {})
        self.load_geometry_image_supervision = bool(
            hasattr(geometry_image_cfg, "get") and geometry_image_cfg.get("enabled", False)
        )

    def __len__(self) -> int:
        return len(self.data_configs)
    
    def _get_data_by_config(self, data_config):
        if 'surface_path' in data_config:
            surface_path = resolve_path(data_config['surface_path'])
            surface_data = np.load(surface_path, allow_pickle=True).item()
            # If parts is empty, the object is the only part
            part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
            if self.shuffle_parts:
                random.shuffle(part_surfaces)
            part_surfaces = load_surfaces(part_surfaces, num_pc=self.surface_num_points) # [N, P, 6]
        else:
            part_surfaces = []
            for surface_path in data_config['surface_paths']:
                surface_path = resolve_path(surface_path)
                surface_data = np.load(surface_path, allow_pickle=True).item()
                part_surfaces.append(load_surface(surface_data, num_pc=self.surface_num_points))
            part_surfaces = torch.stack(part_surfaces, dim=0) # [N, P, 6]
        image_path = resolve_path(data_config['image_path'])
        image = _load_rgb_image(
            image_path=image_path,
            image_size=self.image_size,
            transform=self.transform,
            rotating_ratio=self.rotating_ratio,
        )  # [H, W, 3]
        images = torch.stack([image] * part_surfaces.shape[0], dim=0) # [N, H, W, 3]
        out = {
            "images": images,
            "part_surfaces": part_surfaces,
            "room_geometry": _extract_room_geometry(data_config),
        }
        if self.load_geometry_image_supervision:
            supervision = _load_geometry_image_supervision(data_config, self.image_size)
            if supervision is not None:
                out["geometry_image"] = supervision
        return out
    
    def __getitem__(self, idx: int):
        # The dataset can only support batchsize == 1 training. 
        # Because the number of parts is not fixed.
        # Please see BatchedObjaversePartDataset for batched training.
        data_config = self.data_configs[idx]
        cache = getattr(self, "configs", {}).get("dataset_cache", {})
        prefetch_data_configs(self.data_configs, idx + 1, int(cache.get("prefetch_window", 0)))
        data = self._get_data_by_config(data_config)
        return data
        
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
        if batch_size < self.max_num_parts:
            self.data_configs = [config for config in self.data_configs if config['num_parts'] <= batch_size]
        
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
        data = self._get_data_by_config(data_config)
        return data
    
    def collate_fn(self, batch):
        samples = [data for data in batch if len(data) > 0]
        images = torch.cat([data['images'] for data in samples], dim=0) # [N, H, W, 3]
        surfaces = torch.cat([data['part_surfaces'] for data in samples], dim=0) # [N, P, 6]
        num_parts = torch.LongTensor([data['part_surfaces'].shape[0] for data in samples])
        assert images.shape[0] == surfaces.shape[0] == num_parts.sum() == self.batch_size, \
            f"Batch size mismatch: {images.shape[0]} != {surfaces.shape[0]} != {num_parts.sum()} != {self.batch_size}"
        
        batch = {
            "images": images,
            "part_surfaces": surfaces,
            "num_parts": num_parts,
            "room_geometry": [data.get("room_geometry", {}) for data in samples],
        }
        geometry_images = [data.get("geometry_image") for data in samples]
        if all(item is not None for item in geometry_images):
            keys = set.intersection(*(set(item.keys()) for item in geometry_images))
            batch["geometry_image"] = {
                key: torch.stack([item[key] for item in geometry_images], dim=0)
                for key in keys
                if torch.is_tensor(geometry_images[0][key])
            }
        return batch


class ObjaversePartDatasetOriginal(torch.utils.data.Dataset):
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
        self.val_min_num_parts = configs['val']['min_num_parts']
        self.val_max_num_parts = configs['val']['max_num_parts']

        self.max_iou_mean = configs['dataset'].get('max_iou_mean', None)
        self.max_iou_max = configs['dataset'].get('max_iou_max', None)

        self.shuffle_parts = configs['dataset']['shuffle_parts']
        self.training_ratio = configs['dataset']['training_ratio']
        self.balance_object_and_parts = configs['dataset'].get('balance_object_and_parts', False)

        self.rotating_ratio = configs['dataset'].get('rotating_ratio', 0.0)
        self.rotating_degree = configs['dataset'].get('rotating_degree', 10.0)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-self.rotating_degree, self.rotating_degree), fill=(255, 255, 255)),
        ])

        if isinstance(configs['dataset']['config'], ListConfig):
            data_configs = []
            for config in configs['dataset']['config']:
                local_data_configs = json.load(open(config))
                if self.balance_object_and_parts:
                    if self.training:
                        local_data_configs = local_data_configs[:int(len(local_data_configs) * self.training_ratio)]
                    else:
                        local_data_configs = local_data_configs[int(len(local_data_configs) * self.training_ratio):]
                        local_data_configs = [config for config in local_data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
                data_configs += local_data_configs
        else:
            data_configs = json.load(open(configs['dataset']['config']))
        data_configs = [config for config in data_configs if config['valid']]
        data_configs = [config for config in data_configs if self.min_num_parts <= config['num_parts'] <= self.max_num_parts]
        if self.max_iou_mean is not None and self.max_iou_max is not None:
            data_configs = [config for config in data_configs if config['iou_mean'] <= self.max_iou_mean]
            data_configs = [config for config in data_configs if config['iou_max'] <= self.max_iou_max]
        if not self.balance_object_and_parts:
            if self.training:
                data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
            else:
                data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
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

    def _format_data_config(self, data_config: dict) -> str:
        image_path = data_config.get("image_path", "<missing>")
        if 'surface_path' in data_config:
            surface_desc = data_config['surface_path']
        else:
            surface_desc = data_config.get('surface_paths', '<missing>')
        return f"image_path={image_path}, surfaces={surface_desc}"

    def _retry_index(self, failed_idx: int) -> int:
        if len(self.data_configs) <= 1:
            return failed_idx
        new_idx = random.randrange(len(self.data_configs) - 1)
        if new_idx >= failed_idx:
            new_idx += 1
        return new_idx

    def _getitem_with_retry(self, idx: int):
        max_attempts = min(8, max(1, len(self.data_configs)))
        last_exc = None
        last_desc = None
        for attempt in range(max_attempts):
            data_config = self.data_configs[idx]
            try:
                return self._get_data_by_config(data_config)
            except (OSError, ValueError, EOFError) as exc:
                sample_desc = self._format_data_config(data_config)
                if not self.training:
                    raise RuntimeError(
                        f"Failed to load validation sample idx={idx}: {sample_desc}"
                    ) from exc
                print(
                    f"[ObjaversePartDatasetOriginal] Skipping unreadable sample "
                    f"(attempt {attempt + 1}/{max_attempts}) idx={idx}: {sample_desc}. "
                    f"Original error: {exc}"
                )
                last_exc = exc
                last_desc = sample_desc
                idx = self._retry_index(idx)
        raise RuntimeError(
            f"Exceeded retry budget while loading training sample. Last failed sample: {last_desc}"
        ) from last_exc
    
    def _get_data_by_config(self, data_config):
        if 'surface_path' in data_config:
            surface_path = resolve_path(data_config['surface_path'])
            surface_data = np.load(surface_path, allow_pickle=True).item()
            # If parts is empty, the object is the only part
            part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
            if self.shuffle_parts:
                random.shuffle(part_surfaces)
            part_surfaces = load_surfaces(part_surfaces, num_pc=self.surface_num_points) # [N, P, 6]
        else:
            part_surfaces = []
            for surface_path in data_config['surface_paths']:
                surface_path = resolve_path(surface_path)
                surface_data = np.load(surface_path, allow_pickle=True).item()
                part_surfaces.append(load_surface(surface_data, num_pc=self.surface_num_points))
            part_surfaces = torch.stack(part_surfaces, dim=0) # [N, P, 6]
        image_path = resolve_path(data_config['image_path'])
        image = _load_rgb_image(
            image_path=image_path,
            image_size=self.image_size,
            transform=self.transform,
            rotating_ratio=self.rotating_ratio,
        ) # [H, W, 3]
        images = torch.stack([image] * part_surfaces.shape[0], dim=0) # [N, H, W, 3]
        return {
            "images": images,
            "part_surfaces": part_surfaces,
        }
    
    def __getitem__(self, idx: int):
        # The dataset can only support batchsize == 1 training. 
        # Because the number of parts is not fixed.
        # Please see BatchedObjaversePartDataset for batched training.
        cache = getattr(self, "configs", {}).get("dataset_cache", {})
        prefetch_data_configs(self.data_configs, idx + 1, int(cache.get("prefetch_window", 0)))
        return self._getitem_with_retry(idx)
        
class BatchedObjaversePartDatasetOriginal(ObjaversePartDatasetOriginal):
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
        if batch_size < self.max_num_parts:
            self.data_configs = [config for config in self.data_configs if config['num_parts'] <= batch_size]
        
        if shuffle:
            random.shuffle(self.data_configs)

        self.object_configs = [config for config in self.data_configs if config['num_parts'] == 1]
        self.parts_configs = [config for config in self.data_configs if config['num_parts'] > 1]
        
        self.object_ratio = configs['dataset']['object_ratio']
        # Here we keep the ratio of object to parts
        # self.object_configs = self.object_configs[:int(len(self.parts_configs) * self.object_ratio)]

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
        return self._getitem_with_retry(idx)
    
    def collate_fn(self, batch):
        batch = [data for data in batch if len(data) > 0]
        images = torch.cat([data['images'] for data in batch], dim=0) # [N, H, W, 3]
        surfaces = torch.cat([data['part_surfaces'] for data in batch], dim=0) # [N, P, 6]
        num_parts = torch.LongTensor([data['part_surfaces'].shape[0] for data in batch])
        assert images.shape[0] == surfaces.shape[0] == num_parts.sum() == self.batch_size
        batch = {
            "images": images,
            "part_surfaces": surfaces,
            "num_parts": num_parts,
        }
        return batch

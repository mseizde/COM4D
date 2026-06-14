import trimesh
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Literal, Optional, Tuple, Union


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def axis_index(axis: Literal["x", "y", "z"]) -> int:
    if axis not in AXIS_TO_INDEX:
        raise ValueError(f"Unsupported axis {axis!r}; expected one of {sorted(AXIS_TO_INDEX)}.")
    return AXIS_TO_INDEX[axis]


def load_mesh_or_scene(path: Union[str, Path]) -> Union[trimesh.Trimesh, trimesh.Scene]:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, (trimesh.Trimesh, trimesh.Scene)):
        return loaded
    raise TypeError(f"Unsupported geometry type loaded from {path}: {type(loaded)!r}")


def scene_to_meshes(scene: Union[trimesh.Trimesh, trimesh.Scene, List[trimesh.Trimesh]]) -> List[trimesh.Trimesh]:
    if isinstance(scene, trimesh.Trimesh):
        return [scene]
    if isinstance(scene, trimesh.Scene):
        return [
            mesh
            for mesh in scene.dump(concatenate=False)
            if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0
        ]
    if isinstance(scene, list) and all(isinstance(mesh, trimesh.Trimesh) for mesh in scene):
        return [mesh for mesh in scene if len(mesh.vertices) > 0]
    raise TypeError(f"Expected Trimesh, Scene, or list[Trimesh], got {type(scene)!r}.")


def scene_to_single_mesh(scene: Union[trimesh.Trimesh, trimesh.Scene, List[trimesh.Trimesh]]) -> trimesh.Trimesh:
    meshes = scene_to_meshes(scene)
    if not meshes:
        raise ValueError("Cannot concatenate an empty scene.")
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def bounds_from_mesh_or_scene(mesh_or_scene: Union[trimesh.Trimesh, trimesh.Scene, List[trimesh.Trimesh]]) -> np.ndarray:
    if isinstance(mesh_or_scene, list):
        meshes = scene_to_meshes(mesh_or_scene)
        if not meshes:
            raise ValueError("Cannot compute bounds for an empty mesh list.")
        mins = np.stack([np.asarray(mesh.bounds[0], dtype=np.float64) for mesh in meshes], axis=0)
        maxs = np.stack([np.asarray(mesh.bounds[1], dtype=np.float64) for mesh in meshes], axis=0)
        bounds = np.stack([mins.min(axis=0), maxs.max(axis=0)], axis=0)
    else:
        bounds = np.asarray(mesh_or_scene.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("Geometry has invalid bounds.")
    return bounds


def center_from_bounds(bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.float64)
    return 0.5 * (bounds[0] + bounds[1])


def size_from_bounds(bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.float64)
    return bounds[1] - bounds[0]


def bbox_overlap_extents(bounds_a: np.ndarray, bounds_b: np.ndarray) -> np.ndarray:
    a = np.asarray(bounds_a, dtype=np.float64)
    b = np.asarray(bounds_b, dtype=np.float64)
    return np.maximum(0.0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]))


def bbox_volume(bounds: np.ndarray) -> float:
    size = np.maximum(size_from_bounds(bounds), 0.0)
    return float(np.prod(size))


def bbox_overlap_volume(bounds_a: np.ndarray, bounds_b: np.ndarray) -> float:
    return float(np.prod(bbox_overlap_extents(bounds_a, bounds_b)))


def bbox_iou_3d(bounds_a: np.ndarray, bounds_b: np.ndarray) -> float:
    intersection = bbox_overlap_volume(bounds_a, bounds_b)
    union = bbox_volume(bounds_a) + bbox_volume(bounds_b) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def bbox_overlap_area_xz(bounds_a: np.ndarray, bounds_b: np.ndarray) -> float:
    overlap = bbox_overlap_extents(bounds_a, bounds_b)
    return float(overlap[0] * overlap[2])


def floor_support_error(
    bounds: np.ndarray,
    floor_height: float = 0.0,
    up_axis: Literal["x", "y", "z"] = "y",
) -> float:
    up_idx = axis_index(up_axis)
    return float(np.asarray(bounds, dtype=np.float64)[0, up_idx] - floor_height)


def floor_penetration_depth(
    bounds: np.ndarray,
    floor_height: float = 0.0,
    up_axis: Literal["x", "y", "z"] = "y",
) -> float:
    return float(max(0.0, -floor_support_error(bounds, floor_height, up_axis)))


def is_floating(
    bounds: np.ndarray,
    floor_height: float = 0.0,
    tolerance: float = 0.05,
    up_axis: Literal["x", "y", "z"] = "y",
) -> bool:
    return bool(floor_support_error(bounds, floor_height, up_axis) > tolerance)


def trajectory_speed_stats(centers: np.ndarray, fps: float) -> Dict[str, float]:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape [num_frames, 3].")
    if len(centers) < 2:
        return {"mean": 0.0, "max": 0.0, "std": 0.0}
    speeds = np.linalg.norm(np.diff(centers, axis=0), axis=1) * float(fps)
    return {
        "mean": float(np.mean(speeds)),
        "max": float(np.max(speeds)),
        "std": float(np.std(speeds)),
    }


def trajectory_acceleration_stats(centers: np.ndarray, fps: float) -> Dict[str, float]:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape [num_frames, 3].")
    if len(centers) < 3:
        return {"mean": 0.0, "max": 0.0, "std": 0.0}
    velocities = np.diff(centers, axis=0) * float(fps)
    accelerations = np.linalg.norm(np.diff(velocities, axis=0), axis=1) * float(fps)
    return {
        "mean": float(np.mean(accelerations)),
        "max": float(np.max(accelerations)),
        "std": float(np.std(accelerations)),
    }


def best_yaw_angle(correlation: np.ndarray) -> float:
    """Return the yaw angle that maximizes trace(Rz(theta) * correlation)."""
    correlation = np.asarray(correlation, dtype=np.float64)
    if correlation.shape != (3, 3):
        raise ValueError("correlation must have shape (3, 3).")
    a = correlation[0, 1] - correlation[1, 0]
    b = correlation[0, 0] + correlation[1, 1]
    return float(np.pi / 2.0 - np.arctan2(b, a))


def rotation_matrix_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def align_umeyama(
    model: np.ndarray,
    data: np.ndarray,
    known_scale: bool = False,
    yaw_only: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return s, R, t minimizing ||model - (s * R @ data + t)||^2.

    This is the closed-form Umeyama similarity alignment used by ATE-style
    trajectory metrics. Pass ``known_scale=True`` for SE(3) alignment.
    """
    model = np.asarray(model, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if model.shape != data.shape or model.ndim != 2 or model.shape[1] != 3:
        raise ValueError("model and data must both have shape (N, 3).")
    if len(model) == 0:
        raise ValueError("Umeyama alignment requires at least one point.")
    if not np.isfinite(model).all() or not np.isfinite(data).all():
        raise ValueError("model and data must contain only finite values.")

    n = len(model)
    mu_model = model.mean(axis=0)
    mu_data = data.mean(axis=0)
    model_centered = model - mu_model
    data_centered = data - mu_data

    covariance = model_centered.T @ data_centered / n
    sigma2_data = float(np.mean(np.sum(data_centered**2, axis=1)))
    u, singular_values, vt = np.linalg.svd(covariance)

    w = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt.T) < 0.0:
        w[2, 2] = -1.0

    if yaw_only:
        rot_correlation = data_centered.T @ model_centered
        rotation = rotation_matrix_z(best_yaw_angle(rot_correlation))
    else:
        rotation = u @ w @ vt

    if known_scale or sigma2_data <= 0.0:
        scale = 1.0
    else:
        scale = float(np.trace(np.diag(singular_values) @ w) / sigma2_data)
    translation = mu_model - scale * rotation @ mu_data
    return scale, rotation, translation


def similarity_transform_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    known_scale: bool = False,
    yaw_only: bool = False,
) -> np.ndarray:
    """Return a 4x4 transform mapping src points to dst points."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must both have shape (N, 3).")
    if len(src) == 0:
        return np.eye(4, dtype=np.float64)
    scale, rotation, translation = align_umeyama(dst, src, known_scale=known_scale, yaw_only=yaw_only)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def sample_from_mesh(
    mesh: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
):
    if num_samples is None:
        return mesh.vertices
    else:
        return mesh.sample(num_samples)

def sample_two_meshes(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
):
    points1 = sample_from_mesh(mesh1, num_samples)
    points2 = sample_from_mesh(mesh2, num_samples)
    return points1, points2

def compute_nearest_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    metric: str = 'l2'
) -> np.ndarray:
    # Compute nearest neighbor distance from points1 to points2
    nn = NearestNeighbors(n_neighbors=1, leaf_size=30, algorithm='kd_tree', metric=metric).fit(points2)
    min_dist = nn.kneighbors(points1)[0]
    return min_dist

def compute_mutual_nearest_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    metric: str = 'l2'
) -> np.ndarray:
    min_1_to_2 = compute_nearest_distance(points1, points2, metric=metric)
    min_2_to_1 = compute_nearest_distance(points2, points1, metric=metric)
    return min_1_to_2, min_2_to_1

def compute_mutual_nearest_distance_for_meshes(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
    metric: str = 'l2'
) -> Tuple[np.ndarray, np.ndarray]:
    points1 = sample_from_mesh(mesh1, num_samples)
    points2 = sample_from_mesh(mesh2, num_samples)
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance(points1, points2, metric=metric)
    return min_1_to_2, min_2_to_1

def compute_chamfer_distance(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: int = 10000,
    metric: str = 'l2'
):
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance_for_meshes(mesh1, mesh2, num_samples, metric=metric)
    chamfer_dist = np.mean(min_2_to_1) + np.mean(min_1_to_2)
    return chamfer_dist

def compute_f_score(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: int = 10000,
    threshold: float = 0.1,
    metric: str = 'l2'
):
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance_for_meshes(mesh1, mesh2, num_samples, metric=metric)
    precision_1 = np.mean((min_1_to_2 < threshold).astype(np.float32))
    precision_2 = np.mean((min_2_to_1 < threshold).astype(np.float32))
    denom = precision_1 + precision_2
    fscore = 0.0 if denom == 0.0 else 2 * precision_1 * precision_2 / denom
    return fscore

def compute_cd_and_f_score(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
    threshold: float = 0.1,
    metric: str = 'l2'
):
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance_for_meshes(mesh1, mesh2, num_samples, metric=metric)
    chamfer_dist = np.mean(min_2_to_1) + np.mean(min_1_to_2)
    precision_1 = np.mean((min_1_to_2 < threshold).astype(np.float32))
    precision_2 = np.mean((min_2_to_1 < threshold).astype(np.float32))
    denom = precision_1 + precision_2
    fscore = 0.0 if denom == 0.0 else 2 * precision_1 * precision_2 / denom
    return chamfer_dist, fscore

def compute_cd_and_f_score_in_training(
    gt_surface: np.ndarray,
    pred_mesh: trimesh.Trimesh,
    num_samples: int = 204800,
    threshold: float = 0.1,
    metric: str = 'l2'
):
    gt_points = gt_surface[:, :3]
    num_samples = max(num_samples, gt_points.shape[0])
    gt_points = gt_points[np.random.choice(gt_points.shape[0], num_samples, replace=False)]
    pred_points = sample_from_mesh(pred_mesh, num_samples)
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance(gt_points, pred_points, metric=metric)
    chamfer_dist = np.mean(min_2_to_1) + np.mean(min_1_to_2)
    precision_1 = np.mean((min_1_to_2 < threshold).astype(np.float32))
    precision_2 = np.mean((min_2_to_1 < threshold).astype(np.float32))
    denom = precision_1 + precision_2
    fscore = 0.0 if denom == 0.0 else 2 * precision_1 * precision_2 / denom
    return chamfer_dist, fscore

def get_voxel_set(
    mesh: trimesh.Trimesh,
    num_grids: int = 64,
    scale: float = 2.0,
):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("mesh must be a trimesh.Trimesh object")
    pitch = scale / num_grids
    voxel_girds: trimesh.voxel.base.VoxelGrid = mesh.voxelized(pitch=pitch).fill()
    voxels = set(map(tuple, np.round(voxel_girds.points / pitch).astype(int)))
    return voxels

def compute_IoU(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_grids: int = 64,
    scale: float = 2.0,
):
    if not isinstance(mesh1, trimesh.Trimesh) or not isinstance(mesh2, trimesh.Trimesh):
        raise ValueError("mesh1 and mesh2 must be trimesh.Trimesh objects")
    voxels1 = get_voxel_set(mesh1, num_grids, scale)
    voxels2 = get_voxel_set(mesh2, num_grids, scale)
    intersection = voxels1 & voxels2
    union = voxels1 | voxels2
    iou = len(intersection) / len(union) if len(union) > 0 else 0.0
    return iou

def compute_IoU_for_scene(
    scene: Union[trimesh.Scene, List[trimesh.Trimesh]],
    num_grids: int = 64,
    scale: float = 2.0,
    return_type: Literal["iou", "iou_list"] = "iou",
):
    if isinstance(scene, trimesh.Scene):
        scene = scene.dump()
    if isinstance(scene, list) and len(scene) > 1 and isinstance(scene[0], trimesh.Trimesh):
        meshes = scene
    else:
        raise ValueError("scene must be a trimesh.Scene object or a list of trimesh.Trimesh objects")
    ious = []
    for i in range(len(meshes)):
        for j in range(i+1, len(meshes)):
            iou = compute_IoU(meshes[i], meshes[j], num_grids, scale)
            ious.append(iou)
    if return_type == "iou":
        return np.mean(ious)
    elif return_type == "iou_list":
        return ious
    else:
        raise ValueError("return_type must be 'iou' or 'iou_list'")

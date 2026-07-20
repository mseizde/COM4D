#!/usr/bin/env python3

"""Convert synthetic physics pipeline outputs into COM4D 4D training data.

Input layout can be either one sequence directory:
  outputs/two_ball_dataset/sample_000001/physics_metadata.json

or a root containing many sequence directories:
  outputs/two_ball_dataset/sample_000001/physics_metadata.json
  outputs/two_ball_dataset/sample_000002/physics_metadata.json

This writes:
  <output-root>/glb/<sequence>/frame_0000.glb
  <output-root>/preprocessed/<sequence>_frame_0000/points.npy
  <output-root>/render/<sequence>/frame_0000.png
  <json-output>

The JSON format matches src/datasets/animated_frame.py.

The preprocessor is backward-compatible with the legacy two-ball metadata, and
also supports generalized metadata with an ``objects`` dictionary. In generalized
metadata, all objects with ``dynamic: true`` are reconstructed as parts. Static
objects can optionally be included as interaction parts for physics-focused
training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "two_ball_dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "processed" / "two_ball"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "dataset_json" / "two_ball.json"

sys.path.append(str(PROJECT_ROOT))
from src.utils.data_utils import mesh_to_surface, normalize_mesh  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Exact sequence directory name to include. Can be repeated; applied after --sequence-glob.",
    )
    parser.add_argument("--num-points", type=int, default=204800)
    parser.add_argument("--sphere-subdivisions", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy render_rgb frames into the training render layout.",
    )
    parser.add_argument(
        "--copy-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean and copy paired visible/amodal masks when both are present.",
    )
    parser.add_argument(
        "--require-masks",
        action="store_true",
        help="Fail if any included part lacks a visible/amodal mask pair.",
    )
    parser.add_argument(
        "--write-glb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write posed per-frame GLBs for inspection/reuse.",
    )
    parser.add_argument(
        "--include-parts",
        action="store_true",
        help="Also store per-ball surface samples under the points.npy 'parts' key.",
    )
    parser.add_argument(
        "--include-static-parts",
        action="store_true",
        help="Include interaction-relevant static geometry after dynamic parts: floor, wall, and occluder_box when present.",
    )
    parser.add_argument(
        "--floor-size",
        type=float,
        default=5.0,
        help="Side length for generated floor plane when --include-static-parts is enabled.",
    )
    parser.add_argument(
        "--normalize-surfaces",
        action="store_true",
        help="Center/scale every sampled surface independently to bbox max extent 2 before writing points.npy.",
    )
    parser.add_argument(
        "--sequence-normalize-surfaces",
        action="store_true",
        help=(
            "Apply one shared center/scale transform per sequence before point sampling. "
            "This keeps motion in coordinates while fitting the sequence in the VAE box."
        ),
    )
    return parser.parse_args()


def find_sequence_dirs(input_root: Path, sequence_glob: str) -> list[Path]:
    input_root = input_root.expanduser().resolve()
    if (input_root / "physics_metadata.json").is_file():
        return [input_root]
    return sorted(
        path
        for path in input_root.glob(sequence_glob)
        if path.is_dir() and (path / "physics_metadata.json").is_file()
    )


def frame_name(frame_idx: int) -> str:
    return f"frame_{frame_idx:04d}"


def get_radius(metadata: dict, ball_name: str) -> float:
    return float(metadata.get("ball_radii", {}).get(ball_name, metadata.get("ball_radius", 0.25)))


def legacy_object_specs(metadata: dict) -> dict[str, dict]:
    return {
        "ball_0": {
            "type": "sphere",
            "dynamic": True,
            "radius": get_radius(metadata, "ball_0"),
        },
        "ball_1": {
            "type": "sphere",
            "dynamic": True,
            "radius": get_radius(metadata, "ball_1"),
        },
    }


def object_specs(metadata: dict) -> dict[str, dict]:
    objects = metadata.get("objects")
    if isinstance(objects, dict) and objects:
        return objects
    render_objects = metadata.get("render_objects")
    object_properties = metadata.get("object_properties")
    if isinstance(render_objects, list) and isinstance(object_properties, dict):
        radius = float(object_properties.get("ball_radius", metadata.get("ball_radius", 0.25)))
        specs = {}
        for name in render_objects:
            if not isinstance(name, str):
                continue
            if name.startswith("ball_"):
                specs[name] = {"type": "sphere", "dynamic": True, "radius": radius}
            elif name in {"wall", "occluder_box"}:
                half_extents = object_properties.get(f"{name}_half_extents", [0.5, 0.5, 0.5])
                specs[name] = {
                    "type": "box",
                    "dynamic": False,
                    "size": [2.0 * float(value) for value in half_extents],
                }
            elif name == "floor":
                specs[name] = {"type": "plane", "dynamic": False}
        if specs:
            return specs
    return legacy_object_specs(metadata)


def dynamic_object_specs(metadata: dict) -> dict[str, dict]:
    return {
        name: spec
        for name, spec in object_specs(metadata).items()
        if bool(spec.get("dynamic", True))
    }


def interaction_static_object_specs(metadata: dict) -> dict[str, dict]:
    specs = object_specs(metadata)
    render_order = metadata.get("render_objects")
    names = [name for name in render_order if isinstance(name, str)] if isinstance(render_order, list) else list(specs)
    wanted = {"floor", "wall", "occluder_box"}
    return {
        name: specs[name]
        for name in names
        if name in wanted and name in specs and not bool(specs[name].get("dynamic", name.startswith("ball_")))
    }


def part_object_specs(metadata: dict, include_static_parts: bool) -> dict[str, dict]:
    specs = dict(dynamic_object_specs(metadata))
    if include_static_parts:
        specs.update(interaction_static_object_specs(metadata))
    return specs


def clean_binary_mask(source: Path, destination: Path) -> int:
    values = np.asarray(Image.open(source).convert("L"))
    binary = np.where(values == 255, 255, 0).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary).save(destination)
    return int(np.count_nonzero(binary))


def transform_matrix(position: list[float], quat_xyzw: list[float]) -> np.ndarray:
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    translation = np.eye(4)
    translation[:3, 3] = np.asarray(position, dtype=np.float64)
    return translation @ rotation


def object_pose(name: str, spec: dict, frame: dict) -> tuple[list[float], list[float]]:
    frame_objects = frame.get("objects")
    frame_pose = frame_objects.get(name) if isinstance(frame_objects, dict) else frame.get(name)
    if isinstance(frame_pose, dict):
        position = frame_pose.get("position", spec.get("position", [0.0, 0.0, 0.0]))
        quaternion = frame_pose.get("quaternion", spec.get("quaternion", [0.0, 0.0, 0.0, 1.0]))
        return list(position), list(quaternion)
    return list(spec.get("position", [0.0, 0.0, 0.0])), list(spec.get("quaternion", [0.0, 0.0, 0.0, 1.0]))


def make_sphere_mesh(spec: dict, position: list[float], quat_xyzw: list[float], subdivisions: int) -> trimesh.Trimesh:
    radius = float(spec.get("radius", 0.25))
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_transform(transform_matrix(position, quat_xyzw))
    return mesh


def make_box_mesh(spec: dict, position: list[float], quat_xyzw: list[float]) -> trimesh.Trimesh:
    size = spec.get("size", spec.get("extents", [1.0, 1.0, 1.0]))
    mesh = trimesh.creation.box(extents=np.asarray(size, dtype=np.float64))
    mesh.apply_transform(transform_matrix(position, quat_xyzw))
    return mesh


def make_floor_mesh(size: float) -> trimesh.Trimesh:
    half = float(size) * 0.5
    return trimesh.Trimesh(
        vertices=[
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )


def make_object_mesh(
    name: str,
    spec: dict,
    frame: dict,
    sphere_subdivisions: int,
    floor_size: float,
) -> trimesh.Trimesh:
    position, quat_xyzw = object_pose(name, spec, frame)
    object_type = str(spec.get("type", "sphere")).lower()
    if object_type in {"sphere", "ball"}:
        return make_sphere_mesh(spec, position, quat_xyzw, sphere_subdivisions)
    if object_type in {"box", "cube", "cuboid"}:
        return make_box_mesh(spec, position, quat_xyzw)
    if object_type in {"plane", "floor"}:
        mesh = make_floor_mesh(floor_size)
        mesh.apply_transform(transform_matrix(position, quat_xyzw))
        return mesh
    raise ValueError(f"Unsupported object type for {name}: {object_type!r}")


def build_frame_meshes(
    metadata: dict,
    frame: dict,
    subdivisions: int,
    include_static_parts: bool,
    floor_size: float,
) -> list[tuple[str, trimesh.Trimesh]]:
    specs = part_object_specs(metadata, include_static_parts)
    if not specs:
        raise ValueError("Metadata does not define any dynamic objects.")
    return [
        (name, make_object_mesh(name, spec, frame, subdivisions, floor_size))
        for name, spec in specs.items()
    ]


def compute_sequence_normalization(
    metadata: dict,
    frames: list[dict],
    subdivisions: int,
    include_static_parts: bool,
    floor_size: float,
) -> tuple[np.ndarray, float]:
    min_corner = None
    max_corner = None
    for frame in frames:
        named_part_meshes = build_frame_meshes(
            metadata,
            frame,
            subdivisions,
            include_static_parts=include_static_parts,
            floor_size=floor_size,
        )
        for _, mesh in named_part_meshes:
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
                continue
            min_corner = bounds[0] if min_corner is None else np.minimum(min_corner, bounds[0])
            max_corner = bounds[1] if max_corner is None else np.maximum(max_corner, bounds[1])

    if min_corner is None or max_corner is None:
        return np.zeros(3, dtype=np.float64), 1.0
    center = (min_corner + max_corner) * 0.5
    extent = float(np.max(max_corner - min_corner))
    scale = 1.9 / max(extent, 1e-8)
    return center, scale


def apply_shared_normalization(mesh: trimesh.Trimesh, center: np.ndarray, scale: float) -> trimesh.Trimesh:
    transformed = mesh.copy()
    transformed.apply_translation(-center)
    transformed.apply_scale(float(scale))
    return transformed


def mesh_normalization_metadata(mesh: trimesh.Trimesh) -> dict[str, object]:
    bbox = mesh.bounding_box
    center = np.asarray(bbox.centroid, dtype=np.float32)
    extent = float(np.asarray(bbox.primitive.extents, dtype=np.float64).max())
    return {
        "center": center.tolist(),
        "scale": float(2.0 / max(extent, 1e-8)),
        "extent": extent,
    }


def surface_dict(mesh: trimesh.Trimesh, num_points: int, normalize: bool = False) -> tuple[dict[str, np.ndarray], dict[str, object] | None]:
    metadata = None
    sample_mesh = mesh
    if normalize:
        metadata = mesh_normalization_metadata(mesh)
        sample_mesh = normalize_mesh(mesh.copy(), scale=2.0)
    data = mesh_to_surface(sample_mesh, num_pc=num_points, return_dict=True)
    surface = {
        "surface_points": np.asarray(data["surface_points"], dtype=np.float32),
        "surface_normals": np.asarray(data["surface_normals"], dtype=np.float32),
    }
    return surface, metadata


def write_points(
    output_path: Path,
    object_mesh: trimesh.Trimesh,
    part_meshes: list[trimesh.Trimesh],
    num_points: int,
    include_parts: bool,
    normalize_surfaces: bool,
) -> None:
    object_surface, object_norm = surface_dict(object_mesh, num_points, normalize=normalize_surfaces)
    data = {
        "object": object_surface,
        "parts": [],
    }
    normalization = {"object": object_norm, "parts": []} if normalize_surfaces else None
    if include_parts:
        part_surfaces = []
        part_norms = []
        for part in part_meshes:
            part_surface, part_norm = surface_dict(part, num_points, normalize=normalize_surfaces)
            part_surfaces.append(part_surface)
            part_norms.append(part_norm)
        data["parts"] = part_surfaces
        if normalize_surfaces:
            normalization["parts"] = part_norms
    if normalize_surfaces:
        data["normalization"] = normalization
    np.save(output_path, data)


def process_sequence(
    sequence_dir: Path,
    output_root: Path,
    num_points: int,
    sphere_subdivisions: int,
    frame_limit: int | None,
    overwrite: bool,
    copy_rgb: bool,
    copy_masks: bool,
    require_masks: bool,
    write_glb: bool,
    include_parts: bool,
    include_static_parts: bool,
    floor_size: float,
    normalize_surfaces: bool,
    sequence_normalize_surfaces: bool,
) -> tuple[str, list[dict]]:
    sequence_name = sequence_dir.name
    metadata_path = sequence_dir / "physics_metadata.json"
    with metadata_path.open("r") as f:
        metadata = json.load(f)
    ordered_specs = part_object_specs(metadata, include_static_parts)
    part_names = list(ordered_specs)

    processed_metadata_path = output_root / "metadata" / sequence_name / "physics_metadata.json"
    processed_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not processed_metadata_path.exists():
        shutil.copy2(metadata_path, processed_metadata_path)

    frames = metadata["frames"]
    if frame_limit is not None:
        frames = frames[:frame_limit]

    sequence_center = np.zeros(3, dtype=np.float64)
    sequence_scale = 1.0
    if sequence_normalize_surfaces:
        sequence_center, sequence_scale = compute_sequence_normalization(
            metadata,
            frames,
            sphere_subdivisions,
            include_static_parts=include_static_parts,
            floor_size=floor_size,
        )

    glb_dir = output_root / "glb" / sequence_name
    preproc_root = output_root / "preprocessed"
    render_dir = output_root / "render" / sequence_name
    if write_glb:
        glb_dir.mkdir(parents=True, exist_ok=True)
    if copy_rgb:
        render_dir.mkdir(parents=True, exist_ok=True)
    visible_output_root = output_root / "masks_visible" / sequence_name
    amodal_output_root = output_root / "masks_amodal" / sequence_name
    preproc_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for frame in frames:
        idx = int(frame["frame"])
        name = frame_name(idx)
        frame_preproc_dir = preproc_root / f"{sequence_name}_{name}"
        points_path = frame_preproc_dir / "points.npy"
        num_parts_path = frame_preproc_dir / "num_parts.json"
        frame_preproc_dir.mkdir(parents=True, exist_ok=True)

        if overwrite or not points_path.exists():
            named_part_meshes = build_frame_meshes(
                metadata,
                frame,
                sphere_subdivisions,
                include_static_parts=include_static_parts,
                floor_size=floor_size,
            )
            if sequence_normalize_surfaces:
                named_part_meshes = [
                    (part_name, apply_shared_normalization(mesh, sequence_center, sequence_scale))
                    for part_name, mesh in named_part_meshes
                ]
            part_meshes = [mesh for _, mesh in named_part_meshes]
            object_mesh = trimesh.util.concatenate(part_meshes) if len(part_meshes) > 1 else part_meshes[0].copy()
            write_points(
                output_path=points_path,
                object_mesh=object_mesh,
                part_meshes=part_meshes,
                num_points=num_points,
                include_parts=include_parts,
                normalize_surfaces=normalize_surfaces,
            )

            with num_parts_path.open("w") as f:
                json.dump(
                    {
                        "num_parts": len(part_meshes),
                        "part_names": [part_name for part_name, _ in named_part_meshes],
                        "mesh_path": str((glb_dir / f"{name}.glb").resolve()) if write_glb else None,
                        "source_metadata": str(processed_metadata_path.resolve()),
                        "sequence_normalization": {
                            "center": sequence_center.tolist(),
                            "scale": float(sequence_scale),
                            "enabled": bool(sequence_normalize_surfaces),
                        },
                    },
                    f,
                    separators=(",", ":"),
                )

            if write_glb:
                scene = trimesh.Scene()
                for part_name, part_mesh in named_part_meshes:
                    scene.add_geometry(part_mesh, geom_name=part_name)
                scene.export(glb_dir / f"{name}.glb", file_type="glb")

        object_translation = []
        object_quaternion = []
        object_linear_velocity = []
        object_angular_velocity = []
        frame_objects = frame.get("objects", {})
        for part_name in part_names:
            spec = ordered_specs[part_name]
            position, quaternion = object_pose(part_name, spec, frame)
            quaternion_array = np.asarray(quaternion, dtype=np.float64)
            quaternion_array /= max(float(np.linalg.norm(quaternion_array)), 1e-8)
            state = frame_objects.get(part_name, {}) if isinstance(frame_objects, dict) else {}
            if sequence_normalize_surfaces:
                position = (np.asarray(position, dtype=np.float64) - sequence_center) * sequence_scale
            object_translation.append([float(value) for value in position])
            object_quaternion.append(quaternion_array.tolist())
            linear_velocity = np.asarray(state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
            if sequence_normalize_surfaces:
                linear_velocity = linear_velocity * sequence_scale
            object_linear_velocity.append([float(value) for value in linear_velocity])
            object_angular_velocity.append(
                [float(value) for value in state.get("angular_velocity", [0.0, 0.0, 0.0])]
            )

        visibility = []
        visibility_valid = []
        visible_mask_paths = []
        amodal_mask_paths = []
        for part_name in part_names:
            visible_source = sequence_dir / "masks" / part_name / f"{name}.png"
            amodal_source = sequence_dir / "masks_amodal" / part_name / f"{name}.png"
            valid_pair = copy_masks and visible_source.is_file() and amodal_source.is_file()
            if valid_pair:
                visible_destination = visible_output_root / part_name / f"{name}.png"
                amodal_destination = amodal_output_root / part_name / f"{name}.png"
                visible_area = clean_binary_mask(visible_source, visible_destination)
                amodal_area = clean_binary_mask(amodal_source, amodal_destination)
                visibility.append(float(np.clip(visible_area / max(amodal_area, 1), 0.0, 1.0)))
                visibility_valid.append(amodal_area > 0)
                visible_mask_paths.append(str(visible_destination.resolve()))
                amodal_mask_paths.append(str(amodal_destination.resolve()))
            else:
                if require_masks:
                    raise FileNotFoundError(
                        f"Missing visible/amodal mask pair for {sequence_name}/{part_name}/{name}"
                    )
                visibility.append(0.0)
                visibility_valid.append(False)
                visible_mask_paths.append(None)
                amodal_mask_paths.append(None)


        src_rgb = sequence_dir / "render_rgb" / f"{name}.png"
        dst_rgb = render_dir / f"{name}.png"
        if copy_rgb:
            if not src_rgb.exists():
                raise FileNotFoundError(f"Missing RGB frame: {src_rgb}")
            if overwrite or not dst_rgb.exists():
                shutil.copy2(src_rgb, dst_rgb)
            image_path = dst_rgb
        else:
            image_path = src_rgb

        entries.append(
            {
                "surface_path": str(points_path.resolve()),
                "image_path": str(image_path.resolve()),
                "object_names": part_names,
                "object_translation": object_translation,
                "object_quaternion_xyzw": object_quaternion,
                "object_linear_velocity": object_linear_velocity,
                "object_angular_velocity": object_angular_velocity,
                "visibility": visibility,
                "visibility_valid": visibility_valid,
                "visible_mask_paths": visible_mask_paths,
                "amodal_mask_paths": amodal_mask_paths,
                "physics_metadata_path": str(processed_metadata_path.resolve()),
                "iou_mean": 0.0,
                "iou_max": 0.0,
            }
        )

    return sequence_name, entries


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    json_output = args.json_output.expanduser().resolve()
    if args.normalize_surfaces and args.sequence_normalize_surfaces:
        raise ValueError("Use either --normalize-surfaces or --sequence-normalize-surfaces, not both.")

    sequence_dirs = find_sequence_dirs(input_root, args.sequence_glob)
    if args.sequence:
        wanted = set(args.sequence)
        sequence_dirs = [path for path in sequence_dirs if path.name in wanted]
        missing = sorted(wanted - {path.name for path in sequence_dirs})
        if missing:
            raise FileNotFoundError(f"Requested sequence(s) not found under {input_root}: {', '.join(missing)}")
    if not sequence_dirs:
        raise FileNotFoundError(f"No synthetic physics sequences found under {input_root}")

    dataset_index = {}
    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        results = [
            process_sequence(
                sequence_dir=sequence_dir,
                output_root=output_root,
                num_points=args.num_points,
                sphere_subdivisions=args.sphere_subdivisions,
                frame_limit=args.frame_limit,
                overwrite=args.overwrite,
                copy_rgb=args.copy_rgb,
                copy_masks=args.copy_masks,
                require_masks=args.require_masks,
                write_glb=args.write_glb,
                include_parts=args.include_parts,
                include_static_parts=args.include_static_parts,
                floor_size=args.floor_size,
                normalize_surfaces=args.normalize_surfaces,
                sequence_normalize_surfaces=args.sequence_normalize_surfaces,
            )
            for sequence_dir in tqdm(sequence_dirs, desc="Preprocessing synthetic physics sequences")
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    process_sequence,
                    sequence_dir,
                    output_root,
                    args.num_points,
                    args.sphere_subdivisions,
                    args.frame_limit,
                    args.overwrite,
                    args.copy_rgb,
                    args.copy_masks,
                    args.require_masks,
                    args.write_glb,
                    args.include_parts,
                    args.include_static_parts,
                    args.floor_size,
                    args.normalize_surfaces,
                    args.sequence_normalize_surfaces,
                )
                for sequence_dir in sequence_dirs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing synthetic physics sequences"):
                results.append(future.result())

    for sequence_name, entries in results:
        if entries:
            dataset_index[sequence_name] = entries

    json_output.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w") as f:
        json.dump(dict(sorted(dataset_index.items())), f, indent=2)

    total_frames = sum(len(entries) for entries in dataset_index.values())
    print(f"Wrote {json_output} with {len(dataset_index)} sequences and {total_frames} frames.")
    print(f"Processed data root: {output_root}")


if __name__ == "__main__":
    main()

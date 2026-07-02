#!/usr/bin/env python3
"""Probe whether TripoSG latents support multi-view canonical refinement.

This is deliberately independent of COM4D training. It freezes TripoSG, aligns one
tracked object's clean GT surfaces into object-local coordinates, and compares:
  * reference: one visible-frame latent, unchanged;
  * shared_optimized: one latent optimized using several visible frames;
  * memory_fitted: CanonicalObjectMemory's updater fitted on this sequence;
  * independent_oracle: separate latents optimized with each frame's clean GT.

The independent result is an oracle, including on occluded frames. It is not a fair
occlusion method; it measures the reconstruction ceiling and temporal variation.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from src.models.autoencoders import TripoSGVAEModel
from src.models.object_memory import CanonicalObjectMemory
from src.utils.inference_utils import field_to_mesh, hierarchical_extract_fields


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="dataset_json/physics_train_100_static.json")
    p.add_argument("--sequence", default="rolling_occluder_000002")
    p.add_argument("--object", default="ball_0")
    p.add_argument("--pretrained", default="pretrained_weights/TripoSG")
    p.add_argument("--output-dir", default="../outputs/latent_refinement/rolling_occluder_000002")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-tokens", type=int, default=512)
    p.add_argument("--visible-threshold", type=float, default=0.6)
    p.add_argument("--hidden-threshold", type=float, default=0.2)
    p.add_argument("--max-visible-frames", type=int, default=8)
    p.add_argument("--max-hidden-frames", type=int, default=8)
    p.add_argument("--loss-points", type=int, default=512)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--latent-prior-weight", type=float, default=1e-3)
    p.add_argument("--separation-weight", type=float, default=0.1)
    p.add_argument("--normal-offset", type=float, default=0.01)
    p.add_argument("--memory-heads", type=int, default=8)
    p.add_argument("--mesh-depth", type=int, default=7)
    p.add_argument("--mesh-margin", type=float, default=0.15)
    p.add_argument("--chamfer-points", type=int, default=10000)
    p.add_argument("--skip-meshes", action="store_true")
    return p.parse_args()


def quaternion_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    x, y, z, w = q
    return np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def evenly_spaced(indices, limit):
    if len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, limit).round().astype(int)
    return [indices[i] for i in positions]


def load_sequence(args):
    manifest = json.loads(Path(args.manifest).read_text())
    if args.sequence not in manifest:
        matches = [k for k in manifest if args.sequence in k]
        raise KeyError(f"sequence {args.sequence!r} not found; matches={matches[:10]}")
    frames = manifest[args.sequence]
    names = frames[0]["object_names"]
    if args.object not in names:
        raise KeyError(f"object {args.object!r} not in {names}")
    object_index = names.index(args.object)
    visibility = np.asarray([f["visibility"][object_index] for f in frames], dtype=np.float32)
    visible = evenly_spaced(
        np.flatnonzero(visibility >= args.visible_threshold).tolist(), args.max_visible_frames
    )
    hidden = evenly_spaced(
        np.flatnonzero(visibility < args.hidden_threshold).tolist(), args.max_hidden_frames
    )
    if not visible:
        raise RuntimeError("no visible frames satisfy --visible-threshold")
    if not hidden:
        raise RuntimeError("no hidden frames satisfy --hidden-threshold")
    selected = sorted(set(visible + hidden))
    surfaces = {}
    for frame_index in selected:
        frame = frames[frame_index]
        data = np.load(frame["surface_path"], allow_pickle=True).item()
        part = data["parts"][object_index]
        points = np.asarray(part["surface_points"], dtype=np.float32)
        normals = np.asarray(part["surface_normals"], dtype=np.float32)
        translation = np.asarray(frame["object_translation"][object_index], dtype=np.float32)
        rotation = quaternion_matrix(frame["object_quaternion_xyzw"][object_index])
        # Row-vector convention: world_to_local = (p - t) @ R.
        points = (points - translation[None]) @ rotation
        normals = normals @ rotation
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
        surfaces[frame_index] = np.concatenate((points, normals), axis=-1)
    return frames, object_index, visibility, visible, hidden, surfaces


def tensor_surface(surface, device, dtype):
    return torch.as_tensor(surface, device=device, dtype=dtype)


@torch.no_grad()
def encode_surface(vae, surface, num_tokens, seed):
    return vae.encode(surface[None], num_tokens=num_tokens, seed=seed).latent_dist.mode()


def sample_queries(surface, count, generator):
    count = min(count, surface.shape[0])
    idx = torch.randperm(surface.shape[0], device=surface.device, generator=generator)[:count]
    return surface[idx, :3], surface[idx, 3:6]


def decoder_surface_loss(
    vae, latent, surfaces, frame_indices, count, generator, offset, separation_weight
):
    losses = []
    surface_terms = []
    separation_terms = []
    for frame_index in frame_indices:
        points, normals = sample_queries(surfaces[frame_index], count, generator)
        query = torch.cat((points, points + offset*normals, points - offset*normals), dim=0)
        values = vae.decode(latent, sampled_points=query[None]).sample.float().reshape(-1)
        n = points.shape[0]
        surface_loss = values[:n].abs().mean()
        delta = values[n:2*n] - values[2*n:]
        # Orientation is irrelevant here; require a stable zero crossing around the surface.
        separation_loss = torch.relu(offset - delta.abs()).mean()
        losses.append(surface_loss + separation_weight * separation_loss)
        surface_terms.append(surface_loss.detach())
        separation_terms.append(separation_loss.detach())
    return (
        torch.stack(losses).mean(),
        torch.stack(surface_terms).mean(),
        torch.stack(separation_terms).mean(),
    )


def optimize_shared(args, vae, initial, surfaces, visible, generator):
    latent = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.Adam([latent], lr=args.lr)
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, surface, separation = decoder_surface_loss(
            vae, latent, surfaces, visible, args.loss_points, generator,
            args.normal_offset, args.separation_weight,
        )
        prior = (latent - initial).square().mean()
        total = loss + args.latent_prior_weight * prior
        total.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            history.append({"step": step + 1, "loss": float(total), "surface": float(surface),
                            "separation": float(separation), "prior": float(prior)})
    return latent.detach(), history


def optimize_memory(args, vae, initial, evidence, visibility, surfaces, visible, generator):
    memory = CanonicalObjectMemory(initial.shape[-1], args.memory_heads).to(
        device=initial.device, dtype=initial.dtype
    )
    # Read attention is irrelevant to this isolated updater experiment.
    parameters = [
        *memory.evidence_projection.parameters(), *memory.write_cell.parameters(),
        *memory.write_gate.parameters(),
    ]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    evidence_grid = evidence[None, :, None]
    visibility_grid = visibility[None, :, None]
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        state = memory.initialize(initial.detach().unsqueeze(1), confidence=float(visibility[0]))
        state, _ = memory.update(state, evidence_grid, visibility_grid)
        loss, surface, separation = decoder_surface_loss(
            vae, state.tokens[:, 0], surfaces, visible, args.loss_points, generator,
            args.normal_offset, args.separation_weight,
        )
        prior = (state.tokens[:, 0] - initial).square().mean()
        total = loss + args.latent_prior_weight * prior
        total.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            history.append({"step": step + 1, "loss": float(total), "surface": float(surface),
                            "separation": float(separation), "prior": float(prior)})
    with torch.no_grad():
        state = memory.initialize(initial.detach().unsqueeze(1), confidence=float(visibility[0]))
        state, _ = memory.update(state, evidence_grid, visibility_grid)
    return state.tokens[:, 0].detach(), history


def optimize_independent(args, vae, initial_by_frame, surfaces, indices, generator):
    output = {}
    histories = {}
    for frame_index in indices:
        latent = torch.nn.Parameter(initial_by_frame[frame_index].detach().clone())
        optimizer = torch.optim.Adam([latent], lr=args.lr)
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss, surface, separation = decoder_surface_loss(
                vae, latent, surfaces, [frame_index], args.loss_points, generator,
                args.normal_offset, args.separation_weight,
            )
            prior = (latent - initial_by_frame[frame_index]).square().mean()
            total = loss + args.latent_prior_weight * prior
            total.backward()
            optimizer.step()
        output[frame_index] = latent.detach()
        histories[frame_index] = {"loss": float(total), "surface": float(surface),
                                  "separation": float(separation), "prior": float(prior)}
    return output, histories


@torch.no_grad()
def evaluate_sdf(args, vae, latent, surfaces, indices, generator):
    rows = {}
    for frame_index in indices:
        evaluation_generator = torch.Generator(device=latent.device).manual_seed(
            args.seed + frame_index
        )
        loss, surface, separation = decoder_surface_loss(
            vae, latent, surfaces, [frame_index], args.loss_points, evaluation_generator,
            args.normal_offset, args.separation_weight,
        )
        rows[frame_index] = {
            "sdf_total": float(loss), "surface_abs": float(surface),
            "zero_crossing_hinge": float(separation),
        }
    return rows


@torch.no_grad()
def extract_mesh(args, vae, latent, bound):
    field = hierarchical_extract_fields(
        lambda x: vae.decode(latent, sampled_points=x).sample,
        device=latent.device, dtype=latent.dtype, bounds=bound,
        dense_octree_depth=args.mesh_depth,
        hierarchical_octree_depth=args.mesh_depth,
        max_num_expanded_coords=1e8,
    )
    return field_to_mesh(field, bound, args.mesh_depth, latent.device)


def point_chamfer(mesh, target_points, count, seed):
    if mesh is None or mesh.is_empty:
        return math.inf
    rng = np.random.default_rng(seed)
    predicted, _ = trimesh.sample.sample_surface(mesh, count, seed=rng)
    if len(target_points) > count:
        target_points = target_points[rng.choice(len(target_points), count, replace=False)]
    a = cKDTree(predicted)
    b = cKDTree(target_points)
    return float(np.mean(a.query(target_points)[0] ** 2) + np.mean(b.query(predicted)[0] ** 2))


def mean_rows(rows, indices, key):
    values = [rows[i][key] for i in indices]
    return float(np.mean(values)) if values else None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frames, object_index, visibility, visible, hidden, surfaces_np = load_sequence(args)
    surfaces = {i: tensor_surface(x, device, dtype) for i, x in surfaces_np.items()}
    all_indices = sorted(surfaces)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    vae = TripoSGVAEModel.from_pretrained(args.pretrained, subfolder="vae").to(device, dtype)
    vae.eval().requires_grad_(False)

    initial_by_frame = {}
    for frame_index in all_indices:
        initial_by_frame[frame_index] = encode_surface(
            vae, surfaces[frame_index], args.num_tokens, args.seed + frame_index
        )
    reference_index = visible[0]
    reference = initial_by_frame[reference_index]

    shared, shared_history = optimize_shared(
        args, vae, reference, surfaces, visible, generator
    )
    visible_evidence = torch.cat([initial_by_frame[i] for i in visible], dim=0)
    visible_weights = torch.as_tensor(
        visibility[visible], device=device, dtype=dtype
    )
    memory_fitted, memory_history = optimize_memory(
        args, vae, reference, visible_evidence, visible_weights,
        surfaces, visible, generator,
    )
    independent, independent_history = optimize_independent(
        args, vae, initial_by_frame, surfaces, all_indices, generator
    )

    methods = {
        "reference": reference,
        "shared_optimized": shared,
        "memory_fitted": memory_fitted,
    }
    sdf = {
        name: evaluate_sdf(args, vae, latent, surfaces, all_indices, generator)
        for name, latent in methods.items()
    }
    sdf["independent_oracle"] = {
        i: evaluate_sdf(args, vae, independent[i], surfaces, [i], generator)[i]
        for i in all_indices
    }

    metrics = {
        "sequence": args.sequence,
        "object": args.object,
        "reference_frame": reference_index,
        "visible_frames": visible,
        "hidden_frames": hidden,
        "notes": {
            "memory_fitted": "Updater weights fitted on this sequence; optimistic, not held-out generalization.",
            "independent_oracle": "Uses each frame's clean GT surface, including hidden frames.",
            "coordinates": "Object-local pose alignment; physical scale is preserved.",
        },
        "optimization": {
            "shared_optimized": shared_history,
            "memory_fitted": memory_history,
            "independent_oracle": independent_history,
        },
        "sdf": sdf,
        "summary": {},
    }

    for name, rows in sdf.items():
        metrics["summary"][name] = {
            "visible_surface_abs": mean_rows(rows, visible, "surface_abs"),
            "hidden_surface_abs": mean_rows(rows, hidden, "surface_abs"),
            "visible_sdf_total": mean_rows(rows, visible, "sdf_total"),
            "hidden_sdf_total": mean_rows(rows, hidden, "sdf_total"),
        }

    latent_stack = torch.cat([independent[i] for i in all_indices], dim=0).float()
    metrics["summary"]["independent_oracle"]["temporal_latent_variance"] = float(
        latent_stack.var(dim=0, unbiased=False).mean()
    )
    for name in methods:
        metrics["summary"][name]["temporal_latent_variance"] = 0.0

    if not args.skip_meshes:
        max_extent = max(float(np.abs(surfaces_np[i][:, :3]).max()) for i in all_indices)
        bound = max_extent + args.mesh_margin
        target_points = np.concatenate([surfaces_np[i][:, :3] for i in hidden], axis=0)
        meshes = {}
        for name, latent in methods.items():
            mesh = extract_mesh(args, vae, latent, bound)
            meshes[name] = mesh
            mesh.export(output / f"{name}.glb")
            metrics["summary"][name]["hidden_chamfer"] = point_chamfer(
                mesh, target_points, args.chamfer_points, args.seed
            )
        independent_chamfer = []
        independent_meshes = []
        for frame_index in all_indices:
            mesh = extract_mesh(args, vae, independent[frame_index], bound)
            mesh.export(output / f"independent_frame_{frame_index:04d}.glb")
            independent_meshes.append(mesh)
            if frame_index in hidden:
                independent_chamfer.append(point_chamfer(
                    mesh, surfaces_np[frame_index][:, :3], args.chamfer_points,
                    args.seed + frame_index,
                ))
        metrics["summary"]["independent_oracle"]["hidden_chamfer"] = float(
            np.mean(independent_chamfer)
        )
        pairwise = []
        for left, right in zip(independent_meshes[:-1], independent_meshes[1:]):
            if not left.is_empty and not right.is_empty:
                points, _ = trimesh.sample.sample_surface(right, args.chamfer_points)
                pairwise.append(point_chamfer(left, points, args.chamfer_points, args.seed))
        metrics["summary"]["independent_oracle"]["temporal_mesh_variance_chamfer"] = (
            float(np.mean(pairwise)) if pairwise else None
        )
        for name in methods:
            metrics["summary"][name]["temporal_mesh_variance_chamfer"] = 0.0

    torch.save(
        {
            "reference": reference.cpu(), "shared_optimized": shared.cpu(),
            "memory_fitted": memory_fitted.cpu(),
            "independent_oracle": {i: z.cpu() for i, z in independent.items()},
        },
        output / "latents.pt",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["summary"], indent=2))
    print(f"Wrote {output / 'metrics.json'}")


if __name__ == "__main__":
    main()

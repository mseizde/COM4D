"""Persistent canonical object memory for COM4D frame-major latents."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn


@dataclass
class ObjectPose:
    """Rigid object-to-world pose; rotation is quaternion (x, y, z, w)."""

    translation: torch.Tensor
    rotation: torch.Tensor

    def normalized(self, eps: float = 1e-8) -> "ObjectPose":
        if self.translation.shape[-1] != 3 or self.rotation.shape[-1] != 4:
            raise ValueError("pose shapes must be [..., 3] translation and [..., 4] rotation")
        rotation = self.rotation / self.rotation.norm(dim=-1, keepdim=True).clamp_min(eps)
        return ObjectPose(self.translation, rotation)

    def rotation_matrix(self) -> torch.Tensor:
        x, y, z, w = self.normalized().rotation.unbind(-1)
        values = (
            1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
            2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
            2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
        )
        return torch.stack(values, -1).reshape(self.rotation.shape[:-1] + (3, 3))

    def canonical_to_world(self, points: torch.Tensor) -> torch.Tensor:
        return points @ self.rotation_matrix().transpose(-1, -2) + self.translation.unsqueeze(-2)

    def world_to_canonical(self, points: torch.Tensor) -> torch.Tensor:
        return (points - self.translation.unsqueeze(-2)) @ self.rotation_matrix()


class ObjectPosePredictionHead(nn.Module):
    """Predict object-to-world translation and normalized XYZW quaternion."""

    def __init__(self, channels: int, hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 7),
        )

    def forward(self, tokens: torch.Tensor) -> ObjectPose:
        prediction = self.net(tokens.float().mean(dim=-2))
        translation = prediction[..., :3]
        quaternion = prediction[..., 3:]
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return ObjectPose(translation=translation, rotation=quaternion)


@dataclass
class ObjectMemoryState:
    tokens: torch.Tensor       # [B, O, M, D]
    confidence: torch.Tensor   # [B, O]
    def to(self, *args, **kwargs) -> "ObjectMemoryState":
        return ObjectMemoryState(self.tokens.to(*args, **kwargs), self.confidence.to(*args, **kwargs))

    def detached(self) -> "ObjectMemoryState":
        return ObjectMemoryState(self.tokens.detach(), self.confidence.detach())


def _counts(value: Union[int, torch.Tensor]) -> List[int]:
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]


def frame_major_tensor(values: torch.Tensor, num_frames: Union[int, torch.Tensor],
                       num_objects: Union[int, torch.Tensor], *, pad_value: float = 0.0
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack flattened frame-major values as padded [B,T,O,...]."""
    frame_counts = _counts(num_frames)
    object_counts = _counts(num_objects)
    if len(frame_counts) != len(object_counts):
        if len(object_counts) == 1:
            object_counts *= len(frame_counts)
        else:
            raise ValueError("num_frames and num_objects must have the same length")
    expected = sum(t * o for t, o in zip(frame_counts, object_counts))
    if values.shape[0] != expected:
        raise ValueError(f"counts describe {expected} entries, got {values.shape[0]}")
    shape = (len(frame_counts), max(frame_counts), max(object_counts)) + values.shape[1:]
    packed = values.new_full(shape, pad_value)
    valid = torch.zeros(shape[:3], dtype=torch.bool, device=values.device)
    offset = 0
    for group, (frames, objects) in enumerate(zip(frame_counts, object_counts)):
        count = frames * objects
        packed[group, :frames, :objects] = values[offset:offset + count].reshape(
            frames, objects, *values.shape[1:]
        )
        valid[group, :frames, :objects] = True
        offset += count
    return packed, valid


def visibility_from_masks(visible_masks: torch.Tensor, amodal_masks: Optional[torch.Tensor] = None,
                          eps: float = 1e-6) -> torch.Tensor:
    """Compute visible/amodal area ratio for masks shaped [...,H,W]."""
    visible = visible_masks.float().clamp(0, 1).sum(dim=(-2, -1))
    if amodal_masks is not None:
        full = amodal_masks.float().clamp(0, 1).sum(dim=(-2, -1))
    elif visible.ndim >= 3:
        full = visible.amax(dim=-2, keepdim=True).expand_as(visible)
    else:
        full = visible.amax().expand_as(visible)
    return torch.where(full > eps, (visible / full.clamp_min(eps)).clamp(0, 1), torch.zeros_like(full))


def canonical_tokens_from_context(
    frame_tokens: torch.Tensor,
    visibility: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    quality: Optional[torch.Tensor] = None,
    non_border: Optional[torch.Tensor] = None,
    min_visibility: float = 0.2,
    return_indices: bool = False,
):
    """Select the best reliable canonical observation independently per object.

    Eligible non-border frames are ranked by quality (visibility by default).
    If every eligible frame touches a border, selection falls back to the earliest
    eligible frame. If no frame is eligible, the maximum-visibility frame is
    returned with zero confidence.
    """
    if frame_tokens.ndim != 5 or visibility.shape != frame_tokens.shape[:3]:
        raise ValueError("expected tokens [B,T,O,M,C] and visibility [B,T,O]")
    for name, value in (("quality", quality), ("non_border", non_border)):
        if value is not None and value.shape != visibility.shape:
            raise ValueError(f"{name} must have shape [B,T,O]")

    eligible = visibility >= min_visibility
    if valid is not None:
        eligible &= valid
    if context is not None:
        eligible &= context

    preferred = eligible if non_border is None else eligible & non_border.bool()
    score = visibility if quality is None else quality
    masked_score = score.masked_fill(~preferred, torch.finfo(score.dtype).min)
    best = masked_score.argmax(dim=1)
    earliest = eligible.to(torch.int64).argmax(dim=1)
    fallback = visibility.argmax(dim=1)
    has_preferred = preferred.any(dim=1)
    has_eligible = eligible.any(dim=1)
    index = torch.where(has_preferred, best, torch.where(has_eligible, earliest, fallback))

    b, _, o, m, c = frame_tokens.shape
    gather = index[:, None, :, None, None].expand(b, 1, o, m, c)
    tokens = frame_tokens.gather(1, gather).squeeze(1)
    selected_visibility = visibility.gather(1, index[:, None, :]).squeeze(1)
    selected_quality = score.gather(1, index[:, None, :]).squeeze(1).clamp(0, 1)
    confidence = torch.where(
        has_eligible,
        torch.minimum(selected_visibility, selected_quality),
        torch.zeros_like(selected_visibility),
    )
    if return_indices:
        return tokens, confidence, index
    return tokens, confidence


def relative_object_pose(translation: torch.Tensor, rotation: torch.Tensor,
                         reference_indices: torch.Tensor) -> torch.Tensor:
    """Return reference-local translation and XYZW rotation as [B,T,O,7]."""
    if translation.ndim != 4 or translation.shape[-1] != 3:
        raise ValueError("translation must have shape [B,T,O,3]")
    if rotation.shape != translation.shape[:-1] + (4,):
        raise ValueError("rotation must have shape [B,T,O,4]")
    if reference_indices.shape != (translation.shape[0], translation.shape[2]):
        raise ValueError("reference_indices must have shape [B,O]")
    rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b, _, o, _ = translation.shape
    ref_t = translation.gather(1, reference_indices[:, None, :, None].expand(b, 1, o, 3)).squeeze(1)
    ref_q = rotation.gather(1, reference_indices[:, None, :, None].expand(b, 1, o, 4)).squeeze(1)
    # Keep time and object as explicit batch axes. Plain ``@`` interprets the
    # trailing [O, 3] translation dimensions as a matrix and cannot broadcast
    # [B, O, 3, 3] rotations across T when T != O.
    translation_delta = translation - ref_t[:, None]
    reference_rotation = ObjectPose(ref_t, ref_q).rotation_matrix()
    local_t = torch.einsum(
        "btoi,boij->btoj", translation_delta, reference_rotation
    )
    x1, y1, z1 = (-ref_q[..., :3]).unbind(-1)
    w1 = ref_q[..., 3]
    x1, y1, z1, w1 = (v[:, None] for v in (x1, y1, z1, w1))
    x2, y2, z2, w2 = rotation.unbind(-1)
    relative_q = torch.stack((
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ), dim=-1)
    relative_q = relative_q / relative_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cat((local_t, relative_q), dim=-1)


class CanonicalObjectMemory(nn.Module):
    """Object-local cross-attention reads and visibility-gated GRU writes.

    Frame state/evidence is [B,T,O,S,D]. Canonical memory is [B,O,M,D].
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if channels <= 0 or num_heads <= 0 or channels % num_heads:
            raise ValueError("channels must be positive and divisible by num_heads")
        self.channels = channels
        self.state_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.read_attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        # Behavior preserving when inserted into a pretrained transformer.
        self.read_scale = nn.Parameter(torch.zeros(()))
        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels), nn.SiLU()
        )
        self.pose_conditioner = nn.Sequential(
            nn.Linear(7, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        # Preserve old-checkpoint behavior until training learns pose conditioning.
        nn.init.zeros_(self.pose_conditioner[-1].weight)
        nn.init.zeros_(self.pose_conditioner[-1].bias)
        self.write_cell = nn.GRUCell(channels, channels)
        self.write_gate = nn.Sequential(nn.Linear(2 * channels, 1), nn.Sigmoid())
        # Do not rely on checkpoint loaders to initialize parameters that are
        # absent from an older checkpoint. Some loading paths materialize
        # missing tensors from empty storage instead of preserving the
        # constructor initialization.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every trainable component to a finite, safe state."""
        self.state_norm.reset_parameters()
        self.memory_norm.reset_parameters()
        self.read_attention._reset_parameters()
        self.read_attention.out_proj.reset_parameters()
        nn.init.zeros_(self.read_scale)

        for module in self.evidence_projection:
            reset = getattr(module, "reset_parameters", None)
            if reset is not None:
                reset()
        for module in self.pose_conditioner:
            reset = getattr(module, "reset_parameters", None)
            if reset is not None:
                reset()
        # Preserve old-checkpoint behavior until pose conditioning is learned.
        nn.init.zeros_(self.pose_conditioner[-1].weight)
        nn.init.zeros_(self.pose_conditioner[-1].bias)

        self.write_cell.reset_parameters()
        for module in self.write_gate:
            reset = getattr(module, "reset_parameters", None)
            if reset is not None:
                reset()

    @staticmethod
    def initialize(tokens: torch.Tensor, confidence: float = 0.0) -> ObjectMemoryState:
        if tokens.ndim != 4:
            raise ValueError("memory tokens must have shape [B,O,M,D]")
        return ObjectMemoryState(tokens, tokens.new_full(tokens.shape[:2], confidence))

    @staticmethod
    def _finite_description(name: str, tensor: torch.Tensor) -> str:
        value = tensor.detach().float()
        finite = torch.isfinite(value)
        finite_values = value[finite]
        max_abs = finite_values.abs().amax().item() if finite_values.numel() else float("nan")
        return (
            f"{name}[shape={tuple(value.shape)},"
            f"finite={int(finite.sum().item())}/{value.numel()},"
            f"max_abs_finite={max_abs:.6g}]"
        )


    def read(self, frame_state: torch.Tensor, memory: ObjectMemoryState,
             object_mask: Optional[torch.Tensor] = None,
             relative_pose: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._validate(frame_state, memory)
        b, t, o, s, d = frame_state.shape
        m = memory.tokens.shape[2]
        # Keep memory attention in FP32. FP16 logits can become NaN, and
        # multiplying that readout by a zero-initialized gate still yields NaN.
        device_type = frame_state.device.type
        autocast_context = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_context:
            normalized_state = self.state_norm(frame_state.float())
            normalized_state_finite = torch.isfinite(normalized_state).all()
            pose_finite = torch.tensor(True, device=frame_state.device)
            pose_embedding_finite = torch.tensor(True, device=frame_state.device)
            queries = normalized_state
            if relative_pose is not None:
                pose = self._pose(relative_pose, frame_state)
                pose_finite = torch.isfinite(pose).all()
                if not bool(pose_finite.item()):
                    raise FloatingPointError(
                        "Non-finite relative pose before memory conditioning: "
                        + self._finite_description("relative_pose", pose)
                    )
                pose_embedding = self.pose_conditioner(pose.float())
                pose_embedding_finite = torch.isfinite(pose_embedding).all()
                if not bool(pose_embedding_finite.item()):
                    parameter_text = "; ".join(
                        self._finite_description(f"pose_conditioner.{name}", parameter)
                        for name, parameter in self.pose_conditioner.named_parameters()
                    )
                    raise FloatingPointError(
                        "Non-finite object-memory pose embedding: "
                        + "; ".join((
                            self._finite_description("relative_pose", pose),
                            self._finite_description("pose_embedding", pose_embedding),
                            parameter_text,
                        ))
                    )
                queries = queries + pose_embedding[..., None, :]
            queries = queries.reshape(-1, s, d)
            keys = (
                self.memory_norm(memory.tokens.float()).unsqueeze(1)
                .expand(b, t, o, m, d).reshape(-1, m, d)
            )
            query_finite = torch.isfinite(queries).all()
            key_finite = torch.isfinite(keys).all()
            token_finite = torch.isfinite(memory.tokens).all()
            if not bool((query_finite & key_finite & token_finite).item()):
                self.last_read_diagnostics = {
                    "normalized_state_all_finite": normalized_state_finite.detach(),
                    "relative_pose_all_finite": pose_finite.detach(),
                    "pose_embedding_all_finite": pose_embedding_finite.detach(),
                    "query_all_finite": query_finite.detach(),
                    "key_all_finite": key_finite.detach(),
                    "memory_tokens_all_finite": token_finite.detach(),
                }
                raise FloatingPointError(
                    "Non-finite object-memory attention input: "
                    + "; ".join((
                        self._finite_description("queries", queries),
                        self._finite_description("keys", keys),
                        self._finite_description("memory_tokens", memory.tokens),
                    ))
                )
            math_sdpa_context = (
                torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_mem_efficient=False,
                    enable_math=True,
                )
                if device_type == "cuda"
                else nullcontext()
            )
            with math_sdpa_context:
                readout = self.read_attention(
                    queries, keys, keys, need_weights=False
                )[0]
            readout = readout.reshape(b, t, o, s, d)
        raw_finite = torch.isfinite(readout)
        parameter_finite = torch.stack([
            torch.isfinite(parameter).all()
            for parameter in self.read_attention.parameters()
        ]).all()
        if not bool((raw_finite.all() & parameter_finite).item()):
            self.last_read_diagnostics = {
                "query_all_finite": query_finite.detach(),
                "key_all_finite": key_finite.detach(),
                "memory_tokens_all_finite": token_finite.detach(),
                "mha_parameters_all_finite": parameter_finite.detach(),
                "raw_all_finite": raw_finite.all().detach(),
                "raw_nonfinite_count": (~raw_finite).sum().detach(),
            }
            parameter_text = "; ".join(
                self._finite_description(f"mha.{name}", parameter)
                for name, parameter in self.read_attention.named_parameters()
            )
            raise FloatingPointError(
                "Non-finite object-memory attention output under FP32 math SDPA: "
                + "; ".join((
                    self._finite_description("queries", queries),
                    self._finite_description("keys", keys),
                    self._finite_description("readout", readout),
                    parameter_text,
                ))
            )
        if object_mask is not None:
            readout = readout * self._scalar(
                object_mask, readout, "object_mask"
            )[..., None, None]
        read_scale = torch.tanh(self.read_scale.float())
        scaled_readout = read_scale * readout
        if frame_state.dtype in {torch.float16, torch.bfloat16}:
            dtype_limit = torch.finfo(frame_state.dtype).max
            scaled_readout = scaled_readout.clamp(-dtype_limit, dtype_limit)
        self.last_read_diagnostics = {
            "normalized_state_all_finite": normalized_state_finite.detach(),
            "relative_pose_all_finite": pose_finite.detach(),
            "pose_embedding_all_finite": pose_embedding_finite.detach(),
            "query_all_finite": query_finite.detach(),
            "key_all_finite": key_finite.detach(),
            "memory_tokens_all_finite": token_finite.detach(),
            "mha_parameters_all_finite": parameter_finite.detach(),
            "raw_all_finite": raw_finite.all().detach(),
            "raw_nonfinite_count": (~raw_finite).sum().detach(),
            "raw_max_abs_finite": torch.where(
                raw_finite, readout.abs(), torch.zeros_like(readout)
            ).amax().detach(),
            "scale": read_scale.detach(),
            "scaled_all_finite": torch.isfinite(scaled_readout).all().detach(),
            "scaled_max_abs": scaled_readout.abs().amax().detach(),
        }
        return frame_state + scaled_readout.to(frame_state.dtype)

    def update(self, memory: ObjectMemoryState, evidence: torch.Tensor,
               visibility: torch.Tensor,
               confidence: Optional[torch.Tensor] = None,
               relative_pose: Optional[torch.Tensor] = None) -> Tuple[ObjectMemoryState, torch.Tensor]:
        self._validate(evidence, memory)
        # Inference keeps this module in FP32 while VAE evidence remains FP16.
        working_dtype = next(self.parameters()).dtype
        evidence_work = evidence.to(dtype=working_dtype)
        visible = self._scalar(visibility, evidence_work, "visibility").clamp(0, 1)
        reliable = torch.ones_like(visible) if confidence is None else self._scalar(
            confidence, evidence_work, "confidence").clamp(0, 1)
        reliability = visible * reliable
        projected = self.evidence_projection(evidence_work.mean(-2))
        if relative_pose is not None:
            pose = self._pose(relative_pose, evidence_work)
            projected = projected + self.pose_conditioner(pose.float()).to(projected.dtype)
        pooled = ((projected * reliability[..., None]).sum(1)
                  / reliability.sum(1).clamp_min(1e-8)[..., None])
        old = memory.tokens.to(dtype=working_dtype)
        expanded = pooled.unsqueeze(-2).expand_as(old)
        candidate = self.write_cell(expanded.reshape(-1, self.channels),
                                    old.reshape(-1, self.channels)).reshape_as(old)
        gate = self.write_gate(torch.cat((old, expanded), -1))
        gate = gate * reliability.amax(1)[..., None, None]
        tokens = old + gate * (candidate - old)
        max_reliability = reliability.amax(1)
        old_confidence = memory.confidence.to(dtype=max_reliability.dtype)
        accumulated = 1 - (1 - old_confidence.clamp(0, 1)) * (1 - max_reliability)
        accumulated = torch.where(max_reliability > 0, accumulated, old_confidence)
        return ObjectMemoryState(tokens, accumulated), gate

    def forward(self, frame_state: torch.Tensor, memory: ObjectMemoryState, *,
                evidence: Optional[torch.Tensor] = None,
                visibility: Optional[torch.Tensor] = None,
                confidence: Optional[torch.Tensor] = None,
                update_memory: bool = False) -> Tuple[torch.Tensor, ObjectMemoryState]:
        state = self.read(frame_state, memory)
        if not update_memory:
            return state, memory
        if evidence is None or visibility is None:
            raise ValueError("evidence and visibility are required for memory updates")
        updated, _ = self.update(memory, evidence, visibility, confidence)
        return state, updated

    def _validate(self, frame: torch.Tensor, memory: ObjectMemoryState) -> None:
        if frame.ndim != 5 or memory.tokens.ndim != 4:
            raise ValueError("expected frame [B,T,O,S,D] and memory [B,O,M,D]")
        if memory.confidence.shape != memory.tokens.shape[:2]:
            raise ValueError("memory confidence must have shape [B,O]")
        if frame.shape[0] != memory.tokens.shape[0] or frame.shape[2] != memory.tokens.shape[1]:
            raise ValueError("frame and memory batch/object axes differ")
        if frame.shape[-1] != self.channels or memory.tokens.shape[-1] != self.channels:
            raise ValueError(f"all token channels must equal {self.channels}")

    @staticmethod
    def _scalar(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
        if value.shape == reference.shape[:3] + (1,):
            value = value.squeeze(-1)
        if value.shape != reference.shape[:3]:
            raise ValueError(f"{name} must have shape [B,T,O] or [B,T,O,1]")
        return value.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _pose(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.shape != reference.shape[:3] + (7,):
            raise ValueError("relative_pose must have shape [B,T,O,7]")
        return value.to(device=reference.device, dtype=reference.dtype)

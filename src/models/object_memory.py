"""Persistent canonical object memory for COM4D frame-major latents."""

from dataclasses import dataclass
from typing import Optional, Tuple

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


@dataclass
class ObjectMemoryState:
    tokens: torch.Tensor       # [B, O, M, D]
    confidence: torch.Tensor   # [B, O]

    def detached(self) -> "ObjectMemoryState":
        return ObjectMemoryState(self.tokens.detach(), self.confidence.detach())


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
        self.write_cell = nn.GRUCell(channels, channels)
        self.write_gate = nn.Sequential(nn.Linear(2 * channels, 1), nn.Sigmoid())

    @staticmethod
    def initialize(tokens: torch.Tensor, confidence: float = 0.0) -> ObjectMemoryState:
        if tokens.ndim != 4:
            raise ValueError("memory tokens must have shape [B,O,M,D]")
        return ObjectMemoryState(tokens, tokens.new_full(tokens.shape[:2], confidence))

    def read(self, frame_state: torch.Tensor, memory: ObjectMemoryState,
             object_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._validate(frame_state, memory)
        b, t, o, s, d = frame_state.shape
        m = memory.tokens.shape[2]
        queries = self.state_norm(frame_state).reshape(-1, s, d)
        keys = (self.memory_norm(memory.tokens).unsqueeze(1)
                .expand(b, t, o, m, d).reshape(-1, m, d))
        readout = self.read_attention(queries, keys, keys, need_weights=False)[0]
        readout = readout.reshape_as(frame_state)
        if object_mask is not None:
            readout = readout * self._scalar(object_mask, frame_state, "object_mask")[..., None, None]
        return frame_state + torch.tanh(self.read_scale) * readout

    def update(self, memory: ObjectMemoryState, evidence: torch.Tensor,
               visibility: torch.Tensor,
               confidence: Optional[torch.Tensor] = None) -> Tuple[ObjectMemoryState, torch.Tensor]:
        self._validate(evidence, memory)
        visible = self._scalar(visibility, evidence, "visibility").clamp(0, 1)
        reliable = torch.ones_like(visible) if confidence is None else self._scalar(
            confidence, evidence, "confidence").clamp(0, 1)
        reliability = visible * reliable
        projected = self.evidence_projection(evidence.mean(-2))
        pooled = ((projected * reliability[..., None]).sum(1)
                  / reliability.sum(1).clamp_min(1e-8)[..., None])
        old = memory.tokens
        expanded = pooled.unsqueeze(-2).expand_as(old)
        candidate = self.write_cell(expanded.reshape(-1, self.channels),
                                    old.reshape(-1, self.channels)).reshape_as(old)
        gate = self.write_gate(torch.cat((old, expanded), -1))
        gate = gate * reliability.amax(1)[..., None, None]
        tokens = old + gate * (candidate - old)
        max_reliability = reliability.amax(1)
        accumulated = 1 - (1 - memory.confidence.clamp(0, 1)) * (1 - max_reliability)
        accumulated = torch.where(max_reliability > 0, accumulated, memory.confidence)
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

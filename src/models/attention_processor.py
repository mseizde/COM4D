from typing import Callable, List, Optional, Tuple, Union

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_torch_version, maybe_allow_in_graph
from einops import rearrange
from torch import nn

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _sp_trace(event: str, tensor: Optional[torch.Tensor] = None, group=None, **fields) -> None:
    if os.environ.get("COM4D_SP_TRACE", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(os.environ.get("RANK", "-1"))
        group_rank = _get_group_rank(group) if dist.is_available() and dist.is_initialized() else -1
        local_rank = os.environ.get("LOCAL_RANK", "?")
        cuda_current = torch.cuda.current_device() if torch.cuda.is_available() and torch.cuda.is_initialized() else "uninit"
        parts = [
            f"ts={time.time():.6f}",
            f"pid={os.getpid()}",
            f"rank={rank}",
            f"group_rank={group_rank}",
            f"local_rank={local_rank}",
            f"cuda_current={cuda_current}",
            f"event={event}",
        ]
        if tensor is not None:
            parts.extend([
                f"shape={tuple(tensor.shape)}",
                f"device={tensor.device}",
                f"dtype={tensor.dtype}",
            ])
        parts.extend(f"{key}={value}" for key, value in fields.items())
        trace_file = os.environ.get("COM4D_SP_TRACE_FILE") or f"/tmp/com4d_sp_trace_rank{rank}.log"
        with open(trace_file, "a", encoding="utf-8") as handle:
            handle.write(" ".join(parts) + "\n")
    except Exception as exc:
        print(
            f"[COM4D_SP_TRACE] failed to write event={event!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def trace_sequence_parallel_event(event: str, tensor: Optional[torch.Tensor] = None, group=None, **fields) -> None:
    _sp_trace(event, tensor=tensor, group=group, **fields)


def _distributed_is_ready(group=None) -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group=group) > 1


def _get_group_rank(group=None) -> int:
    if group is None:
        return dist.get_rank()
    return dist.get_group_rank(group, dist.get_rank())


def _pad_sequence_dim(tensor: torch.Tensor, multiple: int, dim: int = 2) -> Tuple[torch.Tensor, int]:
    size = tensor.shape[dim]
    pad_len = (multiple - size % multiple) % multiple
    if pad_len == 0:
        return tensor, 0

    pad = [0] * (2 * tensor.dim())
    pad[2 * (tensor.dim() - dim - 1) + 1] = pad_len
    return F.pad(tensor, tuple(pad)), pad_len


def _depad_sequence_dim(tensor: torch.Tensor, pad_len: int, dim: int = 2) -> torch.Tensor:
    if pad_len == 0:
        return tensor.contiguous()
    keep = tensor.shape[dim] - pad_len
    return tensor.narrow(dim, 0, keep).contiguous()


class _HeadsSequenceAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(group=process_group)
        _sp_trace("a2a.forward.enter", input_tensor, process_group, scatter_dim=scatter_dim, gather_dim=gather_dim, world_size=world_size)
        chunks = [chunk.contiguous() for chunk in torch.chunk(input_tensor, world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(chunks[0]) for _ in range(world_size)]
        _sp_trace("a2a.forward.before", input_tensor, process_group, chunk_shape=tuple(chunks[0].shape))
        dist.all_to_all(outputs, chunks, group=process_group)
        _sp_trace("a2a.forward.after", outputs[0], process_group)
        return torch.cat(outputs, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _HeadsSequenceAllToAll.apply(
            grad_output,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return grad_input, None, None, None


def _all_to_all_heads_sequence(
    tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group=None,
) -> torch.Tensor:
    return _HeadsSequenceAllToAll.apply(tensor, group, scatter_dim, gather_dim)




class _GatherSequenceShards(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, process_group, gather_dim):
        ctx.process_group = process_group
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(group=process_group)
        _sp_trace("gather.forward.enter", input_tensor, process_group, gather_dim=gather_dim, world_size=world_size)
        outputs = [torch.empty_like(input_tensor) for _ in range(world_size)]
        _sp_trace("gather.forward.before", input_tensor, process_group)
        dist.all_gather(outputs, input_tensor.contiguous(), group=process_group)
        _sp_trace("gather.forward.after", outputs[0], process_group)
        return torch.cat(outputs, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        rank = _get_group_rank(ctx.process_group)
        world_size = dist.get_world_size(group=ctx.process_group)
        chunks = torch.chunk(grad_output, world_size, dim=ctx.gather_dim)
        return chunks[rank].contiguous(), None, None


def _gather_sequence_shards(tensor: torch.Tensor, group=None, dim: int = 2) -> torch.Tensor:
    return _GatherSequenceShards.apply(tensor, group, dim)


def _validate_replicated_tensor(tensor: torch.Tensor, *, group=None, name: str, atol: float = 1e-2) -> None:
    if not _distributed_is_ready(group):
        return
    with torch.no_grad():
        check = tensor.detach().float()
        stats = torch.stack([
            check.sum(),
            check.square().sum(),
            check.mean(),
        ])
        gathered = [torch.empty_like(stats) for _ in range(dist.get_world_size(group=group))]
        _sp_trace("validate.before", stats, group, name=name)
        dist.all_gather(gathered, stats, group=group)
        _sp_trace("validate.after", gathered[0], group, name=name)
        ref = gathered[0]
        for idx, other in enumerate(gathered[1:], start=1):
            if not torch.allclose(ref, other, atol=atol, rtol=1e-4):
                raise ValueError(
                    f"sequence_parallel_replicated_batch=True requires identical {name} on all SP ranks; "
                    f"rank 0 stats={ref.tolist()}, rank {idx} stats={other.tolist()}. "
                    "This usually means normal DDP is feeding different samples to each rank."
                )


def _sequence_parallel_replicated_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    group=None,
    attention_mask: Optional[torch.Tensor] = None,
    validate_replicated: bool = True,
) -> torch.Tensor:
    """Shard a replicated full [B, H, L, D] sequence across SP ranks for attention only."""
    if query.is_cuda:
        torch.cuda.set_device(query.device)
    _sp_trace("replicated.enter", query, group, validate=validate_replicated)
    if not _distributed_is_ready(group):
        return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    if attention_mask is not None:
        raise ValueError("sequence-parallel replicated grouped self-attention does not support attention_mask yet.")

    world_size = dist.get_world_size(group=group)
    rank = _get_group_rank(group)
    if query.shape[1] % world_size != 0:
        raise ValueError(
            f"num attention heads ({query.shape[1]}) must be divisible by sequence-parallel size ({world_size})."
        )
    original_seq_len = query.shape[2]
    pad_len = (world_size - original_seq_len % world_size) % world_size
    if validate_replicated:
        _validate_replicated_tensor(query, group=group, name="query")
        _validate_replicated_tensor(key, group=group, name="key")
        _validate_replicated_tensor(value, group=group, name="value")
    _sp_trace("replicated.after_validate", query, group, original_seq_len=original_seq_len, pad_len=pad_len)

    if pad_len > 0:
        query = _pad_sequence_dim(query, world_size, dim=2)[0]
        key = _pad_sequence_dim(key, world_size, dim=2)[0]
        value = _pad_sequence_dim(value, world_size, dim=2)[0]
        padded_seq_len = query.shape[2]
        pad_mask = torch.zeros(
            1, 1, padded_seq_len, padded_seq_len,
            dtype=torch.float32,
            device=query.device,
        )
        pad_mask[..., original_seq_len:] = torch.finfo(torch.float32).min
        attention_mask = pad_mask

    shard_len = query.shape[2] // world_size
    start = rank * shard_len
    query_shard = query.narrow(2, start, shard_len).contiguous()
    key_shard = key.narrow(2, start, shard_len).contiguous()
    value_shard = value.narrow(2, start, shard_len).contiguous()

    _sp_trace("replicated.before_inner", query_shard, group, shard_len=shard_len, start=start)
    local_output = _sequence_parallel_sdpa(
        query_shard,
        key_shard,
        value_shard,
        group=group,
        attention_mask=attention_mask,
    )
    _sp_trace("replicated.after_inner", local_output, group)
    output = _gather_sequence_shards(local_output, group=group, dim=2)
    return _depad_sequence_dim(output, pad_len, dim=2)


def _sequence_parallel_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    group=None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Hunyuan/HY-World style sequence-parallel SDPA for already sequence-sharded
    [B, H, L_local, D] tensors. Each SP rank must hold the same batch items and
    a different sequence shard before this function is called.
    """
    if query.is_cuda:
        torch.cuda.set_device(query.device)
    _sp_trace("sp_sdpa.enter", query, group)
    if not _distributed_is_ready(group):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

    world_size = dist.get_world_size(group=group)
    if query.shape[1] % world_size != 0:
        raise ValueError(
            f"num attention heads ({query.shape[1]}) must be divisible by sequence-parallel size ({world_size})."
        )

    local_length = torch.tensor([query.shape[2]], device=query.device, dtype=torch.int64)
    gathered_lengths = [torch.empty_like(local_length) for _ in range(world_size)]
    _sp_trace("length.before", local_length, group)
    dist.all_gather(gathered_lengths, local_length, group=group)
    _sp_trace("length.after", gathered_lengths[0], group)
    local_lengths = [int(length.item()) for length in gathered_lengths]
    if len(set(local_lengths)) != 1:
        raise ValueError(
            "sequence-parallel grouped self-attention currently requires equal sequence shard lengths "
            f"on all ranks, got {local_lengths}."
        )

    qkv = torch.cat([query, key, value], dim=0).contiguous()

    # [3B, H, L_local, D] -> [3B, H/P, L_global, D]
    _sp_trace("sp_sdpa.before_first_a2a", qkv, group)
    qkv = _all_to_all_heads_sequence(qkv, scatter_dim=1, gather_dim=2, group=group)
    _sp_trace("sp_sdpa.after_first_a2a", qkv, group)
    query_sp, key_sp, value_sp = qkv.chunk(3, dim=0)

    _sp_trace("sp_sdpa.before_sdpa", query_sp, group)
    hidden_states = F.scaled_dot_product_attention(
        query_sp,
        key_sp,
        value_sp,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )

    _sp_trace("sp_sdpa.after_sdpa", hidden_states, group)
    # [B, H/P, L_global, D] -> [B, H, L_local, D]
    hidden_states = _all_to_all_heads_sequence(hidden_states, scatter_dim=2, gather_dim=1, group=group)
    _sp_trace("sp_sdpa.after_second_a2a", hidden_states, group)
    return hidden_states.contiguous()


class FlashTripo2AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Tripo2DiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self, topk=True):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.topk = topk

    def qkv(self, attn, q, k, v, attn_mask, dropout_p, is_causal):
        if k.shape[-2] == 3072:
            topk = 1024
        elif k.shape[-2] == 512:
            topk = 256
        else:
            topk = k.shape[-2] // 3

        if self.topk is True:
            q1 = q[:, :, ::100, :]
            sim = q1 @ k.transpose(-1, -2)
            sim = torch.mean(sim, -2)
            topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
            topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
            v0 = torch.gather(v, dim=-2, index=topk_ind)
            k0 = torch.gather(k, dim=-2, index=topk_ind)
            out = F.scaled_dot_product_attention(q, k0, v0)
        elif self.topk is False:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            idx, counts = self.topk
            start = 0
            outs = []
            for grid_coord, count in zip(idx, counts):
                end = start + count
                q_chunk = q[:, :, start:end, :]
                q1 = q_chunk[:, :, ::50, :]
                sim = q1 @ k.transpose(-1, -2)
                sim = torch.mean(sim, -2)
                topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
                topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
                v0 = torch.gather(v, dim=-2, index=topk_ind)
                k0 = torch.gather(k, dim=-2, index=topk_ind)
                out = F.scaled_dot_product_attention(q_chunk, k0, v0)
                outs.append(out)
                start += count
            out = torch.cat(outs, dim=-2)
        self.topk = False
        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that tripo2 split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
        # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # flashvdm topk
        hidden_states = self.qkv(attn, query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)   

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class TripoSGAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the TripoSG model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that pre-trained models split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
        # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class FusedTripoSGAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0) with fused
    projection layers. This is used in the HunyuanDiT model. It applies a s normalization layer and rotary embedding on
    query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FusedTripoSGAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # NOTE that pre-trained split heads first, then split qkv
        if encoder_hidden_states is None:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            query = attn.to_q(hidden_states)

            kv = attn.to_kv(encoder_hidden_states)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# Modified from https://github.com/VAST-AI-Research/MIDI-3D/blob/main/midi/models/attention_processor.py#L264
class PartCrafterAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the PartCrafter model. It applies a normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        num_parts: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that pre-trained models split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
        # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        if isinstance(num_parts, torch.Tensor):
            # Assume list in training, do not consider classifier-free guidance
            idx = 0
            hidden_states_list = []
            for n_p in num_parts:
                k = key[idx : idx + n_p]
                v = value[idx : idx + n_p]
                q = query[idx : idx + n_p]
                idx += n_p
                if k.shape[2] == q.shape[2]:
                    # Assuming self-attention
                    # Here 'b' is always 1
                    k = rearrange(
                        k, "(b ni) h nt c -> b h (ni nt) c", ni=n_p
                    ) # [b, h, ni*nt, c]
                    v = rearrange(
                        v, "(b ni) h nt c -> b h (ni nt) c", ni=n_p
                    ) # [b, h, ni*nt, c]
                else:
                    # Assuming cross-attention
                    # Here 'b' is always 1
                    k = k[::n_p]     # [b, h, nt, c]
                    v = v[::n_p]     # [b, h, nt, c]
                # Here 'b' is always 1
                q = rearrange(
                    q, "(b ni) h nt c -> b h (ni nt) c", ni=n_p
                ) # [b, h, ni*nt, c]
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                h_s = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    is_causal=False,
                )
                h_s = h_s.transpose(1, 2).reshape(
                    n_p, -1, attn.heads * head_dim
                )
                h_s = h_s.to(query.dtype)
                hidden_states_list.append(h_s)
            hidden_states = torch.cat(hidden_states_list, dim=0)

        elif isinstance(num_parts, int):
            # Assume single instance
            if key.shape[2] == query.shape[2]:
                # Assuming self-attention
                # Here we need 'b' when using classifier-free guidance
                key = rearrange(
                    key, "(b ni) h nt c -> b h (ni nt) c", ni=num_parts
                ) # [b, h, ni*nt, c]
                value = rearrange(
                    value, "(b ni) h nt c -> b h (ni nt) c", ni=num_parts
                ) # [b, h, ni*nt, c]
            else:
                # Assuming cross-attention
                # Here we need 'b' when using classifier-free guidance
                # Control signal is repeated ni times within each (b, ni)
                # We select only the first instance per group
                key = key[::num_parts]     # [b, h, nt, c]
                value = value[::num_parts] # [b, h, nt, c]
            query = rearrange(
                query, "(b ni) h nt c -> b h (ni nt) c", ni=num_parts
            ) # [b, h, ni*nt, c]

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

        else:
            raise ValueError(
                "num_parts must be a torch.Tensor or int, but got {}".format(type(num_parts))
            )
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# Combined processor: supports both num_parts (3D spatial) and num_frames (4D temporal)
class PartFrameCrafterAttnProcessor(nn.Module):
    """
    Multi-instance attention processor that supports either spatial grouping by parts or temporal grouping by frames.
    Accepts either `num_parts` or `num_frames` in kwargs (tensor per-object or int). If both are provided, `num_frames`
    takes precedence.

    The optional sequence-parallel backend is only for already sequence-sharded grouped self-attention. It must not be
    enabled for ordinary DDP batches, where each rank owns different samples rather than a different shard of the same
    sequence.
    """

    def __init__(self, sequence_parallel_attention: bool = False, sequence_parallel_group=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.sequence_parallel_attention = sequence_parallel_attention
        self.sequence_parallel_group = sequence_parallel_group

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        num_parts: Optional[Union[int, torch.Tensor]] = None,
        num_frames: Optional[Union[int, torch.Tensor]] = None,
        sequence_parallel_attention: Optional[bool] = None,
        sequence_parallel_group=None,
        sequence_parallel_sharded: bool = False,
        sequence_parallel_replicated_batch: bool = False,
        sequence_parallel_validate_replicated: bool = True,
        relation_grid_shapes: Optional[List[Tuple[int, int]]] = None,
        relation_bias_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        use_sequence_parallel = (
            self.sequence_parallel_attention
            if sequence_parallel_attention is None
            else sequence_parallel_attention
        )
        sequence_parallel_group = (
            self.sequence_parallel_group
            if sequence_parallel_group is None
            else sequence_parallel_group
        )
        if use_sequence_parallel and not (sequence_parallel_sharded or sequence_parallel_replicated_batch):
            raise ValueError(
                "sequence_parallel_attention=True requires either sequence_parallel_sharded=True or "
                "sequence_parallel_replicated_batch=True. Normal DDP batches contain different samples per rank; "
                "using SP attention on them would mix unrelated examples across GPUs."
            )

        # Choose grouping cardinality. If caller explicitly passes `num_frames=1`, fall back to
        # part-based grouping so static passes can still leverage part embeddings without mixing frames.
        def _is_one(value: Optional[Union[int, torch.Tensor]]) -> bool:
            if value is None:
                return False
            if isinstance(value, int):
                return value == 1
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return False
                ones = torch.ones_like(value)
                if value.dtype.is_floating_point:
                    return bool(torch.all(torch.isclose(value, ones)).item())
                return bool(torch.all(value == ones).item())
            return False

        if _is_one(num_frames):
            num_frames = None

        # Choose grouping cardinality prioritising frames when meaningful, otherwise parts.
        # print("num_frames:", num_frames, "num_parts:", num_parts)
        num_instances = num_frames if num_frames is not None else num_parts

        def _relation_mask(object_index: int, instance_count: int, token_count: int):
            if relation_bias_values is None:
                return None
            if relation_grid_shapes is None or object_index >= len(relation_grid_shapes):
                raise ValueError("relation_grid_shapes must describe every grouped object")
            frames, parts = relation_grid_shapes[object_index]
            if frames * parts != instance_count:
                raise ValueError(
                    f"relation grid {frames}x{parts} does not match {instance_count} instances"
                )
            if (
                relation_bias_values.ndim != 2
                or relation_bias_values.shape[0] != attn.heads
                or relation_bias_values.shape[1] != 3
            ):
                raise ValueError("relation_bias_values must have shape [heads, 3]")
            frame_ids = torch.arange(frames, device=query.device).repeat_interleave(parts)
            part_ids = torch.arange(parts, device=query.device).repeat(frames)
            relation = torch.full(
                (instance_count, instance_count), 2, device=query.device, dtype=torch.long
            )
            relation[part_ids[:, None] == part_ids[None, :]] = 1
            relation[frame_ids[:, None] == frame_ids[None, :]] = 0
            relation = relation.repeat_interleave(token_count, 0).repeat_interleave(token_count, 1)
            values = relation_bias_values.to(device=query.device, dtype=query.dtype)
            return values[:, relation].unsqueeze(0)

        if num_instances is None:
            if use_sequence_parallel:
                raise ValueError("sequence_parallel_attention requires num_parts or num_frames grouping metadata.")
            # Fallback to plain attention without grouping
            proc = TripoSGAttnProcessor2_0()
            return proc(attn, hidden_states, encoder_hidden_states, attention_mask, temb, image_rotary_emb)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # split heads-first
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # Grouping across instances (parts or frames)
        if isinstance(num_instances, torch.Tensor):
            idx = 0
            hidden_states_list = []
            for object_index, n_i in enumerate(num_instances):
                n_i = int(n_i.item())
                k = key[idx : idx + n_i]
                v = value[idx : idx + n_i]
                q = query[idx : idx + n_i]
                idx += n_i
                if k.shape[2] == q.shape[2]:
                    # self-attn: concat tokens across instances
                    k = rearrange(k, "(b ni) h nt c -> b h (ni nt) c", ni=n_i)
                    v = rearrange(v, "(b ni) h nt c -> b h (ni nt) c", ni=n_i)
                    q = rearrange(q, "(b ni) h nt c -> b h (ni nt) c", ni=n_i)
                    if use_sequence_parallel and sequence_parallel_replicated_batch:
                        h_s = _sequence_parallel_replicated_sdpa(
                            q,
                            k,
                            v,
                            group=sequence_parallel_group,
                            attention_mask=attention_mask,
                            validate_replicated=sequence_parallel_validate_replicated,
                        )
                    elif use_sequence_parallel:
                        h_s = _sequence_parallel_sdpa(
                            q,
                            k,
                            v,
                            group=sequence_parallel_group,
                            attention_mask=attention_mask,
                        )
                    else:
                        relation_mask = _relation_mask(object_index, n_i, q.shape[2] // n_i)
                        h_s = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=relation_mask, dropout_p=0.0, is_causal=False
                        )
                    h_s = h_s.transpose(1, 2).reshape(n_i, -1, attn.heads * head_dim)
                else:
                    # #### HERE CROSS_ATTN BUG START
                    # cross-attn: keep per-instance encoder states aligned with each
                    # instance's queries instead of collapsing to the first encoder.
                    # k = k[::n_i]
                    # v = v[::n_i]
                    h_s = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
                    h_s = h_s.transpose(1, 2).reshape(n_i, -1, attn.heads * head_dim)
                    # #### HERE CROSS_ATTN BUG END
                h_s = h_s.to(query.dtype)
                hidden_states_list.append(h_s)
            hidden_states = torch.cat(hidden_states_list, dim=0)
        elif isinstance(num_instances, int):
            if key.shape[2] == query.shape[2]:
                key = rearrange(key, "(b ni) h nt c -> b h (ni nt) c", ni=num_instances)
                value = rearrange(value, "(b ni) h nt c -> b h (ni nt) c", ni=num_instances)
                query = rearrange(query, "(b ni) h nt c -> b h (ni nt) c", ni=num_instances)
                if use_sequence_parallel and sequence_parallel_replicated_batch:
                    hidden_states = _sequence_parallel_replicated_sdpa(
                        query,
                        key,
                        value,
                        group=sequence_parallel_group,
                        attention_mask=attention_mask,
                        validate_replicated=sequence_parallel_validate_replicated,
                    )
                elif use_sequence_parallel:
                    hidden_states = _sequence_parallel_sdpa(
                        query,
                        key,
                        value,
                        group=sequence_parallel_group,
                        attention_mask=attention_mask,
                    )
                else:
                    relation_mask = _relation_mask(0, num_instances, query.shape[2] // num_instances)
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=relation_mask, dropout_p=0.0, is_causal=False
                    )
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
            else:
                # #### HERE CROSS_ATTN BUG START
                # Keep one encoder-state bank per instance instead of reusing only
                # the first encoder-state bank in the grouped batch.
                # key = key[::num_instances]
                # value = value[::num_instances]
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
                # #### HERE CROSS_ATTN BUG END
            hidden_states = hidden_states.to(query.dtype)
        else:
            raise ValueError(
                "num_parts/num_frames must be a torch.Tensor or int, but got {}".format(type(num_instances))
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

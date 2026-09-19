"""CUDA memory-efficient attention for unpadded, full-context text replay.

Uses the installed PyTorch CUTLASS operator (tested with torch 2.11/CUDA 12.8),
not a Triton JIT kernel. Neither scores nor causal/window masks are materialized.
The private ATen API is deliberately isolated here and covered by GPU tests.
"""
from __future__ import annotations

import torch

REPLAY_ATTENTION = "moe_replay_efficient_v1"


def replay_mask(batch_size, q_length, kv_length, q_offset=0, kv_offset=0,
                attention_mask=None, **kwargs):
    if batch_size != 1 or q_length != kv_length or q_offset != 0 or kv_offset != 0:
        raise ValueError("Replay attention requires one full sequence without a KV cache")
    if attention_mask is not None and (
        attention_mask.ndim != 2 or not bool(attention_mask.all())
    ):
        raise ValueError("Replay attention does not support padding or custom masks")
    return None


def replay_attention(module, query, key, value, attention_mask, scaling=None,
                     dropout=0.0, sliding_window=None, s_aux=None, **kwargs):
    if module.training or dropout:
        raise ValueError("Replay attention is inference-only; call model.eval()")
    if query.device.type != "cuda":
        raise ValueError("Replay attention requires CUDA")
    if query.shape[0] != 1 or query.shape[-2] != key.shape[-2]:
        raise ValueError("Replay attention requires one full sequence without a KV cache")
    if attention_mask is not None:
        raise ValueError("Replay attention expects its registered mask implementation")
    groups = query.shape[1] // key.shape[1]
    # CUTLASS expects equal Q/K/V head counts. Only K/V are expanded, never scores.
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    output, lse, *_ = torch.ops.aten._efficient_attention_forward.default(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        None, None, None, None, None, 0.0, 1, s_aux is not None,
        scale=scaling, window_size=sliding_window,
    )
    if s_aux is not None:
        # The sink contributes exp(sink) to the denominator, but no value vector.
        # Keep logsumexp and renormalization in FP32, including for BF16 replay.
        lse = lse[:, :, :query.shape[-2]].float()
        factor = torch.sigmoid(lse - s_aux.float().view(1, -1, 1))
        output = (output.float() * factor.transpose(1, 2).unsqueeze(-1)).to(query.dtype)
    return output.contiguous(), None


def register_replay_attention():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    AttentionInterface.register(REPLAY_ATTENTION, replay_attention)
    AttentionMaskInterface.register(REPLAY_ATTENTION, replay_mask)
    return REPLAY_ATTENTION

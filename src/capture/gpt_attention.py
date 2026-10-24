"""Bound the query rows of native GPT-OSS eager attention, preserving sinks and masks.

This changes no installed files, parameters, routing, keys, or context length.
The native attention implementation performs every chunk's arithmetic.
"""
import torch


def query_chunked_attention(native, chunk_size=128):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Positive chunk size required')

    def forward(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        if module.training or dropout or torch.is_grad_enabled():
            raise RuntimeError('Bounded attention is inference-only')
        if kwargs.get('output_attentions'):
            raise RuntimeError('Full attention matrix capture is unsupported')
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError('Expected batched attention tensors')
        q = query.shape[-2]
        if attention_mask is not None and (attention_mask.ndim != 4 or attention_mask.shape[-2] not in (1, q)):
            raise ValueError('Unsupported attention mask layout')
        pieces = []
        for start in range(0, q, chunk_size):
            end = min(q, start + chunk_size)
            mask = attention_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., start:end, :]
            result, weights = native(module, query[..., start:end, :], key, value, mask, scaling,
                                     dropout=dropout, **kwargs)
            pieces.append(result)
            del weights
        return torch.cat(pieces, dim=1), None
    return forward


def install_gpt_query_chunking(model, chunk_size=128):
    from transformers.models.gpt_oss import modeling_gpt_oss as implementation
    if model.config.model_type != 'gpt_oss' or model.config._attn_implementation != 'eager':
        raise ValueError('Only verified GPT-OSS eager attention is supported')
    native = implementation.eager_attention_forward
    if getattr(native, '_measurement_chunked', False):
        raise RuntimeError('Attention wrapper already installed')
    wrapper = query_chunked_attention(native, chunk_size)
    wrapper._measurement_chunked = True
    wrapper._measurement_native = native
    implementation.eager_attention_forward = wrapper
    return dict(attention_execution='native_eager_query_chunked', query_chunk_size=chunk_size,
                context='all original keys and values; native full/sliding causal masks and attention sinks preserved')

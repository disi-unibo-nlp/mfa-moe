"""gpt-oss replay in chunks of the saved tokens with a KV cache, so very long traces fit in memory (HF backend)."""
from __future__ import annotations


def chunk_ranges(total, prompt_length, chunk_size):
    if not 0 < prompt_length < total or chunk_size < 1:
        raise ValueError('Nonempty prompt/completion and positive chunk size required')
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        yield start, end, max(0, min(end - start, prompt_length - start))


def capture_chunked(model, tokenizer, trace, layers, chunk_size=512):
    import torch
    from models.router_adapters import stream_router_signals, model_family
    from models.token_replay import validate_token_replay
    if model_family(model) != 'gpt_oss':
        raise ValueError('KV-chunked adapter is validated only for GPT-OSS')
    prompt, completion = validate_token_replay(trace.metadata['token_replay'], tokenizer, trace.cot_text)
    tokens = prompt + completion
    device = next(model.parameters()).device
    decoder = getattr(model, 'model', model)
    state = {'cache': None}
    arrays = {}
    covered = torch.zeros(len(completion), dtype=torch.bool)
    semantics = None

    def forward(**kwargs):
        output = decoder(**kwargs)
        cache = getattr(output, 'past_key_values', None)
        if cache is None:
            raise ValueError('Model did not return an explicit KV cache')
        state['cache'] = cache
        return output

    for start, end, local_prompt in chunk_ranges(len(tokens), len(prompt), chunk_size):
        details = {}
        kwargs = dict(input_ids=torch.tensor([tokens[start:end]], device=device),
                      attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
                      position_ids=torch.arange(start, end, device=device).unsqueeze(0),
                      cache_position=torch.arange(start, end, device=device),
                      past_key_values=state['cache'], use_cache=True, return_dict=True)
        router, hidden = stream_router_signals(model, forward, kwargs, local_prompt,
                                               set(layers), True, details)
        begin, stop = max(start, len(prompt)) - len(prompt), end - len(prompt)
        if stop <= 0:
            continue
        values = dict(router=router, hidden=hidden,
                      ids=details['selected_experts'], weights=details['expert_weights'])
        if not details.get('semantics', {}).get('native_selection'):
            raise ValueError('Native selected IDs/weights are required from the HF adapter')
        if semantics is not None and semantics != details['semantics']:
            raise ValueError('Routing semantics changed between chunks')
        semantics = details['semantics']
        if covered[begin:stop].any():
            raise ValueError('Duplicate chunk coverage')
        for name, value in values.items():
            if value.shape[:2] != (len(layers), stop - begin):
                raise ValueError('Chunk layer/token coverage mismatch: ' + name)
            if name not in arrays:
                arrays[name] = torch.empty((len(layers), len(completion), value.shape[-1]), dtype=value.dtype)
            arrays[name][:, begin:stop].copy_(value)
        covered[begin:stop] = True
        del values, router, hidden, details
    state['cache'] = None
    if not covered.all() or not arrays:
        raise ValueError('Missing exact completion coverage')
    return dict(**arrays, selection=arrays['router'], semantics=semantics,
                audit={'native_dispatch': True, 'backend': 'hf_kv_chunked',
                       'layers': layers, 'chunk_size': chunk_size, 'tokens': len(completion)})

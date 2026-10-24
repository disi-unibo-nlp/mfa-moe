"""vLLM worker hooks that record the routers of an exact-token replay.

The engine loads this class as `worker_extension_cls` (see `capture.backends.VLLMBackend`). On each of the two
tensor-parallel workers it wraps the expert selection and the expert dispatch of every routed layer and the decoder
inputs, so that the selected experts, their weights, the router signal and the hidden states of the replayed
completion tokens are written into `CaptureBuffers`. Process-local hooks only: no installed package is edited and
nothing is intervened on.
"""
import hashlib
import inspect
import os
import re
import resource

import torch

from capture.buffers import (
    MAX_TRANSIENT_BYTES, MAX_WORKER_BYTES, CaptureBuffers, approved_capture_path, verified_modular_backend,
)
from capture.native_selection import boundary_gaps, require_native_boundary, selection_scores


def digest_tensor(x):
    return hashlib.sha256(x.contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def frame_entries(requests, keys, starts, tokens, positions):
    """For each token of the forward batch: (request, completion position), or None for prompt tokens.

    Every token the engine runs must equal the saved token id at that position."""
    if len(starts) != len(keys) + 1 or starts[0] != 0 or starts[-1] > len(tokens):
        raise ValueError('Invalid padding boundaries')
    frame = [None] * len(tokens)
    for i, key in enumerate(keys):
        if key not in requests:
            raise ValueError('Unregistered request')
        if starts[i + 1] < starts[i]:
            raise ValueError('Non-monotonic query starts')
        spec = requests[key]['spec']
        for j in range(starts[i], starts[i + 1]):
            pos = int(positions[j])
            if not 0 <= pos < len(spec['tokens']) or tokens[j] != spec['tokens'][pos]:
                raise ValueError('Exact replay token mismatch')
            if pos >= spec['prompt_length']:
                frame[j] = (key, pos - spec['prompt_length'])
    return frame


def store_frame(owner, layer, values):
    """Write a forward chunk at once; positions are validated against saved IDs."""
    grouped = {}
    for row, entry in enumerate(owner['frame']):
        if entry is not None:
            key, pos = entry
            indices, positions = grouped.setdefault(key, ([], []))
            indices.append(row)
            positions.append(pos)
    for key, (indices, positions) in grouped.items():
        selected = {name: value[indices] for name, value in values.items()}
        if 'selection' in selected:
            require_native_boundary(selected['selection'], selected['ids'])
        owner['requests'][key]['buffers'].write(layer, positions, selected)


class NativeHook:
    """Wraps one routed layer: the expert selection and the dispatch to the expert kernel."""

    def __init__(self, owner, layer, block):
        self.owner, self.layer, self.block = owner, layer, block
        self.runner = block.experts
        runner = self.runner
        if (type(runner).__name__ != 'MoERunner' or runner.routed_experts.quant_method.is_monolithic
                or getattr(runner, '_fse_fuse_gate', False) or getattr(block, 'is_sequence_parallel', False)
                or getattr(runner.router, 'eplb_state', None) is not None):
            raise RuntimeError('Unsupported native dispatch path')
        custom = type(runner.router).__name__ == 'CustomRoutingRouter'
        function = getattr(runner.router, 'custom_routing_function', None)
        if custom and not (type(block).__name__ == 'Gemma4MoE' and hasattr(block, 'per_expert_scale')
                           and getattr(function, '__module__', None) == 'vllm.model_executor.models.gemma4'
                           and getattr(function, '__qualname__', None) == 'Gemma4MoE.__init__.<locals>.routing_function'):
            raise RuntimeError('Unverified custom native selector')
        if not custom and type(runner.router).__name__ not in ('FusedTopKRouter', 'GroupedTopKRouter'):
            raise RuntimeError('Unverified native selector')
        if type(runner.router).__name__ == 'GroupedTopKRouter' and (
                runner.router.num_expert_group != 1 or runner.router.topk_group != 1):
            raise RuntimeError('Group-restricted native boundary capture requires a separate adapter')
        self.backend = verified_modular_backend(runner.routed_experts.quant_method)
        self.select_original = runner.router.select_experts
        self.dispatch_original = runner.routed_experts.forward_modular
        self.pending = None
        self.dispatches = 0
        runner.router.select_experts = self.select
        runner.routed_experts.forward_modular = self.dispatch

    def select(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        if self.pending is not None or self.owner['frame'] is None:
            raise RuntimeError('Unpaired native selection')
        weights, ids = self.select_original(hidden_states, router_logits, topk_indices_dtype=topk_indices_dtype,
                                            input_ids=input_ids)
        raw = router_logits.detach().float().cpu()
        sigmoid = getattr(self.runner.router, 'scoring_func', 'softmax') == 'sigmoid'
        signal = torch.nn.functional.logsigmoid(raw) if sigmoid else raw
        bias = getattr(self.runner.router, 'e_score_correction_bias', None)
        grouped = type(self.runner.router).__name__ == 'GroupedTopKRouter'
        fused = False
        if grouped:
            from vllm import envs
            fused = bool(envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK and router_logits.is_cuda and bias is not None)
        selection = selection_scores(router_logits.detach(), bias.detach() if bias is not None else None,
                                     scoring='sigmoid' if sigmoid else 'softmax', fused_grouped=fused).float().cpu()
        plain = raw.sigmoid() if sigmoid else raw.clone()
        if bias is not None:
            plain = plain + bias.detach().float().cpu()
        native_ids = ids.detach().cpu()
        plain_gaps = boundary_gaps(plain, native_ids)
        kernel_gaps = boundary_gaps(selection, native_ids)
        # Two reconstructions of the selection scores are compared with the dispatched experts: the plain one
        # (float sigmoid, then the bias) and the one that follows the selector's own arithmetic. One example per
        # layer and kind is kept; the stored kinds are 'old_mismatch' (the plain scores disagree with the dispatch),
        # 'corrected_mismatch' (the selector-arithmetic scores disagree) and 'sample' (neither).
        for i, entry in enumerate(self.owner['frame']):
            if entry is None:
                continue
            key, pos = entry
            request = self.owner['requests'][key]
            diagnostics = request.setdefault('selector_diagnostics', {})
            kind = ('old_mismatch' if float(plain_gaps[i]) < -1e-6
                    else 'corrected_mismatch' if float(kernel_gaps[i]) < -1e-6 else 'sample')
            label = f'{self.layer}/{kind}'
            if label not in diagnostics:
                other = selection[i].clone()
                other[native_ids[i].long()] = float('-inf')
                competitor = int(other.argmax())
                diagnostics[label] = dict(
                    layer=self.layer, completion_token=pos, absolute_token=pos + request['spec']['prompt_length'],
                    native_ids=native_ids[i].tolist(), native_weights=weights[i].detach().float().cpu().tolist(),
                    score_dtype=str(router_logits.dtype), bias_dtype=str(bias.dtype) if bias is not None else None,
                    raw_logits=raw[i].tolist(),
                    bias=bias.detach().float().cpu().tolist() if bias is not None else None,
                    competing_expert=competitor, old_boundary_discrepancy=float(plain_gaps[i]),
                    corrected_boundary_discrepancy=float(kernel_gaps[i]), fused_grouped=fused,
                    selection_arithmetic=('device_float_tanh_sigmoid_cast_then_bias' if fused and sigmoid
                                          else 'native_dtype_activation_then_bias'))
        self.pending = (ids, weights, signal, selection)
        return weights, ids

    def dispatch(self, *args, **kwargs):
        if args or self.pending is None:
            raise RuntimeError('Unverified dispatch signature')
        ids, weights, signal, selection = self.pending
        if kwargs.get('topk_ids') is not ids or kwargs.get('topk_weights') is not weights:
            raise RuntimeError('Captured selection did not reach expert kernel')
        ids = ids.detach().cpu()
        weights = weights.detach().float().cpu()
        if len(ids) != len(self.owner['frame']):
            raise RuntimeError('Dispatch rows disagree with request frame')
        store_frame(self.owner, self.layer, {'ids': ids, 'weights': weights, 'router': signal, 'selection': selection})
        self.dispatches += 1
        self.pending = None
        return self.dispatch_original(**kwargs)


class MeasurementWorkerExtension:
    def measurement_configure(self, spec):
        from vllm.distributed import get_tensor_model_parallel_rank
        cfg = self.vllm_config
        parallel = cfg.parallel_config
        modelcfg = cfg.model_config
        if (modelcfg.model not in (spec['generation_model'], spec['checkpoint']) or modelcfg.revision != spec['revision']
                or not modelcfg.enforce_eager or parallel.tensor_parallel_size != 2
                or parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel or parallel.enable_eplb
                or parallel.use_sequence_parallel_moe or cfg.speculative_config is not None
                or cfg.cache_config.enable_prefix_caching or getattr(cfg.scheduler_config, 'async_scheduling', False)
                or cfg.use_v2_model_runner):
            raise RuntimeError('Unverified model, checkpoint or execution configuration')
        if hasattr(self, 'measurement'):
            raise RuntimeError('Configure each worker only once')
        owner = {'requests': {}, 'frame': None, 'rank': get_tensor_model_parallel_rank(), 'spec': spec}
        self.measurement = owner
        model = self.model_runner.get_model()
        found = {}
        decoders = {}
        for name, module in model.named_modules():
            m = re.search(r'(?:^|\.)layers\.(\d+)$', name)
            if m:
                decoders[int(m[1])] = module
            m = re.search(r'(?:^|\.)layers\.(\d+)\.(mlp|mixer|moe)$', name)
            if m and hasattr(module, 'experts'):
                found[int(m[1])] = module
        if sorted(found) != spec['layers']:
            raise RuntimeError('Missing or extra native routed layers')
        owner['hooks'] = {layer: NativeHook(owner, layer, block) for layer, block in found.items()}

        def hidden_hook(layer):
            signature = inspect.signature(decoders[layer].forward)

            def capture(module, args, kwargs):
                bound = signature.bind_partial(*args, **kwargs).arguments
                h = bound['hidden_states']
                residual = bound.get('residual')
                # vLLM holds residual and branch output separately until the fused RMSNorm.
                # Gemma overwrites its residual argument with the complete input.
                gemma = type(decoders[layer]).__name__ == 'Gemma4DecoderLayer'
                hidden = h if residual is None or gemma else (h.float() + residual.float()).to(h.dtype)
                if owner['frame'] is None or len(hidden) != len(owner['frame']):
                    raise RuntimeError('Hidden rows lack verified request frame')
                if owner['rank'] == 0:
                    hidden = hidden.detach().cpu()
                    store_frame(owner, layer, {'hidden': hidden})
            return capture

        for layer in found:
            decoders[layer].register_forward_pre_hook(hidden_hook(layer), with_kwargs=True)
        original = self.model_runner._model_forward

        def forward(*args, **kwargs):
            if args or owner['frame'] is not None:
                raise RuntimeError('Unsupported model forward')
            batch = self.model_runner.input_batch
            ids = kwargs['input_ids']
            axes = kwargs['positions'].detach().cpu()
            if axes.ndim == 1:
                axes = axes[None, :]
            starts = self.model_runner.query_start_loc.np[:batch.num_reqs + 1].tolist()
            positions = [0] * len(ids)
            for i in range(batch.num_reqs):
                first = int(batch.num_computed_tokens_cpu[i])
                expected = list(range(first, first + starts[i + 1] - starts[i]))
                if axes.shape[-1] != len(ids) or not torch.equal(
                        axes[:, starts[i]:starts[i + 1]], torch.tensor(expected).expand(axes.shape[0], -1)):
                    raise RuntimeError('Unexpected absolute token positions')
                positions[starts[i]:starts[i + 1]] = expected
            owner['frame'] = frame_entries(owner['requests'], batch.req_ids, starts, ids.detach().cpu().tolist(),
                                           positions)
            before = {layer: hook.dispatches for layer, hook in owner['hooks'].items()}
            try:
                result = original(**kwargs)
                # Every routed layer must have gone through exactly one capture.
                if any(hook.pending is not None or hook.dispatches != before[layer] + 1
                       for layer, hook in owner['hooks'].items()):
                    raise RuntimeError('Backend bypassed dispatch capture')
                return result
            finally:
                owner['frame'] = None
        self.model_runner._model_forward = forward
        hooks = owner['hooks']
        return {'rank': owner['rank'], 'layers': sorted(found),
                'routers': {str(l): type(h.runner.router).__name__ for l, h in hooks.items()},
                'quant_methods': {str(l): type(h.runner.routed_experts.quant_method).__name__ for l, h in hooks.items()},
                'modular_backends': {str(l): h.backend for l, h in hooks.items()},
                'routed_output_scaling': {str(l): float(h.runner.routed_scaling_factor) for l, h in hooks.items()},
                'storage_weights_dtype': 'float16', 'rank_hashes_before_storage_rounding': True}

    def measurement_register(self, key, tokens, prompt_length, max_buffer_bytes=MAX_WORKER_BYTES):
        owner = self.measurement
        if owner['requests'] or not 0 < prompt_length < len(tokens):
            raise RuntimeError('Replay requires one bounded active request per worker')
        owner['requests'][key] = {
            'spec': {'tokens': tokens, 'prompt_length': prompt_length},
            'buffers': CaptureBuffers(owner['spec']['layers'], len(tokens) - prompt_length,
                                      rank=owner['rank'], max_bytes=max_buffer_bytes)}

    def measurement_snapshot(self, key, output, max_bytes=15 * 1024 ** 3):
        """Hash this rank's buffers; rank 0 also writes the tensors to `output`. Returns the hashes and sizes."""
        owner = self.measurement
        request = owner['requests'][key]
        buffers = request['buffers']
        arrays = buffers.require_complete()
        hashes = {}
        for index, layer in enumerate(buffers.layers):
            for name, array in arrays.items():
                if name != 'hidden':
                    hashes[f'{layer}/{name}'] = digest_tensor(array[index])
        tensor_bytes = sum(x.numel() * x.element_size() for x in arrays.values())
        if owner['rank'] == 0:
            path = approved_capture_path(output)
            if not 0 < max_bytes <= MAX_TRANSIENT_BYTES:
                raise ValueError('Invalid transient cap')
            weights = arrays['weights']
            stored_bytes = tensor_bytes - weights.numel() * weights.element_size() + weights.numel() * 2
            if stored_bytes + 1024 ** 2 > max_bytes:
                raise ValueError('Transient tensor storage cap exceeded before write')
            data = buffers.for_storage()
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + '.partial')
            torch.save(data, partial)
            if partial.stat().st_size > max_bytes:
                raise ValueError('Transient serialization cap exceeded')
            os.replace(partial, path)
        return {'rank': owner['rank'], 'tokens': buffers.tokens, 'layers': buffers.layers, 'hashes': hashes,
                'allocated_buffer_bytes': buffers.allocated_bytes, 'native_tensor_bytes': tensor_bytes,
                'worker_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
                'storage_weights_dtype': 'float16', 'rank_hashes_before_storage_rounding': True,
                'dispatch_calls': {str(l): h.dispatches for l, h in owner['hooks'].items()},
                'selector_diagnostics': request.get('selector_diagnostics', {})}

    def measurement_release(self, key):
        del self.measurement['requests'][key]

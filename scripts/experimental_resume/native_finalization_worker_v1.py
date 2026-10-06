"""Read-only native router capture on each TP rank, installed by explicit RPC.

Returns the original router tensors unchanged. Buffers retain executed IDs and
weights for prediction rows; no selection is inferred from affinities.
"""
from pathlib import Path
from types import MethodType


class NativeFinalizationWorker:
    def finalization_install(self):
        import torch
        from moe_steer.vllm_ext import find_moe_runners, validate_engine, validate_layers
        runner = self.model_runner
        if hasattr(self, '_nf'):
            return self._nf['status']
        engine_checks = validate_engine(runner)
        found = find_moe_runners(runner.model)
        layer_checks = validate_layers(found)
        from vllm.distributed import get_tensor_model_parallel_rank
        state = self._nf = {'buffers': {}, 'batch': None, 'rank': get_tensor_model_parallel_rank(),
                           'mapping': [], 'enabled': False}
        original_forward = runner._model_forward

        def forward(instance, *args, **kwargs):
            state['mapping'] = []
            if state['enabled']:
                batch = instance.input_batch; starts = batch.num_computed_tokens_cpu
                qsl = instance.query_start_loc.np
                for i, rid in enumerate(batch.req_ids[:batch.num_reqs]):
                    req = instance.requests[rid]
                    meta = (req.sampling_params.extra_args or {}).get('native_finalization')
                    if meta is None:
                        raise ValueError('capture request lacks assignment metadata')
                    uid = meta['uid']; prompt = len(req.prompt_token_ids)
                    a, b, start = int(qsl[i]), int(qsl[i + 1]), int(starts[i])
                    buf = state['buffers'][uid]
                    if prompt != buf['prompt_tokens']:
                        raise ValueError('worker prompt length differs')
                    # Absolute input position p predicts output p - prompt + 1.
                    lo = max(0, prompt - 1 - start)
                    pred = start + lo - prompt + 1
                    count = b - a - lo
                    buf['prefill_rows'] += min(b - a, max(0, prompt - 1 - start))
                    if count > 0:
                        if pred < 0 or pred + count > buf['cap']:
                            raise ValueError('prediction row outside assignment')
                        state['mapping'].append((uid, a + lo, b, pred))
            return original_forward(*args, **kwargs)

        runner._model_forward = MethodType(forward, runner)
        for layer, (_, experts) in found.items():
            router = experts.router
            original = router.select_experts

            def capture(*args, _original=original, _layer=layer, **kwargs):
                weights, ids = _original(*args, **kwargs)
                if state['enabled']:
                    for uid, a, b, pred in state['mapping']:
                        buf = state['buffers'][uid]; end = pred + b - a
                        buf['ids'][pred:end, _layer].copy_(ids[a:b])
                        buf['weights'][pred:end, _layer].copy_(weights[a:b])
                        buf['recomputed_layer_rows'] += int(buf['seen'][pred:end, _layer].sum())
                        buf['seen'][pred:end, _layer] = True
                return weights, ids

            router.select_experts = capture
        state['status'] = {'rank': state['rank'], 'layers': sorted(found),
            'configured_experts_per_token': 8, 'selection_semantics': 'native softmax top-k, renormalized; returned order preserved',
            'shared_expert_handling': 'separate shared branch; excluded from routed IDs/weights',
            'engine_checks': engine_checks, 'layer_checks': layer_checks,
            'prediction_row': 'absolute_input_position - prompt_tokens + 1',
            'weights': 'direct select_experts return, passed unchanged to expert execution'}
        return state['status']

    def finalization_begin(self, assignments, cap):
        import numpy as np
        import torch
        state = self._nf
        if state['enabled'] or state['buffers']:
            raise ValueError('previous capture batch is not flushed')
        for a in assignments:
            shape = (cap, 40, 8)
            state['buffers'][a['uid']] = {'prompt_tokens': a['prompt_tokens'], 'cap': cap,
                'ids': torch.empty(shape, dtype=torch.int16, device=self.device),
                'weights': torch.empty(shape, dtype=torch.float32, device=self.device),
                'seen': np.zeros((cap, 40), dtype=bool), 'prefill_rows': 0, 'recomputed_layer_rows': 0}
        state['enabled'] = True
        return {'rank': state['rank'], 'assignments': len(assignments)}

    def finalization_flush(self, uid, emitted_tokens, directory):
        import numpy as np
        from utility_scout_v1 import digest, file_sha
        state = self._nf; buf = state['buffers'].pop(uid)
        seen = buf['seen'][:emitted_tokens]
        if not seen.all():
            raise ValueError('native captured prediction coverage is incomplete')
        root = Path(directory); root.mkdir(parents=True, exist_ok=True)
        path = root / (digest(uid) + '-rank' + str(state['rank']) + '.npz')
        weights = buf['weights'][:emitted_tokens].cpu().numpy()
        ids = buf['ids'][:emitted_tokens].cpu().numpy()
        if not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(-1), 1., atol=1e-5):
            raise ValueError('native executed weights invalid')
        with path.open('xb') as stream:
            np.savez_compressed(stream, native_ids=ids, executed_weights=weights)
        if not state['buffers']:
            state['enabled'] = False
        return {'rank': state['rank'], 'path': str(path), 'file_sha256': file_sha(path),
            'shape': list(ids.shape), 'prefill_rows_excluded': buf['prefill_rows'],
            'recomputed_layer_rows': buf['recomputed_layer_rows'],
            'selection_status': 'CAPTURED_NATIVE', 'executed_weights_status': 'CAPTURED',
            'prediction_coverage': 'COMPLETE', 'mask': 'all native; no intervention installed',
            'shared_expert_weights': None, 'shared_expert_weights_status': 'NOT_CAPTURED_SEPARATE_BRANCH'}

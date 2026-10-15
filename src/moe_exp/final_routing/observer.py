"""Restorable request-aware hooks for the pinned synchronous vLLM runtime."""
from __future__ import annotations
import inspect
import re
from .common import digest, require
from .operator import hard_select, policy

ROUTER = re.compile(r'(?:^|\.)layers\.(\d+)\.(?:mlp|moe|mixer|block_sparse_moe|feed_forward)\.experts$')


def discover(model):
    found = {}
    for name, module in model.named_modules():
        match = ROUTER.search(name)
        router = getattr(module, 'router', None)
        if match and router is not None:
            layer = int(match[1])
            require(layer not in found, 'duplicate physical routed layer')
            quant = getattr(getattr(module, 'routed_experts', None), 'quant_method', None)
            require(not getattr(quant, 'is_monolithic', False), 'kernel bypasses router callback')
            found[layer] = (name, router, model.get_submodule(name.rsplit('.', 1)[0]))
    return found


class RequestCapture:
    def __init__(self, runner, contract, requests):
        from moe_exp.verification_study.adapters import PredictionClock
        self.runner, self.contract = runner, contract
        self.model = runner.get_model()
        self.requests = {r['uid']: r for r in requests}
        require(len(self.requests) == len(requests), 'duplicate request ID')
        self.clocks = {uid: PredictionClock(r['prompt'], r['prefix'], r['closure']) for uid, r in self.requests.items()}
        self.records = {uid: {} for uid in self.requests}
        self.counts = {uid: {'rows': 0, 'active': 0, 'changed': 0, 'noop_exact': True,
                             'target_contract': True, 'finite_weights': True} for uid in self.requests}
        self.sampled_states = {}
        self.native_validation = {}
        self.boundaries, self.req_ids, self.contexts = [], [], []
        self.originals = []
        self.handle = None
        self.original_prepare = runner._prepare_inputs
        routers = discover(self.model)
        require(sorted(routers) == contract['layers'], 'physical routed-layer map differs')
        wanted = set()
        for r in requests:
            wanted.update(r.get('capture_layers', []))
            if r.get('action'):
                wanted.add(r['action']['layer'])
        require(wanted and wanted <= routers.keys(), 'requested router is absent')
        self.geometry = []
        try:
            signature = inspect.signature(self.original_prepare)
            def prepare(*args, **kwargs):
                result = self.original_prepare(*args, **kwargs)
                self.req_ids = list(runner.input_batch.req_ids)
                n = len(self.req_ids)
                require(set(self.req_ids) <= self.requests.keys(), 'neighboring/unknown engine request')
                self.boundaries = runner.query_start_loc.np[:n+1].tolist()
                require(len(self.boundaries) == n+1 and self.boundaries[0] == 0 and
                        all(a < b for a, b in zip(self.boundaries, self.boundaries[1:])), 'request boundaries differ')
                return result
            runner._prepare_inputs = prepare
            for layer, (name, router, parent) in sorted(routers.items()):
                require(router.top_k == contract['native_k'] and router.global_num_experts == contract['experts'],
                        'native top-k/experts differ')
                native_policy = policy(router, parent)
                self.geometry.append({'layer': layer, 'module': name, 'router': type(router).__name__,
                                      'scoring': native_policy['scoring'], 'renormalize': native_policy['renormalize']})
                if layer not in wanted:
                    continue
                original = router._compute_routing
                signature = inspect.signature(original)
                self.originals.append((router, original))
                def wrap(*args, _l=layer, _original=original, _signature=signature, _policy=native_policy, **kwargs):
                    import torch
                    bound = _signature.bind(*args, **kwargs)
                    logits = bound.arguments['router_logits']
                    nw, ni = _original(*args, **kwargs)
                    require(len(logits) == len(self.contexts), 'router token ownership differs')
                    if any(r.get('audit') for r in requests) and _l not in self.native_validation:
                        repeated_weights,repeated_ids=_original(*args,**kwargs)
                        repeat_equal=bool(torch.equal(repeated_weights,nw) and torch.equal(repeated_ids,ni))
                        require(repeat_equal,'same-state native selector is not exactly repeatable')
                        q = logits.float().softmax(-1) if _policy['scoring'] == 'softmax' else logits.float().sigmoid()
                        expected = q.gather(1, ni.long())
                        if _policy['renormalize']:
                            expected /= expected.sum(-1, keepdim=True).clamp_min(1e-30)
                        expected *= _policy.get('scale', 1.)
                        if _policy.get('expert_scale') is not None:
                            expected *= _policy['expert_scale'][ni.long()].to(expected.dtype)
                        exact = bool(torch.allclose(expected, nw.float(), rtol=.02, atol=.02))
                        require(exact, 'native affinity/normalization/scaling equation differs')
                        self.native_validation[_l] = {'native_equation_passed': exact,
                            'same_state_native_repeat_exact':repeat_equal,'rtol': .02, 'atol': .02}
                    weights, ids = nw, ni
                    for uid in self.req_ids:
                        r = self.requests[uid]
                        index = [j for j, c in enumerate(self.contexts) if c and c['uid'] == uid]
                        if not index:
                            continue
                        ix = torch.tensor(index, device=logits.device)
                        action = r.get('action')
                        active = [bool(action and action['layer'] == _l and self.contexts[j]['reasoning'] and
                                       len(r['prefix']) <= self.contexts[j]['output_position'] <
                                       len(r['prefix'])+action['pulse_tokens']) for j in index]
                        changed = torch.zeros(len(index), dtype=torch.bool, device=logits.device)
                        if any(active):
                            aw, ai, changed = hard_select(logits[ix], nw[ix], ni[ix], action['experts'],
                                                         action['direction'], active=active, **_policy)
                            if weights is nw:
                                weights, ids = nw.clone(), ni.clone()
                            weights[ix], ids[ix] = aw, ai
                        stats = self.counts[uid]
                        stats['rows'] += len(index)
                        stats['active'] += sum(active)
                        stats['changed'] += int(changed.sum().item())
                        stats['finite_weights'] &= bool(torch.isfinite(weights[ix]).all())
                        require(stats['finite_weights'], 'nonfinite dispatched weights')
                        if bool(changed.any()):
                            selected_ids=ids[ix][changed].long()
                            affinities=logits[ix][changed].float()
                            affinities=affinities.softmax(-1) if _policy['scoring']=='softmax' else affinities.sigmoid()
                            expected=affinities.gather(1,selected_ids)
                            if _policy['renormalize']:expected/=expected.sum(-1,keepdim=True).clamp_min(1e-30)
                            expected*=_policy.get('scale',1.)
                            if _policy.get('expert_scale') is not None:expected*=_policy['expert_scale'][selected_ids].to(expected.dtype)
                            require(bool(torch.allclose(expected,weights[ix][changed].float(),rtol=.02,atol=.02)),
                                    'intervened affinity/normalization/scaling equation differs')
                        require(all(len(set(v)) == self.contract['native_k'] for v in ids[ix].tolist()),
                                'native expert count or distinctness changed')
                        if any(active):
                            active_ids = ids[ix][torch.as_tensor(active, device=ix.device)]
                            wanted_ids = set(action['experts'])
                            passed = all((wanted_ids <= set(v) if action['direction'] == 'activate'
                                          else not wanted_ids.intersection(v)) for v in active_ids.tolist())
                            stats['target_contract'] &= passed
                            require(passed, 'assigned inclusion/exclusion was not dispatched')
                        unchanged = ~changed
                        stats['noop_exact'] &= bool(torch.equal(ids[ix][unchanged], ni[ix][unchanged]) and
                                                    torch.equal(weights[ix][unchanged], nw[ix][unchanged]))
                        if r.get('replay'):
                            # Main-paper native representations belong to the
                            # input token (the historical teacher-forced view),
                            # unlike intervention decisions predicting the next.
                            keep = [j for j in index if 1 <= self.contexts[j]['output_position'] <= len(r['prefix'])]
                            if keep:
                                take = torch.tensor(keep, device=logits.device)
                                record = self.records[uid].setdefault(str(_l), {'positions': [], 'native_ids': [], 'native_weights': []})
                                record['positions'].extend(self.contexts[j]['output_position']-1 for j in keep)
                                record['native_ids'].extend(ni[take].detach().cpu().tolist())
                                record['native_weights'].extend(nw[take].detach().float().cpu().tolist())
                            continue
                        keep = [j for j in index if len(r['prefix']) <= self.contexts[j]['output_position'] <
                                len(r['prefix'])+r['window'] and _l in r.get('capture_layers', [])]
                        if keep:
                            take = torch.tensor(keep, device=logits.device)
                            record = self.records[uid].setdefault(str(_l), {'positions': [], 'native_ids': [],
                                'native_weights': [], 'executed_ids': [], 'executed_weights': [], 'active': []})
                            record['positions'].extend(self.contexts[j]['output_position']-len(r['prefix']) for j in keep)
                            record['native_ids'].extend(ni[take].detach().cpu().tolist())
                            record['native_weights'].extend(nw[take].detach().float().cpu().tolist())
                            record['executed_ids'].extend(ids[take].detach().cpu().tolist())
                            record['executed_weights'].extend(weights[take].detach().float().cpu().tolist())
                            record['active'].extend(bool(action and action['layer'] == _l and self.contexts[j]['reasoning'] and
                                                        self.contexts[j]['output_position'] < len(r['prefix'])+action['pulse_tokens'])
                                                    for j in keep)
                        if r.get('audit') and uid not in self.sampled_states and any(active):
                            self.sampled_states[uid] = {'layer': _l, 'logits': logits[ix][:8].detach().float().cpu().tolist(),
                                                       'native_ids': ni[ix][:8].detach().cpu().tolist(),
                                                       'native_weights': nw[ix][:8].detach().float().cpu().tolist()}
                    return weights, ids
                router._compute_routing = wrap
            def before(module, args, kwargs):
                bound = inspect.signature(module.forward).bind(*args, **kwargs)
                token_ids = bound.arguments['input_ids'].detach().cpu().tolist()
                positions = bound.arguments['positions'].detach().cpu().tolist()
                contexts = [None] * len(token_ids)
                require(self.boundaries and self.boundaries[-1] <= len(token_ids), 'missing request row boundaries')
                for i, uid in enumerate(self.req_ids):
                    start, stop = self.boundaries[i:i+2]
                    pos = [axis[start:stop] for axis in positions] if positions and isinstance(positions[0], list) else positions[start:stop]
                    observed = self.clocks[uid].observe(token_ids[start:stop], pos)
                    contexts[start:stop] = [dict(c, uid=uid) for c in observed]
                self.contexts = contexts
            self.handle = self.model.register_forward_pre_hook(before, with_kwargs=True)
        except BaseException:
            self.restore()
            raise

    def restore(self):
        self.runner._prepare_inputs = self.original_prepare
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        for router, original in self.originals:
            router._compute_routing = original

    def finish(self):
        self.restore()
        for uid, r in self.requests.items():
            for layer, record in self.records[uid].items():
                require(record['positions'] == list(range(len(record['positions']))), 'capture has gaps or duplicate prediction positions')
        return {'capture': self.records, 'stats': self.counts, 'geometry': self.geometry,
                'sampled_states': self.sampled_states, 'native_validation': self.native_validation,
                'closed': {uid: c.closed for uid, c in self.clocks.items()},
                'capture_sha256': digest(self.records)}


class RoutingWorkerExtension:
    def fr_metrics(self):
        import resource
        import torch
        cache = self.vllm_config.cache_config
        return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'free_device_bytes': torch.cuda.mem_get_info()[0],
                'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'num_gpu_blocks': getattr(cache, 'num_gpu_blocks', None),
                'block_size': getattr(cache, 'block_size', None),
                'kv_cache_memory_bytes': getattr(cache, 'kv_cache_memory_bytes', None)}

    def fr_begin(self, *, contract, requests):
        cfg = self.vllm_config
        require(cfg.speculative_config is None and cfg.model_config.enforce_eager and
                not cfg.cache_config.enable_prefix_caching and not cfg.scheduler_config.async_scheduling,
                'unqualified scheduling/cache profile')
        require(getattr(self, '_fr_capture', None) is None, 'capture already attached')
        self._fr_capture = RequestCapture(self.model_runner, contract, requests)
        return {'status': 'attached', 'requests': len(requests)}

    def fr_end(self):
        require(self._fr_capture is not None, 'missing request capture')
        try:
            return self._fr_capture.finish()
        finally:
            self._fr_capture = None

    def fr_abort(self):
        capture = getattr(self, '_fr_capture', None)
        if capture is not None:
            capture.restore()
        self._fr_capture = None
        return {'status': 'detached'}

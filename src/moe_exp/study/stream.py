"""Streaming request-aware routing hooks for continuous admission.

Same intervention as `observer.RequestCapture` (hard inclusion/exclusion through
`hard_select` on the reasoning positions [len(prefix), len(prefix)+pulse) of the
assigned layer), but requests are added and released while the engine runs, the
per-layer callbacks return immediately when no request is inside its pulse or
capture window, and a request that is recomputed from position 0 (vLLM
preemption, or re-admission of saved tokens in a later job) has its pulse
re-applied at the same absolute positions, so cached states of the pulse tokens
are rebuilt under the intervention instead of under native routing.
"""
from __future__ import annotations
import inspect
from .common import digest, require
from .observer import discover
from moe_exp.final_routing.operator import hard_select, policy


class Clock:
    """Absolute prediction clock; a pass restarting at position 0 replays the same tokens."""
    def __init__(self, prompt, prefix, closure):
        require(prompt and closure and all(s for s in closure), 'prompt and native closure sequences required')
        self.prompt = len(prompt)
        self.saved = list(prompt)+list(prefix)
        self.closure = [tuple(s) for s in closure]
        self.width = max(map(len, self.closure))
        self.history = []
        self.tail, self.closed, self.last, self.passes = [], False, -1, 1

    def observe(self, ids, positions):
        if positions and isinstance(positions[0], (list, tuple)):
            require(len(positions) in (3, 4) and all(len(axis) == len(ids) for axis in positions) and
                    all(list(axis) == list(positions[0]) for axis in positions),
                    'non-text or malformed multi-axis token positions')
            positions = list(positions[0])
        require(len(ids) == len(positions), 'token/position shape differs')
        contexts = []
        for token, position in zip(ids, positions):
            if position == 0 and self.last >= 0:
                self.passes += 1
                self.last, self.tail, self.closed = -1, [], False
            require(type(position) is int and position == self.last+1,
                    'noncontiguous request: padding/cache/neighboring request')
            self.last = position
            require(position >= len(self.saved) or token == self.saved[position], 'runtime changed frozen input')
            if position < len(self.history):
                require(self.history[position] == token, 'recomputed request history differs')
            else:
                self.history.append(token)
            if position >= self.prompt:
                self.tail = (self.tail+[token])[-self.width:]
                if any(tuple(self.tail[-len(s):]) == s for s in self.closure):
                    self.closed = True
            contexts.append({'output_position': position-self.prompt+1, 'reasoning': not self.closed})
        return contexts


class StreamCapture:
    def __init__(self, runner, contract):
        self.runner, self.contract = runner, contract
        self.model = runner.get_model()
        self.aliases, self.requests, self.clocks, self.records, self.counts = {}, {}, {}, {}, {}
        self.native_validation, self.geometry, self.originals = {}, [], []
        self.boundaries, self.req_ids, self.contexts, self.plan = [], [], [], {}
        self.handle = None
        self.original_prepare = runner._prepare_inputs
        routers = discover(self.model)
        require(sorted(routers) == contract['layers'], 'physical routed-layer map differs')
        hooked = set(contract['hook_layers'])
        require(hooked <= routers.keys(), 'requested router is absent')
        try:
            def prepare(*args, **kwargs):
                result = self.original_prepare(*args, **kwargs)
                internal_ids = list(runner.input_batch.req_ids)
                unknown = set(internal_ids)-self.aliases.keys()
                require(not unknown, 'unregistered internal request IDs: '+repr(sorted(unknown)))
                self.req_ids = [self.aliases[i] for i in internal_ids]
                n = len(self.req_ids)
                require(len(set(self.req_ids)) == n, 'duplicate external request in worker batch')
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
                if layer not in hooked:
                    continue
                original = router._compute_routing
                self.originals.append((router, original))
                router._compute_routing = self._wrap(layer, original, inspect.signature(original), native_policy)

            def before(module, args, kwargs):
                bound = inspect.signature(module.forward).bind(*args, **kwargs)
                token_ids = bound.arguments['input_ids'].detach().cpu().tolist()
                positions = bound.arguments['positions'].detach().cpu().tolist()
                contexts = [None]*len(token_ids)
                plan = {}
                require(self.boundaries and self.boundaries[-1] <= len(token_ids), 'missing request row boundaries')
                for i, uid in enumerate(self.req_ids):
                    start, stop = self.boundaries[i:i+2]
                    pos = [axis[start:stop] for axis in positions] if positions and isinstance(positions[0], list) \
                        else positions[start:stop]
                    observed = self.clocks[uid].observe(token_ids[start:stop], pos)
                    contexts[start:stop] = observed
                    r = self.requests[uid]
                    origin = len(r['prefix'])
                    rows = [(start+j, c) for j, c in enumerate(observed) if origin <= c['output_position'] < r['end']]
                    if rows:
                        for layer in r['layers']:
                            plan.setdefault(layer, []).append((uid, rows))
                self.contexts, self.plan = contexts, plan
            self.handle = self.model.register_forward_pre_hook(before, with_kwargs=True)
        except BaseException:
            self.restore()
            raise

    def _wrap(self, layer, original, signature, native_policy):
        def wrap(*args, **kwargs):
            import torch
            nw, ni = original(*args, **kwargs)
            work = self.plan.get(layer)
            if layer not in self.native_validation and len(ni):
                logits = signature.bind(*args, **kwargs).arguments['router_logits']
                rw, ri = original(*args, **kwargs)
                repeat = bool(torch.equal(rw, nw) and torch.equal(ri, ni))
                require(repeat, 'same-state native selector is not exactly repeatable')
                require(bool(torch.allclose(self._expected(logits, ni, native_policy), nw.float(), rtol=.02, atol=.02)),
                        'native affinity/normalization/scaling equation differs')
                self.native_validation[layer] = {'native_equation_passed': True,
                                                 'same_state_native_repeat_exact': repeat, 'rtol': .02, 'atol': .02}
            if not work:
                return nw, ni
            logits = signature.bind(*args, **kwargs).arguments['router_logits']
            require(len(logits) == len(self.contexts), 'router token ownership differs')
            weights, ids = nw, ni
            for uid, rows in work:
                r = self.requests[uid]
                action, origin = r.get('action'), len(r['prefix'])
                index = [j for j, _ in rows]
                ix = torch.tensor(index, device=logits.device)
                active = [bool(action and action['layer'] == layer and c['reasoning'] and
                               c['output_position'] < origin+action['pulse_tokens']) for _, c in rows]
                stats = self.counts[uid]
                if any(active):
                    aw, ai, changed = hard_select(logits[ix], nw[ix], ni[ix], action['experts'],
                                                  action['direction'], active=active, **native_policy)
                    if weights is nw:
                        weights, ids = nw.clone(), ni.clone()
                    weights[ix], ids[ix] = aw, ai
                    require(bool(torch.isfinite(weights[ix]).all()), 'nonfinite dispatched weights')
                    if bool(changed.any()):
                        sel = ids[ix][changed].long()
                        expected = self._expected(logits[ix][changed], sel, native_policy)
                        require(bool(torch.allclose(expected, weights[ix][changed].float(), rtol=.02, atol=.02)),
                                'intervened affinity/normalization/scaling equation differs')
                    flags = torch.as_tensor(active, device=ix.device)
                    dispatched = ids[ix][flags].tolist()
                    require(all(len(set(v)) == self.contract['native_k'] for v in dispatched),
                            'native expert count or distinctness changed')
                    wanted = set(action['experts'])
                    passed = all((wanted <= set(v) if action['direction'] == 'activate' else not wanted.intersection(v))
                                 for v in dispatched)
                    stats['target_contract'] &= passed
                    require(passed, 'assigned inclusion/exclusion was not dispatched')
                    unchanged = ~changed
                    stats['noop_exact'] &= bool(torch.equal(ids[ix][unchanged], ni[ix][unchanged]) and
                                                torch.equal(weights[ix][unchanged], nw[ix][unchanged]))
                    positions = [c['output_position']-origin for (_, c), a in zip(rows, active) if a]
                    stats['active_positions'].update(positions)
                    stats['changed_positions'].update(p for p, ch in zip(positions, changed[flags].tolist()) if ch)
                if layer in r['capture_layers']:
                    record = self.records[uid].setdefault(str(layer), {'positions': [], 'native_ids': [],
                        'native_weights': [], 'executed_ids': [], 'executed_weights': [], 'active': []})
                    done = len(record['positions'])
                    keep = [(k, j, c) for k, (j, c) in enumerate(rows) if c['output_position']-origin >= done]
                    if keep:
                        require(keep[0][2]['output_position']-origin == done, 'capture has gaps')
                        take = torch.tensor([j for _, j, _ in keep], device=logits.device)
                        record['positions'].extend(c['output_position']-origin for _, _, c in keep)
                        record['native_ids'].extend(ni[take].detach().cpu().tolist())
                        record['native_weights'].extend(nw[take].detach().float().cpu().tolist())
                        record['executed_ids'].extend(ids[take].detach().cpu().tolist())
                        record['executed_weights'].extend(weights[take].detach().float().cpu().tolist())
                        record['active'].extend(active[k] for k, _, _ in keep)
            return weights, ids
        return wrap

    @staticmethod
    def _expected(logits, selected, native_policy):
        q = logits.float().softmax(-1) if native_policy['scoring'] == 'softmax' else logits.float().sigmoid()
        expected = q.gather(1, selected.long())
        if native_policy['renormalize']:
            expected = expected/expected.sum(-1, keepdim=True).clamp_min(1e-30)
        expected = expected*native_policy.get('scale', 1.)
        if native_policy.get('expert_scale') is not None:
            expected = expected*native_policy['expert_scale'][selected.long()].to(expected.dtype)
        return expected

    def add(self, requests, aliases):
        for r in requests:
            uid = r['uid']
            require(uid not in self.requests, 'request already attached')
            action = r.get('action')
            layers = set(r['capture_layers']) | ({action['layer']} if action else set())
            require(layers <= set(self.contract['hook_layers']), 'request layer is not hooked')
            end = len(r['prefix'])+max(r['window'], action['pulse_tokens'] if action else 0)
            self.requests[uid] = {**r, 'layers': sorted(layers), 'end': end}
            self.clocks[uid] = Clock(r['prompt'], r['prefix'], r['closure'])
            self.records[uid] = {}
            self.counts[uid] = {'active_positions': set(), 'changed_positions': set(),
                                'noop_exact': True, 'target_contract': True}
        require(len(aliases) == len(set(aliases.values())), 'request map is not one-to-one')
        require(set(aliases.values()) == {r['uid'] for r in requests}, 'request map does not match admission')
        require(not set(aliases) & self.aliases.keys(), 'internal request ID reused')
        self.aliases.update(aliases)
        return {'mapping_sha256': digest(aliases), 'requests': len(requests)}

    def release(self, uid):
        require(uid in self.requests, 'unknown released request')
        self.requests.pop(uid)
        clock = self.clocks.pop(uid)
        for internal in [k for k, v in self.aliases.items() if v == uid]:
            del self.aliases[internal]
        stats = self.counts.pop(uid)
        stats = {**stats, 'active_positions': sorted(stats['active_positions']),
                 'changed_positions': sorted(stats['changed_positions'])}
        records = self.records.pop(uid)
        value = {'capture': records, 'stats': stats, 'closed': clock.closed, 'passes': clock.passes,
                 'observed_tokens': len(clock.history)}
        return {**value, 'sha256': digest(value)}

    def restore(self):
        self.runner._prepare_inputs = self.original_prepare
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        for router, original in self.originals:
            router._compute_routing = original
        self.originals = []


class StreamWorkerExtension:
    def fs_begin(self, *, contract):
        cfg = self.vllm_config
        require(cfg.speculative_config is None and cfg.model_config.enforce_eager and
                not cfg.cache_config.enable_prefix_caching and not cfg.scheduler_config.async_scheduling,
                'unqualified scheduling/cache profile')
        require(getattr(self, '_fs_capture', None) is None, 'capture already attached')
        self._fs_capture = StreamCapture(self.model_runner, contract)
        return {'status': 'attached', 'geometry': self._fs_capture.geometry}

    def fs_add(self, *, requests, aliases):
        return self._fs_capture.add(requests, aliases)

    def fs_release(self, *, uids):
        return {uid: self._fs_capture.release(uid) for uid in uids}

    def fs_validation(self):
        return self._fs_capture.native_validation

    def fs_end(self):
        capture = getattr(self, '_fs_capture', None)
        if capture is not None:
            capture.restore()
        self._fs_capture = None
        return {'status': 'detached'}

    def fr_metrics(self):
        import resource
        import torch
        cache = self.vllm_config.cache_config
        return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'free_device_bytes': torch.cuda.mem_get_info()[0],
                'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'num_gpu_blocks': getattr(cache, 'num_gpu_blocks', None),
                'block_size': getattr(cache, 'block_size', None)}

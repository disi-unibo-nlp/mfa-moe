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
from moe_exp.io import digest, require
from moe_exp.steering.observer import discover
from moe_exp.steering.operators import hard_select, policy, score_override_select, score_override_dual


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
            contexts.append({'output_position': position-self.prompt+1, 'reasoning': not self.closed,
                             'absolute_position': position, 'prompt': position < self.prompt,
                             'generated_position': position-self.prompt})
        return contexts


class StreamCapture:
    def __init__(self, runner, contract):
        self.runner, self.contract = runner, contract
        self.model = runner.get_model()
        self.aliases, self.requests, self.clocks, self.records, self.counts = {}, {}, {}, {}, {}
        self.native_validation, self.geometry, self.originals = {}, [], []
        self.whole_slots, self.whole_totals, self.whole_role_cache = {}, {}, {}
        self.next_whole_slot = 0
        self.whole_step, self.whole_row_cache, self.whole_audit_pending = 0, None, []
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
                # The previous step has finished on the device; its audit slices belong to the old contexts.
                self._flush_whole_audit()
                self.whole_step += 1
                contexts = [None]*len(token_ids)
                plan = {}
                require(self.boundaries and self.boundaries[-1] <= len(token_ids), 'missing request row boundaries')
                for i, uid in enumerate(self.req_ids):
                    start, stop = self.boundaries[i:i+2]
                    pos = [axis[start:stop] for axis in positions] if positions and isinstance(positions[0], list) \
                        else positions[start:stop]
                    clock = self.clocks[uid]
                    passes = clock.passes
                    observed = clock.observe(token_ids[start:stop], pos)
                    if clock.passes != passes:  # recompute from position 0: the capture is rebuilt from this pass
                        self.records[uid] = {}
                        self.counts[uid].update(active_positions=set(), changed_positions=set(), dose={})
                        if uid in self.whole_slots:
                            for totals in self.whole_totals.values():
                                if self.whole_slots[uid] < len(totals):
                                    totals[self.whole_slots[uid]].zero_()
                    contexts[start:stop] = observed
                    r = self.requests[uid]
                    origin = len(r['prefix'])
                    if r.get('whole'):
                        rows = [(start+j, c) for j, c in enumerate(observed)
                                if c['prompt'] or (c['reasoning'] and c['generated_position'] < r['end'])]
                    else:
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
            if all(self.requests[uid].get('whole') for uid, _ in work):
                return self._whole_batch(layer, work, logits, nw, ni, native_policy)
            weights, ids = nw, ni
            for uid, rows in work:
                r = self.requests[uid]
                if r.get('whole'):
                    if weights is nw:
                        weights, ids = nw.clone(), ni.clone()
                    self._whole_layer(uid, layer, rows, logits, nw, ni, weights, ids, native_policy)
                    continue
                action, origin = r['actions'].get(layer), len(r['prefix'])
                index = [j for j, _ in rows]
                ix = torch.tensor(index, device=logits.device)
                active = [bool(action and c['reasoning'] and
                               c['output_position'] < origin+action['pulse_tokens']) for _, c in rows]
                stats = self.counts[uid]
                if any(active):
                    operator = action.get('operator', 'hard_select')
                    if operator == 'hard_select':
                        aw, ai, changed = hard_select(logits[ix], nw[ix], ni[ix], action['experts'],
                                                      action['direction'], active=active, **native_policy)
                        validation_logits = logits[ix]
                    elif operator == 'score_override':
                        aw, ai, changed = score_override_select(
                            logits[ix], nw[ix], ni[ix], action['experts'], action['direction'],
                            epsilon=action.get('epsilon', .01), active=active, **native_policy)
                        validation_logits = logits[ix]
                    else:
                        raise ValueError('unknown routing operator: '+operator)
                    if weights is nw:
                        weights, ids = nw.clone(), ni.clone()
                    weights[ix], ids[ix] = aw, ai
                    require(bool(torch.isfinite(weights[ix]).all()), 'nonfinite dispatched weights')
                    if bool(changed.any()) and operator == 'hard_select':
                        sel = ids[ix][changed].long()
                        expected = self._expected(validation_logits[changed], sel, native_policy)
                        require(bool(torch.allclose(expected, weights[ix][changed].float(), rtol=.02, atol=.02)),
                                'intervened affinity/normalization/scaling equation differs')
                    if operator == 'score_override':
                        scores = validation_logits.float().log_softmax(-1)
                        flags = torch.as_tensor(active, device=scores.device)
                        target = torch.tensor(action['experts'], device=scores.device)
                        extreme = (scores.max(-1).values+action.get('epsilon', .01)
                                   if action['direction'] == 'activate' else
                                   scores.min(-1).values-action.get('epsilon', .01))
                        scores[:, target] = extreme[:, None]
                        expected = self._expected(scores[flags], ids[ix][flags], native_policy)
                        require(bool(torch.allclose(expected, weights[ix][flags].float(), rtol=.02, atol=.02)),
                                'score override mixture normalization/scaling differs')
                    flags = torch.as_tensor(active, device=ix.device)
                    dispatched = ids[ix][flags].tolist()
                    require(all(len(set(v)) == self.contract['native_k'] for v in dispatched),
                            'native expert count or distinctness changed')
                    wanted = set(action['experts'])
                    passed = all((wanted <= set(v) if action['direction'] == 'activate' else not wanted.intersection(v))
                                 for v in dispatched)
                    stats['target_contract'] &= passed
                    require(passed, 'assigned inclusion/exclusion was not dispatched '
                            f'operator={operator} direction={action["direction"]} '
                            f'wanted={sorted(wanted)} dispatched={dispatched}')
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

    def _whole_buffer(self, layer, device):
        """Grow a small aggregate counter matrix and refresh its per-unit views."""
        import torch
        totals = self.whole_totals.get(layer)
        if totals is None or len(totals) < self.next_whole_slot:
            capacity = max(64, 1 << (self.next_whole_slot-1).bit_length())
            grown = torch.zeros((capacity, len(DOSE_KEYS)), device=device, dtype=torch.float32)
            if totals is not None:
                grown[:len(totals)] = totals
            self.whole_totals[layer] = totals = grown
            for uid, slot in self.whole_slots.items():
                if str(layer) in self.counts[uid]['dose']:
                    self.counts[uid]['dose'][str(layer)] = dict(zip(DOSE_KEYS, totals[slot].unbind()))
        return totals

    def _whole_roles(self, work, device):
        """Cache every hooked layer's role masks for the current scheduler request order.

        Returns:
            Request identity, layer-to-index map, a [layers, requests, 2, experts] activate/deactivate
            mask and, only when some request nudges, a [layers, requests, experts] float shift (else
            None), copied to the device once per change of batch membership.
        """
        import torch
        identity = tuple(uid for uid, _ in work)
        cached = self.whole_role_cache.get('all')
        if cached is None or cached[0] != identity:
            layers = list(self.contract['hook_layers'])
            roles = torch.zeros((len(layers), len(work), 2, self.contract['experts']), dtype=torch.bool)
            shifts = torch.zeros((len(layers), len(work), self.contract['experts']), dtype=torch.float32)
            for index, (uid, _) in enumerate(work):
                for position, layer in enumerate(layers):
                    action = self.requests[uid]['actions'].get(layer)
                    if action:
                        roles[position, index, 0, action.get('activate', [])] = True
                        roles[position, index, 1, action.get('deactivate', [])] = True
                        shifts[position, index, action.get('shift', [])] = action.get('delta', 0.)
            nudging = bool(shifts.any())
            require(not (nudging and roles.any()), 'a batch cannot mix nudged and forced whole rows')
            cached = self.whole_role_cache['all'] = (identity, {l: i for i, l in enumerate(layers)}, roles.to(device),
                                                     shifts.to(device) if nudging else None)
        return cached

    def _whole_rows(self, work, device):
        """Build one step's row ownership tensors once; every hooked layer sees the same whole rows."""
        import torch
        identity = (self.whole_step, tuple((uid, len(rows)) for uid, rows in work))
        if self.whole_row_cache is None or self.whole_row_cache[0] != identity:
            indices, groups, base, prompts, audit = [], [], [], [], []
            for group, (uid, rows) in enumerate(work):
                request = self.requests[uid]
                for index, context in rows:
                    indices.append(index)
                    groups.append(group)
                    prompts.append(context['prompt'])
                    base.append(bool(context['reasoning'] and (not context['prompt'] or request['prompt_scope'])))
                    audit.append(bool(request.get('audit_capture')))
            row_values = torch.tensor([indices, groups, base, prompts], dtype=torch.long).to(device)
            group_values = torch.tensor([[len(rows) for _, rows in work],
                                         [self.whole_slots[uid] for uid, _ in work]], dtype=torch.long).to(device)
            audit_rows = [i for i, flag in enumerate(audit) if flag]
            self.whole_row_cache = (identity, {
                'take': row_values[0], 'groups': row_values[1], 'base': row_values[2].bool(),
                'prompt': row_values[3].bool(), 'lengths': group_values[0], 'slots': group_values[1],
                'audit': torch.tensor(audit_rows, dtype=torch.long).to(device) if audit_rows else None,
                'audit_owners': [(uid, rows) for uid, rows in work if self.requests[uid].get('audit_capture')]})
        return self.whole_row_cache[1]

    def _whole_batch(self, layer, work, logits, nw, ni, native_policy):
        """Execute and measure one layer for every whole row, without host synchronization.

        Eager kernels only: both tensor-parallel ranks must reproduce identical routing and counters.
        """
        import torch
        from moe_exp.steering.operators import score_override_dual_masked, score_shift_masked
        step = self._whole_rows(work, logits.device)
        _, positions, all_roles, all_shifts = self._whole_roles(work, logits.device)
        layer_roles = all_roles[positions[layer]]
        roles, take = layer_roles[step['groups']], step['take']
        if all_shifts is None:
            # Forced (dual) rows. Rows of requests without targets at this layer keep their exact native bits.
            active = step['base'] & layer_roles.flatten(1).any(-1)[step['groups']]
            aw, ai = score_override_dual_masked(logits[take], nw[take], ni[take], roles[:, 0], roles[:, 1],
                                                active, **native_policy)
            nudged = None
        else:
            # Nudged rows: lower the chosen experts' logits and let the router select as usual.
            shift = all_shifts[positions[layer]][step['groups']]
            active = step['base'] & (shift != 0).any(-1)
            aw, ai = score_shift_masked(logits[take], nw[take], ni[take], shift, active, **native_policy)
            nudged = shift != 0
        weights, ids = nw.clone(), ni.clone()
        weights[take], ids[take] = aw, ai
        values = batched_dose(nw[take], ni[take], aw, ai, active, step['prompt'], roles, step['lengths'], nudged)
        totals = self._whole_buffer(layer, logits.device)
        totals.index_add_(0, step['slots'], values)
        for uid, _ in work:
            if str(layer) not in self.counts[uid]['dose']:
                self.counts[uid]['dose'][str(layer)] = dict(zip(DOSE_KEYS, totals[self.whole_slots[uid]].unbind()))
        if step['audit'] is not None:
            rows = take[step['audit']]
            packed = torch.cat((ni[rows].float(), nw[rows].float(), ai[step['audit']].float(),
                                aw[step['audit']].float(), active[step['audit'], None].float()), -1)
            self.whole_audit_pending.append((layer, step['audit_owners'], packed))
        return weights, ids

    def _flush_whole_audit(self):
        """Move the finished step's audit slices to the host in one transfer."""
        if not self.whole_audit_pending:
            return
        import torch
        pending, self.whole_audit_pending = self.whole_audit_pending, []
        host = torch.cat([packed for _, _, packed in pending]).cpu().tolist()
        offset = 0
        for layer, owners, packed in pending:
            self._whole_audit(layer, owners, host[offset:offset+len(packed)])
            offset += len(packed)

    def _whole_audit(self, layer, owners, packed):
        """Split one layer's audit rows into the unchanged per-request audit records."""
        offset, width = 0, self.contract['native_k']
        for uid, rows in owners:
            record = self.records[uid].setdefault(str(layer), {'positions': [], 'native_ids': [],
                'native_weights': [], 'executed_ids': [], 'executed_weights': [], 'active': []})
            for values, (_, context) in zip(packed[offset:offset+len(rows)], rows):
                position = context['generated_position']
                if position < len(record['positions']):
                    continue
                require(position == len(record['positions']), 'whole audit capture gap')
                record['positions'].append(position)
                record['native_ids'].append([int(v) for v in values[:width]])
                record['native_weights'].append(values[width:2*width])
                record['executed_ids'].append([int(v) for v in values[2*width:3*width]])
                record['executed_weights'].append(values[3*width:4*width])
                record['active'].append(bool(values[4*width]))
            offset += len(rows)

    def _whole_layer(self, uid, layer, rows, logits, nw, ni, weights, ids, native_policy):
        """Apply one dual action and reduce dose online; audit rows retain generated captures."""
        import torch
        request = self.requests[uid]
        action = request['actions'].get(layer, {})
        require(not action.get('shift'), 'nudged whole rows run only on the batched path')
        activate, deactivate = action.get('activate', []), action.get('deactivate', [])
        active = [bool((activate or deactivate) and c['reasoning'] and
                       (not c['prompt'] or request['prompt_scope'])) for _, c in rows]
        index = torch.tensor([i for i, _ in rows], device=logits.device)
        aw, ai = nw[index], ni[index]
        if any(active):
            aw, ai, _ = score_override_dual(logits[index], aw, ai, activate, deactivate,
                                            active=active, epsilon=.01, **native_policy)
        weights[index], ids[index] = aw, ai
        aggregate = dose_totals(nw[index], ni[index], aw, ai, active, [c['prompt'] for _, c in rows],
                                activate, deactivate, self.contract['experts'])
        dose = self.counts[uid]['dose'].setdefault(str(layer), {key: 0 for key in aggregate})
        for key, value in aggregate.items():
            dose[key] += value
        if request.get('audit_capture'):
            record = self.records[uid].setdefault(str(layer), {'positions': [], 'native_ids': [],
                'native_weights': [], 'executed_ids': [], 'executed_weights': [], 'active': []})
            keep = [(j, c) for j, c in enumerate(c for _, c in rows)
                    if c['generated_position'] >= len(record['positions'])]
            if keep:
                require(keep[0][1]['generated_position'] == len(record['positions']), 'whole audit capture gap')
                take = torch.tensor([j for j, _ in keep], device=logits.device)
                record['positions'].extend(c['generated_position'] for _, c in keep)
                record['native_ids'].extend(ni[index][take].detach().cpu().tolist())
                record['native_weights'].extend(nw[index][take].detach().float().cpu().tolist())
                record['executed_ids'].extend(ai[take].detach().cpu().tolist())
                record['executed_weights'].extend(aw[take].detach().float().cpu().tolist())
                record['active'].extend(active[j] for j, _ in keep)

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
            # One action (pre-registered design) or one action per layer (post-hoc strong-dose arm).
            actions = list(r.get('actions') or ([r['action']] if r.get('action') else []))
            if r.get('whole'):
                for action in actions:
                    positive, negative = set(action['activate']), set(action['deactivate'])
                    require(not positive & negative and len(positive) < self.contract['native_k'] and
                            self.contract['experts']-len(negative) >= self.contract['native_k'],
                            'whole dual roles cannot preserve native k')
                    operator = action.get('operator', 'score_override_dual')
                    require(operator in ('score_override_dual', 'score_shift'), 'unknown whole operator: '+operator)
                    if operator == 'score_shift':
                        # A nudge only lowers logits: no forced roles, a positive delta, valid distinct experts.
                        nudged = action.get('shift', [])
                        require(not positive and not negative and nudged and float(action.get('delta', 0)) > 0 and
                                len(set(nudged)) == len(nudged) and
                                all(type(e) is int and 0 <= e < self.contract['experts'] for e in nudged),
                                'invalid whole nudge action')
                self.whole_slots[uid] = self.next_whole_slot
                self.next_whole_slot += 1
            require(len({a['layer'] for a in actions}) == len(actions), 'at most one action per layer')
            layers = set(r['capture_layers']) | {a['layer'] for a in actions}
            require(layers <= set(self.contract['hook_layers']), 'request layer is not hooked')
            end = len(r['prefix'])+max([r['window']]+[a['pulse_tokens'] for a in actions])
            self.requests[uid] = {**r, 'actions': {a['layer']: a for a in actions}, 'layers': sorted(layers), 'end': end}
            self.clocks[uid] = Clock(r['prompt'], r['prefix'], r['closure'])
            self.records[uid] = {}
            self.counts[uid] = {'active_positions': set(), 'changed_positions': set(), 'dose': {},
                                'noop_exact': True, 'target_contract': True}
        require(len(aliases) == len(set(aliases.values())), 'request map is not one-to-one')
        require(set(aliases.values()) == {r['uid'] for r in requests}, 'request map does not match admission')
        require(not set(aliases) & self.aliases.keys(), 'internal request ID reused')
        self.aliases.update(aliases)
        return {'mapping_sha256': digest(aliases), 'requests': len(requests)}

    def release(self, uid):
        require(uid in self.requests, 'unknown released request')
        self._flush_whole_audit()
        self.requests.pop(uid)
        clock = self.clocks.pop(uid)
        for internal in [k for k, v in self.aliases.items() if v == uid]:
            del self.aliases[internal]
        stats = self.counts.pop(uid)
        self.whole_slots.pop(uid, None)
        if stats['dose']:
            import torch
            for layer, totals in stats['dose'].items():
                keys = list(totals)
                values = torch.stack([totals[key].double() for key in keys]).detach().cpu().tolist()
                stats['dose'][layer] = dict(zip(keys, values))
                require(stats['dose'][layer]['finite'] == stats['dose'][layer]['observations'] and
                        stats['dose'][layer]['membership_failures'] == 0, 'whole routing contract failed')
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


DOSE_KEYS = ('active_positions', 'active_prompt_positions', 'active_generated_positions', 'changed_membership',
             'native_target_weight', 'executed_target_weight', 'native_activate_weight', 'executed_activate_weight',
             'native_deactivate_weight', 'executed_deactivate_weight', 'native_activate_share', 'executed_activate_share',
             'l1', 'finite', 'membership_failures', 'observations',
             'native_shift_weight', 'executed_shift_weight', 'native_shift_members', 'executed_shift_members')


def batched_dose(native_weights, native_ids, weights, ids, active, prompt, roles, lengths, nudged=None):
    """Reduce contiguous request segments deterministically, preserving exact rank agreement.

    `nudged` marks the experts a gentle nudge lowers (rows x experts); its counters record how much
    routing weight and how many top-k slots those experts hold before and after the nudge.
    """
    import torch
    experts = roles.shape[-1]
    native = torch.zeros((len(ids), experts), device=weights.device, dtype=torch.float32)
    executed = torch.zeros_like(native)
    native.scatter_(1, native_ids.long(), native_weights.float())
    executed.scatter_(1, ids.long(), weights.float())
    members = torch.zeros_like(native, dtype=torch.bool).scatter_(1, ids.long(), True)
    activate, deactivate = roles[:, 0], roles[:, 1]
    na, nd = (native*activate).sum(-1), (native*deactivate).sum(-1)
    ea, ed = (executed*activate).sum(-1), (executed*deactivate).sum(-1)
    changed = (native_ids.sort(-1).values != ids.sort(-1).values).any(-1)
    failed = ((activate & ~members).any(-1) | (deactivate & members).any(-1)) & active
    values = torch.stack((active, active & prompt, active & ~prompt, active & changed,
        (na+nd)*active, (ea+ed)*active, na*active, ea*active, nd*active, ed*active,
        na/native_weights.sum(-1).clamp_min(1e-30)*active,
        ea/weights.sum(-1).clamp_min(1e-30)*active,
        (native-executed).abs().sum(-1)*active, (~torch.isfinite(weights).all(-1)).float(),
        failed, torch.zeros_like(active),
        *shift_counters(native, executed, native_ids, members, nudged, active)), -1).float()
    # Atomic floating sums can differ between tensor-parallel ranks. Each
    # request owns a contiguous segment, so one fixed-order segment sum suffices.
    totals = (values.clone() if len(values) == len(lengths) else
              torch.segment_reduce(values, 'sum', lengths=lengths, unsafe=True))
    totals[:, 13] = (totals[:, 13] == 0).float()
    totals[:, 15] = 1
    return totals


def shift_counters(native, executed, native_ids, members, nudged, active):
    """Weight and top-k slots held by nudged experts, natively and after the nudge (zeros if none)."""
    import torch
    if nudged is None:
        zero = torch.zeros(len(native), device=native.device)
        return zero, zero, zero, zero
    native_members = torch.zeros_like(members).scatter_(1, native_ids.long(), True)
    return ((native*nudged).sum(-1)*active, (executed*nudged).sum(-1)*active,
            (native_members & nudged).sum(-1)*active, (members & nudged).sum(-1)*active)


def dose_totals(native_weights, native_ids, weights, ids, active, prompt, activate, deactivate, experts):
    """Reduce same-state dose to additive counters without retaining per-token arrays.

    Args:
        native_weights, native_ids, weights, ids: Native and executed top-k tensors.
        active, prompt: Token masks defining intervention and prompt scope.
        activate, deactivate: Assigned pair roles for this layer.
        experts: Physical routed expert count.

    Returns:
        Additive position, membership, target-weight and L1 counters.
    """
    import torch
    flags = torch.as_tensor(active, device=weights.device, dtype=torch.bool)
    prompt_mask = torch.as_tensor(prompt, device=weights.device, dtype=torch.bool)
    native_dense = torch.zeros((len(ids), experts), device=weights.device, dtype=torch.float32)
    executed_dense = torch.zeros_like(native_dense)
    native_dense.scatter_(1, native_ids.long(), native_weights.float())
    executed_dense.scatter_(1, ids.long(), weights.float())
    changed = (native_ids.sort(-1).values != ids.sort(-1).values).any(-1)
    targeted = sorted(set(activate+deactivate))
    member = torch.ones(len(ids), device=weights.device, dtype=torch.bool)
    if activate:
        wanted = torch.tensor(activate, device=ids.device)
        member &= (ids[:, :, None] == wanted).any(1).all(-1)
    if deactivate:
        unwanted = torch.tensor(deactivate, device=ids.device)
        member &= ~(ids[:, :, None] == unwanted).any((1, 2))
    return {
        'active_positions': flags.sum(),
        'active_prompt_positions': (flags & prompt_mask).sum(),
        'active_generated_positions': (flags & ~prompt_mask).sum(),
        'changed_membership': (flags & changed).sum(),
        'native_target_weight': (native_dense[:, targeted].sum(-1)*flags).sum(),
        'executed_target_weight': (executed_dense[:, targeted].sum(-1)*flags).sum(),
        'native_activate_weight': (native_dense[:, activate].sum(-1)*flags).sum(),
        'executed_activate_weight': (executed_dense[:, activate].sum(-1)*flags).sum(),
        'native_deactivate_weight': (native_dense[:, deactivate].sum(-1)*flags).sum(),
        'executed_deactivate_weight': (executed_dense[:, deactivate].sum(-1)*flags).sum(),
        'native_activate_share': (native_dense[:, activate].sum(-1)/native_weights.sum(-1).clamp_min(1e-30)*flags).sum(),
        'executed_activate_share': (executed_dense[:, activate].sum(-1)/weights.sum(-1).clamp_min(1e-30)*flags).sum(),
        'l1': ((native_dense-executed_dense).abs().sum(-1)*flags).sum(),
        'finite': torch.isfinite(weights).all().long(),
        'membership_failures': (flags & ~member).sum(),
        'observations': torch.ones((), device=weights.device, dtype=torch.long),
        # The per-request path never nudges; these keep its counters aligned with DOSE_KEYS.
        **{key: torch.zeros((), device=weights.device) for key in DOSE_KEYS[16:]},
    }


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

    def fs_state(self):
        """Expose request-map emptiness for the allocated abort/isolation canary."""
        capture = self._fs_capture
        return {'requests': len(capture.requests), 'aliases': len(capture.aliases),
                'clocks': len(capture.clocks), 'records': len(capture.records)}

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

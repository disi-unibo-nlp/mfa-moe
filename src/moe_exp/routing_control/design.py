"""Frozen families, sparse action constraints, pulse dispatch, and full-stage cost gates."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

TRANSITIONS = ('candidate_to_verify', 'approach_to_commit', 'failed_check_to_revise')
STAGES = {
    'qualification': .75, 'discovery': 2., 'mechanism': 2.25, 'utility': 4.75, 'reserve': .75,
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def families(question_keys: Sequence[str], edges: Sequence[Mapping], exact_groups=()) -> dict:
    """Connected components across every model alias, including paths through confirm."""
    parent = {key: key for key in question_keys}

    def find(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(a, b):
        if a not in parent or b not in parent:
            raise ValueError('duplicate edge contains an unknown canonical question')
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for edge in edges:
        join(edge['a'], edge['b'])
    for group in exact_groups:
        for key in group[1:]:
            join(group[0], key)
    groups = defaultdict(list)
    for key in sorted(parent):
        groups[find(key)].append(key)
    return {digest(members): members for members in groups.values()}


def freeze_pools(groups: Mapping[str, Sequence[str]], split: Mapping[str, str]) -> dict:
    eligible = []
    excluded = []
    for family, members in groups.items():
        if any(split.get(q) == 'confirm' for q in members):
            excluded.append(family)
        elif all(split.get(q) in ('dev', 'tune') for q in members):
            eligible.append(family)
        else:
            raise ValueError('unresolved family split')
    order = sorted(eligible, key=lambda f: hashlib.sha256(('routing-control-v1|' + f).encode()).hexdigest())
    pools, cursor = {}, 0
    for name, n in [('discovery', 48), ('mechanism', 128), ('utility', 96)]:
        pools[name] = order[cursor:cursor + n]
        cursor += n
    return {'order_rule': 'ascending SHA256(routing-control-v1|family_id)',
            'families': dict(groups), 'confirm_connected_excluded': sorted(excluded),
            'eligible_family_count': len(order), 'parent_pools': pools,
            'feasibility': 'PASS' if len(order) >= cursor else 'FAIL_INSUFFICIENT_FAMILIES',
            'representative_questions': {f: sorted(groups[f], key=lambda q: hashlib.sha256(
                ('forum-v1|' + q).encode()).hexdigest())[0] for f in order},
            'replacement': 'none; eligibility failures are retained and reported'}


@dataclass(frozen=True)
class Action:
    name: str
    transition: str
    experts: tuple[tuple[int, tuple[int, ...]], ...]
    bias: float

    def validate(self, *, n_layers=40, n_experts=256):
        layers = [layer for layer, _ in self.experts]
        if self.transition not in TRANSITIONS or self.bias not in (.5, 1.):
            raise ValueError('unregistered transition or bias')
        if not 1 <= len(layers) <= 4 or layers != list(range(min(layers), max(layers) + 1)):
            raise ValueError('actions require at most four adjacent ordered layers')
        for layer, ids in self.experts:
            if type(layer) is not int or not 0 <= layer < n_layers:
                raise ValueError('invalid layer')
            if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids):
                raise ValueError('actions require one or two distinct experts per layer')
            if any(type(e) is not int or not 0 <= e < n_experts for e in ids):
                raise ValueError('invalid expert')
        return self


def propose_actions(contrasts: Mapping[str, np.ndarray], *, source_families, discovery_families) -> list[Action]:
    """Matched-native contrasts propose one set per fixed transition, never causal claims.

    Each array contains family-mean paired differences [families, layers, experts].
    Unsupported/nonpositive proposals are omitted. Discovery supplies all matches/transforms.
    """
    if not set(source_families) <= set(discovery_families):
        raise ValueError('expert proposals contain a nondiscovery family')
    if set(contrasts) - set(TRANSITIONS):
        raise ValueError('no replacement transition search is registered')
    actions = []
    for transition in TRANSITIONS:
        if transition not in contrasts:
            continue
        array = np.asarray(contrasts[transition], float)
        if array.ndim != 3 or array.shape[0] != len(source_families) or not np.isfinite(array).all():
            raise ValueError('invalid paired family contrasts')
        if not array.shape[0] or array.shape[2] < 8:
            continue
        score = array.mean(axis=0)
        bands = []
        for left in range(score.shape[0]):
            for width in range(1, min(4, score.shape[0] - left) + 1):
                targets, gain = [], 0.
                for layer in range(left, left + width):
                    ids = tuple(int(e) for e in np.argsort(-score[layer], kind='stable')[:2]
                                if score[layer, e] > 0)
                    if not ids:
                        break
                    targets.append((layer, ids))
                    gain += float(score[layer, list(ids)].sum())
                if len(targets) == width:
                    bands.append((gain / width, -width, -left, tuple(targets)))
        if not bands:
            continue
        targets = max(bands)[3]
        for bias in (.5, 1.):
            actions.append(Action(f'{transition}_bias{bias:g}', transition, targets, bias).validate(
                n_layers=score.shape[0], n_experts=score.shape[1]))
    assert len(actions) <= 6
    return actions


def random_controls(action: Action, native_rates, *, n_sets=4, seed=20261001, tolerance=.10):
    """Freeze several random sets with identical layer/count support and matched native exposure.

    This is a proposal match only; realized dose must also be calibrated on discovery data.
    Insufficient matched experts fail feasibility without relaxing the tolerance.
    """
    action.validate()
    rates = np.asarray(native_rates, float)
    if rates.ndim != 2 or not np.isfinite(rates).all() or (rates < 0).any() or (rates > 1).any():
        raise ValueError('invalid discovery native expert-use rates')
    if n_sets < 2 or not 0 < tolerance < 1:
        raise ValueError('freeze several random sets and a bounded exposure match')
    rng = np.random.default_rng(seed)
    sets = []
    for k in range(n_sets):
        targets = []
        for layer, selected in action.experts:
            if layer >= rates.shape[0] or max(selected) >= rates.shape[1]:
                raise ValueError('expert-use rate support differs from the action')
            chosen = []
            for target in selected:
                rate = rates[layer, target]
                pool = [e for e in range(rates.shape[1]) if e not in selected and e not in chosen
                        and abs(rates[layer, e] - rate) <= tolerance * max(rate, 1e-12)]
                if not pool:
                    raise ValueError('insufficient exposure-matched random experts')
                chosen.append(int(rng.choice(pool)))
            targets.append((layer, tuple(chosen)))
        sets.append(Action(f'{action.name}_random{k}', action.transition, tuple(targets), action.bias).validate())
    if len({a.experts for a in sets}) != n_sets:
        raise ValueError('insufficient distinct random sets')
    return sets


def balanced_random_assignment(family_order, seeds=(0, 1), *, n_sets=4):
    if len(set(family_order)) != len(family_order) or len(set(seeds)) != len(seeds) or n_sets < 2:
        raise ValueError('invalid family/seed random-set assignment')
    # Rotating cycles differ between seeds; within each seed, counts differ by at most one.
    return [{'family': family, 'seed': seed, 'random_set': (i + j) % n_sets}
            for i, family in enumerate(family_order) for j, seed in enumerate(seeds)]


@dataclass(frozen=True)
class Template:
    starting_condition: str
    actions: tuple[Action, ...]
    slots: tuple[int, ...]
    horizon: int = 1024

    def validate(self):
        if self.starting_condition not in TRANSITIONS or not 1 <= len(self.actions) <= 2:
            raise ValueError('freeze one local starting condition and at most two actions')
        if self.slots != (0, 512)[:len(self.actions)] or self.horizon != 1024:
            raise ValueError('registered pulse slots are 0 and 512, with a 1024-token horizon')
        for action in self.actions:
            action.validate()
        return self

    def action_at(self, position: int, *, closure_position: int | None = None) -> Action | None:
        self.validate()
        if type(position) is not int or position < 0:
            raise ValueError('branch-relative position must be a nonnegative integer')
        if closure_position is not None and position >= closure_position:
            return None
        return next((action for slot, action in zip(self.slots, self.actions)
                     if slot <= position < slot + 256), None)

    def reversed(self):
        return Template(self.starting_condition, tuple(reversed(self.actions)), self.slots, self.horizon)


def edit_logits(logits, rows, *, layer: int, templates: Mapping[str, Template],
                closure_positions: Mapping[str, int] | None = None):
    """CPU reference for a future qualified worker adapter. Call the native router afterward.

    `rows` maps each batch row to (request_id, absolute branch-relative position).
    No top-k or shared-expert code is touched. Recomputed rows derive the same edit from position.
    """
    z = np.asarray(logits)
    if z.ndim != 2 or len(rows) != len(z) or not np.issubdtype(z.dtype, np.floating):
        raise ValueError('expected floating [tokens, experts] logits with explicit row ownership')
    if not np.isfinite(z).all():
        raise ValueError('nonfinite router logits')
    output = z.copy()
    for index, (request, position) in enumerate(rows):
        template = templates.get(request)
        if template is None:
            continue
        action = template.action_at(position, closure_position=(closure_positions or {}).get(request))
        if action is None:
            continue
        for target_layer, ids in action.experts:
            if target_layer == layer:
                if max(ids) >= z.shape[1]:
                    raise ValueError('target expert outside the runtime router')
                output[index, list(ids)] = (z[index, list(ids)].astype(np.float32) + action.bias).astype(z.dtype)
    return output


def price_stage(stage: str, *, gpu_count: int, decode_tokens: int, prefill_tokens: int,
                decode_tps: float, prefill_tps: float, loads: int, load_seconds: float,
                measurement_seconds: float, retry_seconds: float, overhead_seconds: float,
                throughput_qualified: bool) -> dict:
    terms = (decode_tps, prefill_tps, load_seconds, measurement_seconds, retry_seconds, overhead_seconds)
    if stage not in STAGES or type(gpu_count) is not int or gpu_count < 1:
        raise ValueError('unregistered stage or GPU count')
    if any(not math.isfinite(x) or x < 0 for x in terms) or min(decode_tps, prefill_tps) <= 0:
        raise ValueError('invalid runtime measurements')
    if any(type(x) is not int or x < 0 for x in (loads, decode_tokens, prefill_tokens)):
        raise ValueError('invalid full-stage counts')
    seconds = decode_tokens / decode_tps + prefill_tokens / prefill_tps + loads * load_seconds
    seconds += measurement_seconds + retry_seconds + overhead_seconds
    cost = gpu_count * seconds / 3600
    status = ('HOLD_UNQUALIFIED_TIMING' if not throughput_qualified else
              'PASS' if cost <= STAGES[stage] else 'STOP_REVISE_RESOURCE_PROPOSAL')
    return {'stage': stage, 'gpu_hours': cost, 'ceiling': STAGES[stage], 'status': status,
            'includes': ['all loads', 'prefill', 'decode', 'measurement', 'retries', 'overhead'],
            'seconds': seconds}

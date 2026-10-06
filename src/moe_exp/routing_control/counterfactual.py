"""CPU helpers for a discovery-only, single-position routing-loss screen.

The realized native token is an offline measurement target. It never enters
the routing metadata or an online decision. Each position starts from the exact
native prefix, so only one prediction row is intervened on.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed input seal: ' + str(path))
    return value


def prediction_input(row, position):
    if type(position) is not int or not 0 <= position < 4:
        raise ValueError('freeze the first four post-boundary prediction positions')
    original, prefix, suffix = (row[k] for k in ('prompt_ids', 'prefix_ids', 'native_suffix_ids'))
    if not original or not prefix or len(suffix) != 4:
        raise ValueError('complete native replay tokens required')
    if any(type(t) is not int or t < 0 for t in original + prefix + suffix):
        raise ValueError('invalid native token ID')
    return list(original) + list(prefix) + list(suffix[:position]), suffix[position]


def designated_logprob(entries, target):
    """Read the designated token, never substitute the sampled token's score."""
    if isinstance(entries, dict):
        if target not in entries:
            raise ValueError('designated token absent from logprobs')
        entry = entries[target]
        value = float(entry.logprob if hasattr(entry, 'logprob') else entry)
    else:
        found = [entry for entry in (entries or []) if len(entry) >= 2 and entry[0] == target]
        if len(found) != 1:
            raise ValueError('missing or duplicated designated-token logprob')
        value = float(found[0][1])
    if not math.isfinite(value) or value > 1e-6:
        raise ValueError('invalid raw next-token log probability')
    return value


def verify_route(route, targets, operator):
    """Check native k=8 at all layers and force membership at the edited layer."""
    if len(route) != 40 or any(len(ids) != 8 or len(set(ids)) != 8 or
        any(type(e) is not int or not 0 <= e < 256 for e in ids) for ids in route):
        raise ValueError('route must retain 40 layers and eight unique routed experts')
    layer, ids = targets
    if type(layer) is not int or not 0 <= layer < 40 or not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids):
        raise ValueError('invalid sparse target layer or expert pair')
    hits = sum(int(e) in set(map(int, route[layer])) for e in ids)
    if operator == 'force_positive' and hits != len(ids):
        raise ValueError('positive-force membership failed')
    if operator == 'force_negative' and hits:
        raise ValueError('negative-force membership failed')
    return hits


def pricing(prefill, requests, pilot_prefill, pilot_decode, pilot_fixed,
            pilot_warm, pilot_outside):
    if min(prefill, requests, pilot_prefill, pilot_decode, pilot_fixed, pilot_warm) <= 0:
        raise ValueError('positive complete-stage and observed pilot timings required')
    ratio = max(prefill / pilot_prefill, requests / pilot_decode)
    projected = 1.25 * pilot_fixed + 1.5 * ratio * pilot_warm + 1.5 * pilot_outside + 180
    minutes = max(30, math.ceil(projected / 300) * 5)
    return {'maximum_prefill_tokens_without_reuse': prefill,
        'maximum_generated_tokens': requests,
        'projected_seconds_including_shutdown_and_margin': projected,
        'proposed_wall_minutes': minutes, 'two_A100_allocation_GPU_hours': minutes / 30,
        'rule': '1.25x measured fixed setup + 1.5x larger prefill/decode ratio times warm batch + 1.5x outside-driver shutdown + 180s, rounded to 5min with 30min minimum',
        'caveat': 'Qualification of designated-token scoring and fixed-k force is separate from semantic validation; reprice after measured screen runtime.'}

"""Outcome-blind family/dose allocations and explicit continuation identities."""
from collections import defaultdict
import math
from .common import ARMS, BENCHMARKS, CLASSES, digest, require


def seed_for(seed, *parts):
    return int(digest([seed, 'final-routing-continuation-v1', *parts])[:8], 16)


def strengths(k):
    return [math.ceil(k * f) for f in (.25, .5, .75)]


def allocate(slots, seed):
    """Freeze both permitted designs from the shared 129-slot population."""
    slots = sorted({(r['benchmark'], r['family'], r['fold']) for r in slots})
    require(len(slots) == 129, 'expected exactly 129 frozen family/benchmark slots')
    require(len({(b, f) for b, f, _ in slots}) == 129, 'slot has conflicting folds')
    grouped = defaultdict(list)
    blocks = defaultdict(list)
    for b, f, fold in slots:
        grouped[b].append((f, fold))
        blocks[b, fold].append(f)
    require(set(grouped) == set(BENCHMARKS), 'all six benchmarks are required')
    full = {}
    # Alternate within benchmark/fold; alternate odd-block starts to balance margins.
    odd = 0
    for (b, fold), families in sorted(blocks.items()):
        ordered = sorted(families, key=lambda f: digest([seed, 'full-outer', b, fold, f]))
        for i, f in enumerate(ordered):
            full[b, f] = 'low' if (i + odd) % 2 == 0 else 'high'
        odd += len(ordered) % 2
    # Six benchmarks x two places leaves four proportional residual places per dose.
    totals = {b: len(v) for b, v in grouped.items()}
    quotas = {b: 2 for b in BENCHMARKS}
    ideal = {b: 4 * totals[b] / 129 for b in BENCHMARKS}
    for b in BENCHMARKS:
        quotas[b] += math.floor(ideal[b])
    remainder = 16 - sum(quotas.values())
    order = sorted(BENCHMARKS, key=lambda b: (-(ideal[b] % 1), digest([seed, 'quota', b])))
    for b in order[:remainder]:
        quotas[b] += 1
    fallback = {}
    for b in BENCHMARKS:
        n = quotas[b]
        require(len(grouped[b]) >= 2 * n, f'not enough families for fallback in {b}')
        chosen = sorted(grouped[b], key=lambda x: digest([seed, 'fallback-sample', b, x[0]]))[:2*n]
        split = sorted(chosen, key=lambda x: digest([seed, 'fallback-dose', b, x[0]]))
        for i, (f, _) in enumerate(split):
            fallback[b, f] = 'low' if i < n else 'high'
    records = [{'benchmark': b, 'family': f, 'fold': fold, 'full_outer': full[b, f],
                'fallback_outer': fallback.get((b, f)),
                'fallback_dose_probability': quotas[b] / totals[b],
                'fallback_selection_probability': 2 * quotas[b] / totals[b]}
               for b, f, fold in slots]
    require(sum(r['fallback_outer'] == 'low' for r in records) == 16, 'low fallback count')
    require(sum(r['fallback_outer'] == 'high' for r in records) == 16, 'high fallback count')
    return {'records': records, 'seed': seed, 'outcomes_used': False, 'quotas_per_dose': quotas,
            'fallback_sampling': 'uniform without replacement within benchmark; random equal split'}


def assignments(model, profile_sha, prefixes, nominations, allocation, seed, mode):
    require(mode in ('full', 'fallback'), 'unknown design')
    slots = {(r['benchmark'], r['family']): r for r in allocation['records']}
    lookup = {(r['fold'], r['class'], r['depth_third']): r for r in nominations}
    require(len(prefixes) == 129, 'model must retain every original slot')
    rows = []
    for prefix in prefixes:
        b, f = prefix['benchmark'], prefix['family']
        slot = slots[b, f]
        p = prefix.get('prefix')
        source = {k: p.get(k) for k in ('attempt_id', 'question', 'prefix_tokens', 'sentence_index')} if p else None
        base = {'model': model, 'profile_sha256': profile_sha, 'benchmark': b, 'family': f,
                'fold': slot['fold'], 'source': source,
                'seed': seed_for(seed, model, b, f, source), 'replicate': 0,
                'prefix_status': prefix['status']}
        durations = next((r for r in nominations if r['fold'] == slot['fold'] and r.get('horizon')), {})
        native = {**base, 'arm': 'native', 'dose': 'central', 'action': None,
                  'horizon': durations.get('horizon'), 'pulse_tokens': durations.get('pulse_tokens'),
                  'class': None, 'depth_third': None, 'nomination_status': 'nominated'}
        native['uid'] = digest(native)
        rows.append(native)
        outer = slot[mode + '_outer']
        for c in CLASSES:
            for depth in range(3):
                n = lookup.get((slot['fold'], c, depth))
                for dose in ['central'] + ([outer] if outer else []):
                    index = {'low': 0, 'central': 1, 'high': 2}[dose]
                    for arm in ARMS[1:]:
                        ids = n.get('control_sets' if arm.startswith('random') else 'expert_sets', [[], [], []])[index] if n else []
                        supported = bool(n and n['status'] == 'nominated' and ids)
                        action = {'layer': n['layer'], 'experts': ids,
                                  'direction': 'activate' if arm.endswith('_activate') else 'deactivate',
                                  'pulse_tokens': n['pulse_tokens'], 'start': 0} if supported else None
                        row = {**base, 'arm': arm, 'class': c, 'depth_third': depth, 'dose': dose,
                               'native_uid': native['uid'], 'action': action,
                               'horizon': n.get('horizon') if n else None,
                               'pulse_tokens': n.get('pulse_tokens') if n else None,
                               'nomination_status': 'nominated' if supported else 'insufficient_support'}
                        row['uid'] = digest(row)
                        rows.append(row)
    require(len({r['uid'] for r in rows}) == len(rows), 'duplicate assignment identity')
    return rows

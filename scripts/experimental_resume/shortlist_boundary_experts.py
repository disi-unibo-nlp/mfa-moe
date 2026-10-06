"""Exploratory discovery-only native expert shortlist at audited start boundaries.

The offline future-sentence LLM rating defines a contrast. The routing feature
is computed only from the last 64 native reasoning tokens already emitted at
the start boundary. Nothing produced here is an online detector or causal effect.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
BASE = R / 'steering-v1/runs/routing-control-v1/dense-discovery'
UNITS = BASE / 'UNITS.json'
FRAME = BASE / 'TRANSITION_AUDIT_CANDIDATES.json'
FIXTURE = BASE / 'TRANSITION_AUDIT_FIXTURES.json'
RATINGS = BASE / 'ratings-f5b2e28c-74c16a5d'
ROUTES = BASE / 'route-profiles-84a85c92-40c3d3b0'
ATTEMPTS = R / 'v3_analysis/results-r2/qwen36/A/attempts.parquet'
V3AN = R / 'v3_analysis'
OUT = R / 'steering-v1/runs/routing-control-v1/expert-shortlist-v2/RESULT.json'
WIDTH = 64
TRANSITIONS = ('candidate_to_verify', 'approach_to_commit', 'failed_check_to_revise')
EXPECTED_UNITS = '84a85c92b595a7829446b6a53b376e1126df3b6c36eabaddb3b4e884eb52fa72'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def plain(value):
    """Convert NumPy aggregate scalars before creating a canonical JSON seal."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [plain(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed input seal: ' + str(path))
    return value


def rating_rows(frame, fixture, binding):
    if binding['fixtures_sha256'] != frame['sha256']:
        raise ValueError('rating batch bound to another fixture')
    if [r['uid'] for r in frame['records']] != [r['uid'] for r in fixture['records']]:
        raise ValueError('labeled fixture differs from reader-visible frame')
    rows = []
    for start in range(0, len(fixture['records']), binding['batch_size']):
        part = sealed(RATINGS / 'batches' / f'{start:06d}.json')
        if part['binding_sha256'] != binding['sha256'] or part['start'] != start:
            raise ValueError('rating batch identity changed')
        rows.extend(part['records'])
    if len(rows) != len(fixture['records']):
        raise ValueError('incomplete rating set')
    joined, exclusions = [], Counter()
    for source, rated in zip(fixture['records'], rows, strict=True):
        if rated['uid'] != source['uid'] or len(rated['readers']) != 2:
            raise ValueError('rating and source UID differ')
        a, b = rated['readers']
        if not all(x['finish_reason'] == 'stop' and x['rating'] is not None for x in (a, b)):
            exclusions['unresolved'] += 1
            continue
        if not all(x['rating']['start'] for x in (a, b)):
            exclusions['start_not_both'] += 1
            continue
        if a['rating']['target'] != b['rating']['target']:
            exclusions['target_disagree'] += 1
            continue
        joined.append({'uid': source['uid'], 'transition': source['transition'],
                       'family': source['analysis_meta']['family'],
                       'attempt_id': source['analysis_meta']['attempt_id'],
                       'source_sentence_index': source['analysis_meta']['source_sentence_index'],
                       'source_class': source['analysis_meta']['source_class_audit'],
                       'prefix_tokens': source['analysis_meta']['prefix_tokens'],
                       'target': bool(a['rating']['target']),
                       'detector_fired': source['analysis_meta']['detector_fired']})
    return joined, dict(exclusions)


def profile(ids, weights, tokens):
    # P(expert is among native top eight) per token. Row sum is eight.
    if ids.shape != weights.shape or ids.ndim != 3 or ids.shape[0] != 40 or ids.shape[2] != 8:
        raise ValueError('unexpected native route tensor shape')
    if len(tokens) != WIDTH or np.any(np.diff(tokens) <= 0):
        raise ValueError('not exactly 64 ordered reasoning tokens')
    selected = ids[:, tokens, :].astype(np.int64)
    mass = weights[:, tokens, :].astype(np.float64)
    if selected.min() < 0 or selected.max() >= 256 or not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError('invalid native selected expert or weight')
    denom = mass.sum(axis=2, keepdims=True)
    if (denom <= 0).any():
        raise ValueError('zero native gate mass')
    mass /= denom
    frequency = np.stack([np.bincount(selected[layer].ravel(), minlength=256) / WIDTH
                          for layer in range(40)])
    gate = np.stack([np.bincount(selected[layer].ravel(), weights=mass[layer].ravel(), minlength=256) / WIDTH
                     for layer in range(40)])
    if not np.allclose(frequency.sum(1), 8) or not np.allclose(gate.sum(1), 1):
        raise ValueError('native top-k or gate normalization differs')
    return frequency, gate


def match_distance(positive, negative):
    # Same audited source class is mandatory; nearby depth and sentence length
    # improve comparability but cannot guarantee semantic exchangeability.
    return (abs(math.log1p(positive['prefix_tokens']) - math.log1p(negative['prefix_tokens']))
            + 0.25 * abs(math.log1p(positive['sentence_tokens']) -
                         math.log1p(negative['sentence_tokens'])))


def make_pairs(rows, *, within_only):
    positive = [r for r in rows if r['target']]
    negative = [r for r in rows if not r['target']]
    pairs = []
    for p in sorted(positive, key=lambda r: r['uid']):
        candidates = [n for n in negative if n['source_class'] == p['source_class']
                      and (not within_only or n['family'] == p['family'])]
        if not candidates:
            continue
        n = min(candidates, key=lambda r: (match_distance(p, r), r['uid']))
        pairs.append((p, n, match_distance(p, n)))
    return pairs


def equal_positive_family_delta(pairs, key):
    grouped = defaultdict(list)
    for p, n, _ in pairs:
        grouped[p['family']].append(p[key] - n[key])
    if not grouped:
        return None, {}
    by_family = {family: np.mean(values, axis=0) for family, values in grouped.items()}
    return np.mean(list(by_family.values()), axis=0), by_family


def rank_band(delta, rows, exposure):
    if delta is None:
        return None
    # At most four adjacent layers and two experts per layer, as in protocol.
    # The minimum native exposure here is a disclosed ranking filter, not a
    # prospective success threshold or a causal claim.
    eligible = exposure >= 0.01
    options = []
    for start in range(37):
        chosen = []
        for layer in range(start, start + 4):
            ranked = sorted((e for e in range(256) if eligible[layer, e] and delta[layer, e] > 0),
                            key=lambda e: (-delta[layer, e], e))[:2]
            chosen.extend((layer, e) for e in ranked)
        score = sum(float(delta[l, e]) for l, e in chosen) / 4
        options.append((score, start, chosen))
    score, start, chosen = max(options, key=lambda x: (x[0], -x[1]))
    if score <= 0 or not chosen:
        return None
    represented = []
    for layer, expert in chosen:
        pos_seen = sum(bool(r['target']) and bool(r['frequency'][layer, expert] > 0) for r in rows)
        neg_seen = sum(not r['target'] and bool(r['frequency'][layer, expert] > 0) for r in rows)
        represented.append({'layer': layer, 'expert': expert,
                            'risk_difference': float(delta[layer, expert]),
                            'native_global_exposure': float(exposure[layer, expert]),
                            'positive_windows_with_selection': int(pos_seen),
                            'negative_windows_with_selection': int(neg_seen)})
    return {'band_start': start, 'band_end_exclusive': start + 4,
            'band_score_sum_top_two_per_layer_divided_by_four': score,
            'expert_candidates': represented,
            'status': 'OBSERVATIONAL_SHORTLIST_NOT_FROZEN_CAUSAL_ACTION'}


def summary(rows, exposure):
    results = {}
    for transition in TRANSITIONS:
        selected = [r for r in rows if r['transition'] == transition]
        strict = make_pairs(selected, within_only=True)
        pooled = make_pairs(selected, within_only=False)
        delta, per_family = equal_positive_family_delta(pooled, 'frequency')
        gate_delta, _ = equal_positive_family_delta(pooled, 'gate')
        support = {
            'reader_agreed_valid_start_windows': len(selected),
            'target_positive_windows': sum(r['target'] for r in selected),
            'target_negative_windows': sum(not r['target'] for r in selected),
            'start_families': len({r['family'] for r in selected}),
            'positive_families': len({r['family'] for r in selected if r['target']}),
            'negative_families': len({r['family'] for r in selected if not r['target']}),
            'same_family_same_class_matched_positives': len(strict),
            'same_family_pair_families': len({p['family'] for p, _, _ in strict}),
            'cross_family_allowed_same_class_matched_positives': len(pooled),
            'cross_family_allowed_pair_positive_families': len(per_family),
            'cross_family_allowed_pair_negative_families': len({n['family'] for _, n, _ in pooled}),
            'cross_family_pair_count': sum(p['family'] != n['family'] for p, n, _ in pooled),
            'source_class_counts': dict(Counter(r['source_class'] for r in selected)),
            'pair_distance_median': float(np.median([d for _, _, d in pooled])) if pooled else None,
            'pair_distance_max': float(max((d for _, _, d in pooled), default=0)) if pooled else None,
        }
        result = {'support': support,
                  'exploratory_expert_ranking': rank_band(delta, selected, exposure),
                  'paired_mean_gate_mass_difference_for_selected': None}
        if result['exploratory_expert_ranking'] is not None:
            result['paired_mean_gate_mass_difference_for_selected'] = [
                float(gate_delta[x['layer'], x['expert']])
                for x in result['exploratory_expert_ranking']['expert_candidates']]
        if transition == 'failed_check_to_revise':
            # Two nominal positives in the LLM audit were manually judged
            # semantically false. Preserve counts, withhold an expert proposal.
            result['exploratory_expert_ranking'] = None
            result['paired_mean_gate_mass_difference_for_selected'] = None
            result['hold_reason'] = 'two reader-agreed starts fail manual semantic inspection; no valid contrast established'
        elif len(per_family) < 5:
            result['hold_reason'] = 'fewer than five independent positive families in matched contrast'
        results[transition] = result
    return results


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('native tensor shortlist requires CPU Slurm allocation')
    import pandas as pd
    sys.path.insert(0, str(V3AN))
    from v3an.extract import _load_tensors

    units, frame, fixture, binding = sealed(UNITS), sealed(FRAME), sealed(FIXTURE), sealed(RATINGS / 'BINDING.json')
    route_binding = sealed(ROUTES / 'BINDING.json')
    route_summary = sealed(ROUTES / 'SUMMARY.json')
    exposure_summary = sealed(ROUTES / 'NATIVE_ROUTE_SUMMARY.json')
    if (units['sha256'] != EXPECTED_UNITS or units['attempts'] != 48 or
        fixture['units_sha256'] != units['sha256'] or
        fixture['frame_counts'] != frame['frame_counts'] or
        fixture['families'] != 48 or route_binding['units_sha256'] != units['sha256'] or
        route_summary['binding_sha256'] != route_binding['sha256'] or
        exposure_summary['route_summary_sha256'] != route_summary['sha256']):
        raise ValueError('48-family discovery source or route bindings differ')
    source_units = {(r['family'], r['attempt_id'], r['sentence_index']): r for r in units['records']}
    rows, exclusions = rating_rows(frame, fixture, binding)
    table = pd.read_parquet(ATTEMPTS, columns=['attempt_id', 'tensor_path', 'completion_tokens'])
    by_attempt = {str(r['attempt_id']): r for r in table.to_dict('records')}
    grouped = defaultdict(list)
    for row in rows:
        unit = source_units[(row['family'], row['attempt_id'], row['source_sentence_index'])]
        if unit['token_end'] != row['prefix_tokens']:
            raise ValueError('audited boundary differs from frozen sentence token end')
        row['sentence_tokens'] = unit['token_end'] - unit['token_start']
        grouped[row['family']].append(row)
    if not set(grouped).issubset({r['family'] for r in units['records']}):
        raise ValueError('non-discovery family in audit')
    source_tensor_hashes = {}
    short_prefix = Counter()
    for family in sorted(grouped):
        original = sealed(ROUTES / f'{family}.json')
        attempt = grouped[family][0]['attempt_id']
        if original['attempt_id'] != attempt or original['family'] != family:
            raise ValueError('family-to-attempt route binding changed')
        target = Path(by_attempt[attempt]['tensor_path'])
        if sha(target) != original['tensor_sha256']:
            raise ValueError('native routed tensor changed')
        source_tensor_hashes[family] = original['tensor_sha256']
        ids, weights, _, reasoning = _load_tensors(target)
        if len(reasoning) == 0 or ids.shape[1] != int(by_attempt[attempt]['completion_tokens']):
            raise ValueError('native tensor token layout changed')
        for row in grouped[family]:
            if row['attempt_id'] != attempt or row['prefix_tokens'] > ids.shape[1]:
                raise ValueError('audited prefix outside native attempt')
            prior = reasoning[reasoning < row['prefix_tokens']]
            if len(prior) < WIDTH:
                short_prefix[row['transition']] += 1
                continue
            row['frequency'], row['gate'] = profile(ids, weights, prior[-WIDTH:])
        print(json.dumps({'families_profiled': len(source_tensor_hashes),
                          'rated_starts': sum(len(x) for x in grouped.values())}), flush=True)
    ranked_rows = [r for r in rows if 'frequency' in r]
    exposure = np.asarray(exposure_summary['native_expert_selection_exposure'], dtype=float)
    if exposure.shape != (40, 256) or not np.isfinite(exposure).all():
        raise ValueError('global native exposure shape changed')
    results = summary(ranked_rows, exposure)
    body = plain({'schema': 'exploratory-boundary-expert-shortlist-v2',
            'scope': '48 frozen Qwen3.6 discovery families; LLM-audited starts and targets; native observational routing only',
            'job_id': os.environ['SLURM_JOB_ID'],
            'units_sha256': units['sha256'], 'frame_sha256': frame['sha256'],
            'fixture_sha256': fixture['sha256'],
            'rating_binding_sha256': binding['sha256'],
            'route_summary_sha256': route_summary['sha256'],
            'exposure_summary_sha256': exposure_summary['sha256'],
            'native_tensor_family_sha256': source_tensor_hashes,
            'driver_sha256': sha(__file__),
            'rating_exclusions': exclusions,
            'under_64_reasoning_token_prefix_by_transition': dict(short_prefix),
            'boundary_definition': 'last exact 64 native reasoning tokens with token index < completed source sentence token_end',
            'native_top_k': 8, 'layers': 40, 'routed_experts_per_layer': 256,
            'matching': 'target-positive start to nearest target-negative start in the same dense source class, by log prefix depth plus quarter log source-sentence token length; same-family support also reported separately; controls may be reused',
            'aggregation': 'each target-positive family equal weight after averaging its matched pairs; exposure floor 0.01 for shortlist rank only',
            'results': results,
            'interpretation': 'future sentence is an offline contrast only, never an online input; ranking is neither causal importance nor a qualified action'})
    value = {**body, 'sha256': digest(body)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing output differs; use a new version')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'output': str(OUT), 'results': {k: v['support'] for k, v in results.items()}}), flush=True)


if __name__ == '__main__':
    main()

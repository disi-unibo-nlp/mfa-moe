"""Discovery-only native expert source contrast for transitions into Verify.

This uses saved seven-class LLM labels as an *offline* outcome proxy, never as
online controller input. Verify class is broader than substantive verification.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_transition_audit import label_rows
from shortlist_boundary_experts import (
    digest, sha, sealed, plain, equal_positive_family_delta, make_pairs,
)

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
BASE = R / 'steering-v1/runs/routing-control-v1/dense-discovery'
UNITS = BASE / 'UNITS.json'
ROUTES = BASE / 'route-profiles-84a85c92-40c3d3b0'
OUT = R / 'steering-v1/runs/routing-control-v1/dense-verify-source-v2/RESULT.json'
EXPECTED_UNITS = '84a85c92b595a7829446b6a53b376e1126df3b6c36eabaddb3b4e884eb52fa72'
CLASSES = {'Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor'}


def proposal(delta, per_family, exposure):
    if delta is None:
        return None
    options = []
    for length in range(1, 5):
        for first in range(40 - length + 1):
            selected = []
            for layer in range(first, first + length):
                eligible = sorted((expert for expert in range(256)
                                   if delta[layer, expert] > 0 and exposure[layer, expert] >= 0.002),
                                  key=lambda expert: (-delta[layer, expert], expert))[:2]
                if not eligible:
                    break
                selected.extend((layer, expert) for expert in eligible)
            if {layer for layer, _ in selected} == set(range(first, first + length)):
                score = sum(float(delta[layer, expert]) for layer, expert in selected) / length
                options.append((score, -length, -first, selected))
    if not options:
        return None
    score, neg_length, neg_first, selected = max(options)
    candidates = []
    for layer, expert in selected:
        values = [float(v[layer, expert]) for v in per_family.values()]
        population = float(exposure[layer, expert])
        pool = [other for other in range(256) if other != expert and
                (layer, other) not in selected and
                abs(float(exposure[layer, other]) - population) <= 0.1 * population]
        loo = [(sum(values) - value) / (len(values) - 1) for value in values] if len(values) > 1 else []
        candidates.append({'layer': layer, 'expert': expert,
                           'paired_selection_risk_difference': float(delta[layer, expert]),
                           'native_global_exposure': population,
                           'positive_family_difference_count': sum(x > 0 for x in values),
                           'negative_family_difference_count': sum(x < 0 for x in values),
                           'independent_positive_families': len(values),
                           'leave_one_family_out_min': min(loo) if loo else None,
                           'leave_one_family_out_max': max(loo) if loo else None,
                           'exposure_matched_random_candidates_plusminus_10_percent': len(pool),
                           'first_four_matched_random_experts': sorted(pool, key=lambda e: (abs(float(exposure[layer, e]) - population), e))[:4]})
    return {'band_start': -neg_first, 'band_layers': -neg_length,
            'band_score_mean_positive_risk_difference': score,
            'experts': candidates,
            'action_shape_valid': len(selected) <= 8 and
                                  len({l for l, _ in selected}) == -neg_length and
                                  all(1 <= sum(l == layer for l, _ in selected) <= 2
                                      for layer in range(-neg_first, -neg_first - neg_length)),
            'status': 'DENSE_VERIFY_CLASS_ACTION_SOURCE_ONLY_NOT_SEMANTIC_ACTION'}


def crossfit(pairs, exposure):
    folds = defaultdict(list)
    for p, n, distance in pairs:
        fold = int(hashlib.sha256(('dense-verify-fold-v1|' + p['family']).encode()).hexdigest()[:8], 16) % 4
        folds[fold].append((p, n, distance))
    results = []
    for fold in range(4):
        held = folds[fold]
        train = [pair for other in range(4) if other != fold for pair in folds[other]]
        train_delta, train_families = equal_positive_family_delta(train, 'frequency')
        held_delta, held_families = equal_positive_family_delta(held, 'frequency')
        selected = proposal(train_delta, train_families, exposure)
        effect = None
        if selected is not None and held_delta is not None:
            values = [float(held_delta[x['layer'], x['expert']]) for x in selected['experts']]
            effect = sum(values) / selected['band_layers']
        results.append({'fold': fold, 'training_positive_families': len(train_families),
                        'held_out_positive_families': len(held_families),
                        'training_source_action_shape_valid': selected is not None and selected['action_shape_valid'],
                        'held_out_mean_selected_expert_risk_difference': effect,
                        'held_out_positive_direction': effect is not None and effect > 0})
    return results


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('dense Verify source shortlist requires CPU Slurm')
    units = sealed(UNITS)
    route_binding, route_summary = sealed(ROUTES / 'BINDING.json'), sealed(ROUTES / 'SUMMARY.json')
    exposure_summary = sealed(ROUTES / 'NATIVE_ROUTE_SUMMARY.json')
    if (units['sha256'] != EXPECTED_UNITS or units['attempts'] != 48 or
        route_binding['units_sha256'] != units['sha256'] or
        route_summary['binding_sha256'] != route_binding['sha256'] or
        exposure_summary['route_summary_sha256'] != route_summary['sha256']):
        raise ValueError('dense discovery source bindings differ')
    labels, label_summary = label_rows(units)
    rows_by_family = defaultdict(list)
    for unit, label in zip(units['records'], labels, strict=True):
        if (unit['family'], unit['attempt_id'], unit['sentence_index']) != (
            label['family'], label['attempt_id'], label['sentence_index']):
            raise ValueError('unit-label identity mismatch')
        rows_by_family[unit['family']].append((unit, label))
    if len(rows_by_family) != 48:
        raise ValueError('fewer than 48 discovery families')
    records = []
    inspected = Counter()
    for family in sorted(rows_by_family):
        receipt = sealed(ROUTES / f'{family}.json')
        npz_path = ROUTES / f'{family}.npz'
        if receipt['binding_sha256'] != route_binding['sha256'] or sha(npz_path) != receipt['npz_sha256']:
            raise ValueError('native sentence route file changed')
        with np.load(npz_path, allow_pickle=False) as npz:
            indices = npz['sentence_index'].tolist()
            freq = npz['frequency']
            gate = npz['gate']
        if len(indices) != len(rows_by_family[family]) or freq.shape != (len(indices), 40, 256):
            raise ValueError('sentence route profile dimensions differ')
        profile = {int(index): (freq[pos] * 8, gate[pos]) for pos, index in enumerate(indices)}
        ordered = sorted(rows_by_family[family], key=lambda pair: pair[0]['sentence_index'])
        for (source, source_label), (later, later_label) in zip(ordered, ordered[1:]):
            if (later['sentence_index'] != source['sentence_index'] + 1 or
                later['segment'] != source['segment']):
                continue
            if (source_label['label'] not in CLASSES or later_label['label'] not in CLASSES or
                source_label['finish_reason'] != 'stop' or later_label['finish_reason'] != 'stop'):
                inspected['unparsed_adjacent'] += 1
                continue
            inspected['parsed_adjacent'] += 1
            if source_label['label'] == 'Verify':
                continue  # Study entry into Verify, not persistence within it.
            inspected['nonverify_source'] += 1
            records.append({'uid': digest(['dense-verify-source-v1', family, source['sentence_index'], later['sentence_index']]),
                            'family': family, 'source_class': source_label['label'],
                            'prefix_tokens': int(source['token_end']),
                            'sentence_tokens': int(source['token_end'] - source['token_start']),
                            'target': later_label['label'] == 'Verify',
                            'frequency': profile[source['sentence_index']][0],
                            'gate': profile[source['sentence_index']][1]})
    pairs = make_pairs(records, within_only=True)
    delta, by_family = equal_positive_family_delta(pairs, 'frequency')
    exposure = np.asarray(exposure_summary['native_expert_selection_exposure'], dtype=float)
    if exposure.shape != (40, 256):
        raise ValueError('global native route exposure dimensions differ')
    selected = proposal(delta, by_family, exposure)
    folds = crossfit(pairs, exposure)
    support = {'discovery_families': len(rows_by_family),
               'dense_verify_labels_total': sum(label['label'] == 'Verify' and label['finish_reason'] == 'stop' for label in labels),
               'contiguous_parsed_sentence_pairs': inspected['parsed_adjacent'],
               'nonverify_source_pairs': inspected['nonverify_source'],
               'verify_entry_positive_pairs': sum(r['target'] for r in records),
               'nonverify_entry_negative_pairs': sum(not r['target'] for r in records),
               'within_family_same_source_class_matched_positives': len(pairs),
               'within_family_matched_positive_families': len(by_family),
               'matched_negative_families': len({n['family'] for _, n, _ in pairs}),
               'source_class_counts': dict(Counter(r['source_class'] for r in records)),
               'median_match_distance': float(np.median([d for _, _, d in pairs])) if pairs else None,
               'max_match_distance': float(max((d for _, _, d in pairs), default=0)) if pairs else None,
               'unparsed_adjacent_pairs': inspected['unparsed_adjacent']}
    body = plain({'schema': 'dense-verify-native-source-shortlist-v2',
                  'job_id': os.environ['SLURM_JOB_ID'],
                  'population': '48 frozen discovery families, native traces, direct LLM seven-class labels',
                  'units_sha256': units['sha256'], 'label_summary_sha256': label_summary['sha256'],
                  'route_summary_sha256': route_summary['sha256'],
                  'exposure_summary_sha256': exposure_summary['sha256'],
                  'driver_sha256': sha(__file__),
                  'route_feature': 'native expert frequency in entire completed source sentence immediately before next sentence; not exact last-64 boundary',
                  'offline_target': 'next contiguous sentence has direct-LLM class Verify; this does not establish substantive verification',
                  'matching': 'within same discovery family and same source class, nearest log prefix depth plus quarter log source sentence token length, with control reuse',
                  'aggregation': 'average matched differences within positive family, then equal positive-family mean',
                  'support': support, 'proposal': selected, 'four_fold_family_crossfit': folds,
                  'interpretation': 'native observational class source only; same-prefix action test required for semantic control'})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing dense source result differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'output': str(OUT), 'support': support,
                      'proposal_shape_valid': selected is not None and selected['action_shape_valid']}), flush=True)


if __name__ == '__main__':
    main()

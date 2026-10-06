"""Discovery-only one-expert approach contrast with four feasible random peers.

The seven-class labels identify broad next-sentence behavior, not an identified
approach or a semantic commitment. They never enter an online controller.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_transition_audit import label_rows
from shortlist_boundary_experts import (ATTEMPTS, R, ROUTES, UNITS, V3AN, WIDTH,
                                         digest, equal_positive_family_delta,
                                         make_pairs, plain, profile, sealed, sha)
from shortlist_dense_verify_source import CLASSES

OUT = R / 'steering-v1/runs/routing-control-v1/approach-boundary-supported-v2/RESULT.json'
TARGET_CLASSES = {'Plan', 'Implement'}


def proposal(delta, per_family, exposure):
    """Prospective highest-risk-difference single expert with four ±10% peers."""
    if delta is None:
        return None
    options = []
    for layer in range(40):
        for expert in range(256):
            native = float(exposure[layer, expert])
            if native < .002 or delta[layer, expert] <= 0:
                continue
            peers = [other for other in range(256) if other != expert and
                     abs(float(exposure[layer, other]) - native) <= .1 * native]
            if len(peers) < 4:
                continue
            options.append((float(delta[layer, expert]), -layer, -expert, peers))
    if not options:
        return None
    score, neg_layer, neg_expert, peers = max(options)
    layer, expert = -neg_layer, -neg_expert
    family_values = [float(row[layer, expert]) for row in per_family.values()]
    loo = [(sum(family_values) - value) / (len(family_values) - 1)
           for value in family_values] if len(family_values) > 1 else []
    native = float(exposure[layer, expert])
    selected = {'layer': layer, 'expert': expert,
                'paired_selection_risk_difference': score,
                'native_global_exposure': native,
                'positive_family_difference_count': sum(v > 0 for v in family_values),
                'negative_family_difference_count': sum(v < 0 for v in family_values),
                'independent_positive_families': len(family_values),
                'leave_one_family_out_min': min(loo) if loo else None,
                'leave_one_family_out_max': max(loo) if loo else None,
                'exposure_matched_random_candidates_plusminus_10_percent': len(peers),
                'first_four_matched_random_experts': sorted(
                    peers, key=lambda e: (abs(float(exposure[layer, e]) - native), e))[:4]}
    return {'band_start': layer, 'band_layers': 1,
            'band_score_mean_positive_risk_difference': score,
            'experts': [selected], 'action_shape_valid': True,
            'status': 'DISCOVERY_ONLY_ONE_EXPERT_FOUR_CONTROL_FEASIBLE_CLASS_PROXY'}


def selected_effect(delta, selected):
    if delta is None or selected is None:
        return None
    return float(sum(delta[x['layer'], x['expert']] for x in selected['experts']) /
                 selected['band_layers'])


def crossfit(pairs, exposure):
    folds = defaultdict(list)
    for p, n, distance in pairs:
        index = int(hashlib.sha256(('approach-boundary-fold-v1|' + p['family']).encode()).hexdigest()[:8], 16) % 4
        folds[index].append((p, n, distance))
    result = []
    for fold in range(4):
        train = [pair for index in range(4) if index != fold for pair in folds[index]]
        held = folds[fold]
        delta, families = equal_positive_family_delta(train, 'frequency')
        selected = proposal(delta, families, exposure)
        held_delta, held_families = equal_positive_family_delta(held, 'frequency')
        effect = selected_effect(held_delta, selected)
        result.append({'fold': fold, 'training_positive_families': len(families),
                       'held_out_positive_families': len(held_families),
                       'held_out_selected_mean_risk_difference': effect,
                       'held_out_positive_direction': effect is not None and effect > 0})
    return result


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('last64 native route tensor analysis requires CPU Slurm')
    import pandas as pd
    sys.path.insert(0, str(V3AN))
    from v3an.extract import _load_tensors

    units = sealed(UNITS)
    binding = sealed(ROUTES / 'BINDING.json')
    summary = sealed(ROUTES / 'SUMMARY.json')
    exposure_source = sealed(ROUTES / 'NATIVE_ROUTE_SUMMARY.json')
    if (units['attempts'] != 48 or binding['units_sha256'] != units['sha256'] or
            summary['binding_sha256'] != binding['sha256'] or
            exposure_source['route_summary_sha256'] != summary['sha256']):
        raise ValueError('frozen discovery native source differs')
    labels, label_summary = label_rows(units)
    grouped = defaultdict(list)
    for unit, label in zip(units['records'], labels, strict=True):
        if (unit['family'], unit['attempt_id'], unit['sentence_index']) != (
                label['family'], label['attempt_id'], label['sentence_index']):
            raise ValueError('dense unit/label identity differs')
        grouped[unit['family']].append((unit, label))
    if len(grouped) != 48:
        raise ValueError('not all 48 frozen discovery families present')
    source_records = defaultdict(list)
    excluded = Counter()
    for family in sorted(grouped):
        sequence = sorted(grouped[family], key=lambda pair: pair[0]['sentence_index'])
        for (source, source_label), (later, later_label) in zip(sequence, sequence[1:]):
            if (later['sentence_index'] != source['sentence_index'] + 1 or
                    later['segment'] != source['segment']):
                continue
            if any(x['finish_reason'] != 'stop' or x['label'] not in CLASSES
                   for x in (source_label, later_label)):
                excluded['unparsed'] += 1
                continue
            if source_label['label'] != 'Explore':
                continue
            source_records[family].append({
                'uid': digest(['approach-boundary-supported-v2', family,
                               source['sentence_index'], later['sentence_index']]),
                'family': family, 'attempt_id': source['attempt_id'],
                'source_class': 'Explore',
                'prefix_tokens': int(source['token_end']),
                'sentence_tokens': int(source['token_end'] - source['token_start']),
                'target': later_label['label'] in TARGET_CLASSES,
                'later_class': later_label['label']})
    meta = [record for family in sorted(source_records) for record in source_records[family]]
    pair_meta = make_pairs(meta, within_only=True)
    needed = {r['uid'] for p, n, _ in pair_meta for r in (p, n)}
    table = pd.read_parquet(ATTEMPTS, columns=['attempt_id', 'tensor_path', 'completion_tokens'])
    attempts = {str(row['attempt_id']): row for row in table.to_dict('records')}
    tensors = {}
    records = []
    for family in sorted(source_records):
        receipt = sealed(ROUTES / f'{family}.json')
        attempt_id = receipt['attempt_id']
        if (receipt['family'] != family or receipt['binding_sha256'] != binding['sha256'] or
                attempt_id not in attempts):
            raise ValueError('family route receipt differs')
        tensor = Path(attempts[attempt_id]['tensor_path'])
        if sha(tensor) != receipt['tensor_sha256']:
            raise ValueError('native routing tensor changed')
        tensors[family] = receipt['tensor_sha256']
        ids, weights, _, reasoning = _load_tensors(tensor)
        if ids.shape[1] != int(attempts[attempt_id]['completion_tokens']):
            raise ValueError('native tensor token layout differs')
        for record in (r for r in source_records[family] if r['uid'] in needed):
            if record['attempt_id'] != attempt_id or record['prefix_tokens'] > ids.shape[1]:
                raise ValueError('native boundary exceeds tensor')
            prior = reasoning[reasoning < record['prefix_tokens']]
            if len(prior) < WIDTH:
                excluded['under_64_reasoning_tokens'] += 1
                continue
            record['frequency'], record['gate'] = profile(ids, weights, prior[-WIDTH:])
            records.append(record)
        print(json.dumps({'families_profiled': len(tensors), 'boundary_rows': len(records)}), flush=True)
    exposure = np.asarray(exposure_source['native_expert_selection_exposure'], dtype=float)
    if exposure.shape != (40, 256) or not np.isfinite(exposure).all():
        raise ValueError('native exposure differs')
    by_uid = {r['uid']: r for r in records}
    pairs = [(by_uid[p['uid']], by_uid[n['uid']], distance) for p, n, distance in pair_meta
             if p['uid'] in by_uid and n['uid'] in by_uid]
    delta, by_family = equal_positive_family_delta(pairs, 'frequency')
    selected = proposal(delta, by_family, exposure)
    folds = crossfit(pairs, exposure)
    random_supported = bool(selected) and all(
        x['exposure_matched_random_candidates_plusminus_10_percent'] >= 4
        for x in selected['experts'])
    # This is an exploratory causal-screen source. Semantic efficacy and later
    # validation remain separate, so fold signs are recorded rather than used
    # as a confirmatory acceptance test.
    action_supported = (selected is not None and selected['action_shape_valid'] and
                        len(by_family) >= 16 and random_supported and
                        selected['experts'][0]['leave_one_family_out_min'] > 0)
    actions = []
    if action_supported:
        experts = defaultdict(list)
        for item in selected['experts']:
            experts[item['layer']].append(item['expert'])
        shape = [[layer, sorted(ids)] for layer, ids in sorted(experts.items())]
        actions = [{'name': f'approach_proxy_bias{bias}',
                    'transition': 'approach_to_commit',
                    'experts': shape, 'bias': bias} for bias in (.5, 1.)]
    support = {'eligible_explore_source_rows': len(meta),
               'plan_or_implement_next_rows': sum(r['target'] for r in meta),
               'other_next_rows': sum(not r['target'] for r in meta),
               'next_class_counts': dict(Counter(r['later_class'] for r in meta)),
               'matched_positive_pairs': len(pairs),
               'matched_positive_families': len(by_family),
               'median_match_distance': float(np.median([d for _, _, d in pairs])) if pairs else None,
               'profiled_boundary_rows': len(records)}
    body = plain({'schema': 'approach-last64-single-expert-supported-shortlist-v2',
                  'job_id': os.environ['SLURM_JOB_ID'],
                  'population': '48 frozen discovery families, native Qwen3.6 traces, direct LLM seven-class labels',
                  'units_sha256': units['sha256'], 'label_summary_sha256': label_summary['sha256'],
                  'route_summary_sha256': summary['sha256'],
                  'native_exposure_sha256': exposure_source['sha256'],
                  'source_tensor_sha256_by_family': tensors,
                  'driver_sha256': sha(__file__), 'excluded': dict(excluded),
                  'boundary_feature': 'last 64 native reasoning tokens strictly before source sentence token_end',
                  'prospective_selection_rule': 'highest positive equal-family paired risk difference among one-expert actions with native exposure >=0.002 and at least four other experts within ±10% exposure in the same layer; require >=16 matched families and positive leave-one-family-out minimum for exploratory causal proposal',
                  'offline_proxy': 'source Explore class then next contiguous Plan/Implement class; identified approach and substantive commitment not established',
                  'matching': 'same frozen family and Explore source; nearest log prefix depth and source sentence length; control reuse',
                  'aggregation': 'paired differences averaged within each positive family, then equal family weight',
                  'support': support, 'proposal': selected,
                  'four_fold_family_crossfit': folds,
                  'random_exposure_supported': random_supported,
                  'hold_for_causal_test': not action_supported,
                  'proposed_actions_if_supported': actions,
                  'interpretation': 'observational class-proxy source only, never semantic or causal efficacy; full-prefix reader eligibility and same-prefix intervention remain required'})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('existing approach boundary result differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'support': support, 'hold_for_causal_test': not action_supported,
                      'action_count': len(actions)}))


if __name__ == '__main__':
    main()

"""Discovery-only exact-boundary Verify proxy and persistence expert contrasts.

Uses native last-64-token routing and offline seven-class labels. The labels
are broad behavior proxies, not substantive verification or online inputs.
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
from shortlist_dense_verify_source import CLASSES, proposal

OUT = R / 'steering-v1/runs/routing-control-v1/boundary-proxy-negative-v1/RESULT.json'
SEMANTIC_SHORTLIST = R / 'steering-v1/runs/routing-control-v1/expert-shortlist-v2/RESULT.json'
EXCLUDE_TARGET = ((28, 9), (28, 189))


def selected_effect(delta, selected):
    if delta is None or selected is None:
        return None
    return float(sum(delta[x['layer'], x['expert']] for x in selected['experts']) /
                 selected['band_layers'])


def choose(pairs, exposure, *, persistence):
    delta, by_family = equal_positive_family_delta(pairs, 'frequency')
    if delta is None:
        return None, by_family
    oriented = -delta if persistence else delta.copy()
    oriented_by_family = {family: -value if persistence else value
                          for family, value in by_family.items()}
    oriented = oriented.copy()
    for layer, expert in EXCLUDE_TARGET:
        oriented[layer, expert] = -1e6
    selected = proposal(oriented, oriented_by_family, exposure)
    return selected, by_family


def crossfit(pairs, exposure, *, persistence):
    folds = defaultdict(list)
    for p, n, dist in pairs:
        fold = int(hashlib.sha256(('boundary-proxy-fold-v1|' + p['family']).encode()).hexdigest()[:8], 16) % 4
        folds[fold].append((p, n, dist))
    result = []
    for fold in range(4):
        train = [pair for index in range(4) if index != fold for pair in folds[index]]
        held = folds[fold]
        selected, train_family = choose(train, exposure, persistence=persistence)
        held_delta, held_family = equal_positive_family_delta(held, 'frequency')
        if held_delta is not None and persistence:
            held_delta = -held_delta
        effect = selected_effect(held_delta, selected)
        result.append({'fold': fold, 'training_positive_families': len(train_family),
                       'held_out_positive_families': len(held_family),
                       'held_out_selected_mean_oriented_difference': effect,
                       'held_out_positive_direction': effect is not None and effect > 0})
    return result


def characterize(pairs, exposure, *, persistence, eligible_rows):
    selected, by_family = choose(pairs, exposure, persistence=persistence)
    folds = crossfit(pairs, exposure, persistence=persistence)
    kinds = Counter(n['negative_kind'] for _, n, _ in pairs)
    positive_families = {p['family'] for p, _, _ in pairs}
    return {'support': {'eligible_rows_before_boundary_availability': len(eligible_rows),
                        'verify_entry_positive_rows_before_boundary_availability': sum(r['target'] for r in eligible_rows),
                        'comparator_negative_rows_before_boundary_availability': sum(not r['target'] for r in eligible_rows),
                        'same_family_source_class_matched_positives': len(pairs),
                        'matched_positive_families': len(positive_families),
                        'matched_negative_families': len({n['family'] for _, n, _ in pairs}),
                        'matched_negative_kinds': dict(kinds),
                        'median_match_distance': float(np.median([distance for _, _, distance in pairs])) if pairs else None},
            'proposal': selected, 'four_fold_family_crossfit': folds,
            'hold_for_causal_test': selected is None or len(positive_families) < 24 or
            not all(x['held_out_positive_direction'] for x in folds)}


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('native tensor contrast requires CPU Slurm')
    import pandas as pd
    sys.path.insert(0, str(V3AN))
    from v3an.extract import _load_tensors

    units = sealed(UNITS)
    sem = sealed(SEMANTIC_SHORTLIST)
    binding, summary = sealed(ROUTES / 'BINDING.json'), sealed(ROUTES / 'SUMMARY.json')
    exposure_source = sealed(ROUTES / 'NATIVE_ROUTE_SUMMARY.json')
    if (sem['units_sha256'] != units['sha256'] or
        binding['units_sha256'] != units['sha256'] or
        summary['binding_sha256'] != binding['sha256'] or
        exposure_source['route_summary_sha256'] != summary['sha256'] or
        units['attempts'] != 48):
        raise ValueError('discovery native source/semantic hold binding differs')
    labels, label_summary = label_rows(units)
    grouped = defaultdict(list)
    for unit, label in zip(units['records'], labels, strict=True):
        if (unit['family'], unit['attempt_id'], unit['sentence_index']) != (
            label['family'], label['attempt_id'], label['sentence_index']):
            raise ValueError('dense unit and class label identity differs')
        grouped[unit['family']].append((unit, label))
    if len(grouped) != 48:
        raise ValueError('not all frozen discovery families are present')
    source_records = defaultdict(list)
    excluded = Counter()
    for family in sorted(grouped):
        sequence = sorted(grouped[family], key=lambda pair: pair[0]['sentence_index'])
        for index in range(len(sequence) - 1):
            source, source_label = sequence[index]
            later, later_label = sequence[index + 1]
            if later['sentence_index'] != source['sentence_index'] + 1 or later['segment'] != source['segment']:
                continue
            if any(x['finish_reason'] != 'stop' or x['label'] not in CLASSES
                   for x in (source_label, later_label)):
                excluded['unparsed_class'] += 1
                continue
            if source_label['label'] == 'Verify':
                continue
            earlier_class = None
            if index and sequence[index - 1][0]['sentence_index'] + 1 == source['sentence_index'] and (
                sequence[index - 1][0]['segment'] == source['segment'] and
                sequence[index - 1][1]['finish_reason'] == 'stop'):
                earlier_class = sequence[index - 1][1]['label']
            later_class = later_label['label']
            persistence = later_class == source_label['label']
            loop = (not persistence and earlier_class is not None and
                    earlier_class != source_label['label'] and later_class == earlier_class)
            source_records[family].append({'uid': digest(['boundary-proxy-negative-v1', family,
                                                           source['sentence_index'], later['sentence_index']]),
                                           'family': family, 'attempt_id': source['attempt_id'],
                                           'source_class': source_label['label'],
                                           'prefix_tokens': int(source['token_end']),
                                           'sentence_tokens': int(source['token_end'] - source['token_start']),
                                           'target': later_class == 'Verify',
                                           'persistence_or_loop': persistence or loop,
                                           'negative_kind': 'persistence' if persistence else 'two_step_loop' if loop else 'other'})
    all_meta = [record for group in source_records.values() for record in group]
    persistence_meta = [r for r in all_meta if r['target'] or r['persistence_or_loop']]
    verify_pair_meta = make_pairs(all_meta, within_only=True)
    persistence_pair_meta = make_pairs(persistence_meta, within_only=True)
    needed = {r['uid'] for collection in (verify_pair_meta, persistence_pair_meta)
              for p, n, _ in collection for r in (p, n)}
    table = pd.read_parquet(ATTEMPTS, columns=['attempt_id', 'tensor_path', 'completion_tokens'])
    attempts = {str(row['attempt_id']): row for row in table.to_dict('records')}
    tensor_hashes = {}
    records = []
    for family in sorted(source_records):
        receipt = sealed(ROUTES / f'{family}.json')
        attempt_id = receipt['attempt_id']
        if attempt_id not in attempts or receipt['family'] != family or receipt['binding_sha256'] != binding['sha256']:
            raise ValueError('family route receipt differs')
        tensor = Path(attempts[attempt_id]['tensor_path'])
        if sha(tensor) != receipt['tensor_sha256']:
            raise ValueError('native route tensor differs')
        tensor_hashes[family] = receipt['tensor_sha256']
        ids, weights, _, reasoning = _load_tensors(tensor)
        if ids.shape[1] != int(attempts[attempt_id]['completion_tokens']):
            raise ValueError('native route token layout differs')
        for record in (r for r in source_records[family] if r['uid'] in needed):
            if record['attempt_id'] != attempt_id or record['prefix_tokens'] > ids.shape[1]:
                raise ValueError('native source prefix exceeds tensor')
            prior = reasoning[reasoning < record['prefix_tokens']]
            if len(prior) < WIDTH:
                excluded['under_64_reasoning_tokens'] += 1
                continue
            record['frequency'], record['gate'] = profile(ids, weights, prior[-WIDTH:])
            records.append(record)
        print(json.dumps({'families_profiled': len(tensor_hashes), 'boundary_rows': len(records)}), flush=True)
    exposure = np.asarray(exposure_source['native_expert_selection_exposure'], dtype=float)
    if exposure.shape != (40, 256) or not np.isfinite(exposure).all():
        raise ValueError('native exposure differs')
    by_uid = {r['uid']: r for r in records}
    def with_profiles(pairs):
        return [(by_uid[p['uid']], by_uid[n['uid']], distance) for p, n, distance in pairs
                if p['uid'] in by_uid and n['uid'] in by_uid]
    verify = characterize(with_profiles(verify_pair_meta), exposure, persistence=False,
                          eligible_rows=all_meta)
    negative = characterize(with_profiles(persistence_pair_meta), exposure, persistence=True,
                            eligible_rows=persistence_meta)
    semantic = sem['results']['candidate_to_verify']['support']
    if semantic['target_positive_windows'] != 6 or semantic['same_family_same_class_matched_positives'] != 1:
        raise ValueError('semantic exact-boundary six-positive hold changed')
    body = plain({'schema': 'boundary-proxy-and-persistence-shortlist-v1',
                  'job_id': os.environ['SLURM_JOB_ID'],
                  'population': '48 frozen discovery families, native Qwen3.6 routing, offline seven-class labels',
                  'units_sha256': units['sha256'], 'label_summary_sha256': label_summary['sha256'],
                  'route_summary_sha256': summary['sha256'],
                  'native_exposure_sha256': exposure_source['sha256'],
                  'semantic_shortlist_sha256': sem['sha256'],
                  'source_tensor_sha256_by_family': tensor_hashes,
                  'driver_sha256': sha(__file__), 'excluded': dict(excluded),
                  'candidate_windows_before_boundary_feature': len(all_meta),
                  'boundary_feature_needed_windows': len(needed),
                  'boundary_feature_available_windows': len(records),
                  'boundary': 'exact last 64 native reasoning tokens strictly before source sentence token_end',
                  'matching': 'same frozen family and direct LLM source class; nearest log prefix depth and sentence length; control reuse',
                  'aggregation': 'average paired difference within Verify-positive family, then equal family weight',
                  'semantic_target_support_hold': {'positive_windows': 6, 'same_family_pairs': 1,
                                                   'reason': 'no stable semantic expert ranking can be inferred'},
                  'verify_entry_proxy': verify,
                  'persistence_or_loop_negative_control': negative,
                  'interpretation': 'both expert proposals are native observational class-proxy sources, not causal or semantic effects; future labels never enter online controller'})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if sealed(OUT) != value:
            raise ValueError('output exists with changed seal')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'sha256': value['sha256'],
                      'verify_support': verify['support'], 'negative_support': negative['support'],
                      'verify_hold': verify['hold_for_causal_test'],
                      'negative_hold': negative['hold_for_causal_test']}))


if __name__ == '__main__':
    main()

"""Independently audit the completed six-benchmark run using NumPy/SciPy and saved CSVs.

Read-only: prints JSON. Verifies input/annotation/replay fingerprints, inclusion,
all saved binary and metric-pair coefficients in all 20 scopes, and the reported
expert maxima. Does not rerun bootstraps, judge answers, or recompute activations.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/correlation_pipeline/reasoning-vllm-v1'
SOURCE = RUN / 'analysis/unsloth--Qwen3.5-35B-A3B'


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def trace_digest(t):
    return digest({'prompt': t['prompt'], 'messages': t.get('generation_messages'),
                   'system_prompt': t.get('system_prompt'), 'cot_text': t['cot_text'],
                   **({'token_replay': t['metadata']['token_replay']} if 'token_replay' in t['metadata'] else {})})


def key(row):
    return row['dataset'], row['problem_id']


def check_coefficients(rows, records, *, rank=False):
    arrays = {}
    scopes = {}
    error = 0.0
    for r in records:
        scope = r['scope']
        if scope not in scopes:
            scopes[scope] = np.array([scope == 'all' or t['dataset'] == scope for t in rows])
        names = [r['feature_x'], r['feature_y']] if 'feature_x' in r else [r['feature'], r['target']]
        for name in names:
            if name not in arrays:
                arrays[name] = np.array([float(t[name]) if t[name] else np.nan for t in rows])
        x, y = (arrays[name] for name in names)
        mask = scopes[scope] & np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        assert len(x) == r['n_traces'], r
        if rank:
            x, y = rankdata(x), rankdata(y)
        assert len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0
        observed = float(np.corrcoef(x, y)[0, 1])
        delta = abs(observed - r['spearman_rho' if rank else 'point_biserial_r'])
        assert delta < 1e-9, (delta, r)
        error = max(error, delta)
    return {'coefficients_checked': len(records), 'maximum_absolute_error': error}


def check_repeated(rows, analysis):
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['source_problem_id']), []).append(row)
    max_error = 0.0
    checked = 0
    for record in analysis['problem_level']:
        if record['problem_feature'] != 'feature_mean':
            continue
        x, y = [], []
        for (dataset, _), attempts in groups.items():
            if dataset != record['dataset']:
                continue
            assert len(attempts) == len({r['sample_id'] for r in attempts}) == 32
            values = [float(r[record['feature']]) for r in attempts if r[record['feature']]]
            values = [v for v in values if np.isfinite(v)]
            if values:
                x.append(np.mean(values))
                y.append(np.mean([float(r['is_correct']) for r in attempts]))
        assert len(x) == record['n_problems']
        delta = abs(float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]) - record['spearman_rho'])
        assert delta < 1e-9, record
        max_error = max(max_error, delta); checked += 1
    for record in analysis['within_problem']:
        differences = []
        for (dataset, _), attempts in groups.items():
            if dataset != record['dataset']:
                continue
            yes, no = [], []
            for row in attempts:
                value = float(row[record['feature']]) if row[record['feature']] else np.nan
                if np.isfinite(value):
                    (yes if float(row['is_correct']) else no).append(value)
            if yes and no:
                differences.append(np.mean(yes) - np.mean(no))
        assert len(differences) == record['n_mixed_outcome_problems']
        delta = abs(float(np.mean(differences)) - record['mean_correct_minus_incorrect'])
        assert delta < 1e-9, record
        max_error = max(max_error, delta); checked += 1
    return {'coefficients_checked': checked, 'maximum_absolute_error': max_error}


def main():
    b = json.loads((SOURCE / 'correlations.json').read_text())
    f = json.loads((RUN / 'forward/unsloth--Qwen3.5-35B-A3B/summary.json').read_text())
    assert b['status'] == f['status'] == b['reasoning_view_analysis']['status'] == 'complete'
    rows = read_csv(SOURCE / 'trace_features.csv')
    lookup = {key(r): r for r in rows}
    assert len(lookup) == len(rows) == b['n_traces'] == 4647
    assert len({(r['dataset'], r['source_problem_id']) for r in rows}) == b['n_problem_clusters'] == 1547
    assert Counter(r['dataset'] for r in rows) == b['datasets']
    assert sum(b['excluded_truncated_generations'].values()) == 0
    assert sum(g['eligible_repeated_analysis_groups'] for g in b['repeated_problem_analysis']['group_completeness']) == 100
    outcomes = Counter()
    excluded = Counter()
    matched = Counter()
    units = Counter()
    retained_units = Counter()
    for d in f['datasets']:
        name = d['dataset']
        annotations = {}
        path = RUN / f'annotations/Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/{name}/annotations.jsonl'
        for line in path.read_text().splitlines():
            a = json.loads(line)
            assert a['problem_id'] not in annotations
            annotations[a['problem_id']] = a
        source_fingerprints = {}
        with (ROOT / d['input']).open() as source:
            for line in source:
                t = json.loads(line)
                a = annotations.get(t['problem_id'])
                td = trace_digest(t)
                if a is not None:
                    assert a['status'] == 'complete' and a['trace_sha256'] == td
                    assert a['classifier'] == b['reasoning_view_analysis']['contract']['classifier']
                    selection = a['sentence_selection']
                    if selection is not None:
                        assert [u['index'] for u in a['units']] == selection['indices']
                    for u in a['units']:
                        assert t['cot_text'][u['start']:u['end']] == u['text']
                    matched['annotations'] += 1
                    units.update(u['label'] for u in a['units'])
                    if key(t) in lookup:
                        retained_units.update(u['label'] for u in a['units'])
                assert key(t) not in source_fingerprints
                source_fingerprints[key(t)] = td
                matched['generation'] += 1
        assert len(source_fingerprints) == d['traces']
        assert set(annotations).issubset({k[1] for k in source_fingerprints})
        with (ROOT / d['output']).open() as source:
            seen = set()
            for line in source:
                t = json.loads(line)
                k = key(t)
                assert k not in seen
                seen.add(k)
                td = trace_digest(t)
                assert source_fingerprints[k] == td
                assert t['model_logs']['layer_indices'] == f['layer_indices'] == [27, 34, 35, 36, 38, 39]
                meta = t['metadata']
                assert meta['correlation_views']['trace_sha256'] == td
                assert meta['correlation_views']['annotation_sha256'] == (digest(annotations[t['problem_id']]) if t['problem_id'] in annotations else None)
                assert meta['correlation_views']['population_version'] == 2
                assert meta['generation_config']['max_tokens'] == 32768
                limited = meta['finish_reason'] == 'length' or meta['usage']['completion_tokens'] >= 32768
                assert k in lookup
                assert int(float(lookup[k]['generation_hit_token_limit'])) == int(limited)
                excluded[name] += int(limited)
                matched['forward'] += 1
                row = lookup[k]
                assert row['generation_sha256'] == hashlib.sha256(t['cot_text'].encode()).hexdigest()
                assert float(row['is_correct']) == int(bool(t['is_correct']))
                outcomes[name] += int(bool(t['is_correct']))
                compact = meta['correlation_features']
                assert float(row['token_count']) == compact['token_count']
                for feature, value in compact['values'].items():
                    if feature in row and value is not None and row[feature]:
                        assert np.isclose(float(row[feature]), value, rtol=1e-10, atol=1e-12)
            assert seen == set(source_fingerprints)
    assert dict(excluded) == {x['scope']: x['token_limit_hits'] for x in b['generation_budget_audit']['scopes'] if x['scope'] != 'all'}
    assert matched['generation'] == matched['forward'] == 4647
    summary = {
        'analysis_sha256': hashlib.sha256((SOURCE / 'correlations.json').read_bytes()).hexdigest(),
        'matched_traces': dict(matched), 'retained_traces': len(rows),
        'token_limit_hits_by_dataset': dict(excluded), 'excluded_by_dataset': b['excluded_truncated_generations'], 'correct_by_dataset': dict(outcomes),
        'tagged_units_input': dict(units), 'tagged_units_retained': dict(retained_units),
        'binary': check_coefficients(rows, b['binary_correlations']),
        'metric_pairs': check_coefficients(rows, b['cross_feature_correlations']['trace_level'], rank=True),
        'repeated': check_repeated(rows, b['repeated_problem_analysis']),
        'views': [],
        'storage': {k: sum(d['storage'][k] for d in f['datasets']) for k in f['datasets'][0]['storage']},
        'not_recomputed': ['bootstrap intervals', 'BH q-values', 'mathematical answer scoring', 'model activations'],
    }
    for view in b['reasoning_view_analysis']['views']:
        path = ROOT / view['path']
        v = json.loads((path / 'correlations.json').read_text())
        vr = read_csv(path / 'trace_features.csv')
        assert len(vr) == len(rows) and {key(r) for r in vr} == set(lookup)
        for r in vr:
            for target in ['is_correct', 'has_backtracking', 'has_contradiction', 'has_self_correction']:
                assert r[target] == lookup[key(r)][target]
        if view['view'] == 'full':
            summary['full_reasoning_repeated'] = check_repeated(vr, v['repeated_problem_analysis'])
        coverage = sum(float(r['selected_token_count']) > 0 for r in vr)
        assert coverage == v['coverage']['traces_with_tokens']
        for dataset in b['datasets']:
            assert sum(r['dataset'] == dataset and float(r['selected_token_count']) > 0 for r in vr) == v['coverage']['datasets'][dataset]['traces_with_tokens']
        summary['views'].append({'view': view['view'], 'name': view['name'], 'traces_with_tokens': coverage,
                                 'binary': check_coefficients(vr, v['binary_correlations']),
                                 'metric_pairs': check_coefficients(vr, v['cross_feature_correlations']['trace_level'], rank=True)})
    maxima = []
    for dataset in b['datasets']:
        candidates = [r for r in b['expert_identity_analysis']['trace_point_biserial'] if r['scope'] == dataset and r['target'] == 'is_correct']
        if candidates:
            maxima.append(max(candidates, key=lambda r: abs(r['point_biserial_r'])))
    expert_rows = read_csv(SOURCE / 'expert_trace_features.csv')
    summary['expert_maxima'] = maxima
    summary['expert_maxima_check'] = check_coefficients(expert_rows, maxima)
    selected_features = {r['feature'] for r in maxima}
    continuous = [r for r in b['expert_identity_analysis']['trace_spearman']
                  if r['feature'] in selected_features and r['scope'] != 'all']
    summary['expert_continuous_metric_check'] = check_coefficients(expert_rows, continuous, rank=True)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

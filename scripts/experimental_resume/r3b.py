"""R3-B: paired reference-label dynamics and clean composition-controlled persistence.

No judge calls, model fits, or routing tensor loads. Requires the completed rk1c
freeze and a CPU Slurm allocation; missing reference mappings remain incomplete.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
OUT = ROOT / 'forum/tests/r3_dynamics_validity'
JUDGE = REPO / 'results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3'
MODELS = ('gpt', 'qwen36', 'gemma', 'glm', 'nemotron', 'qwen330b', 'qwen35')
SEED = 20260929


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    copied = OUT / 'code/rk1'
    copied.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / 'reasoning-kinematics/rk1/code/rk1').glob('*.py')):
        path = copied / source.name
        if path.exists() and sha(path) != sha(source):
            raise ValueError('historical helper copy changed')
        if not path.exists():
            shutil.copy2(source, path)
    target = OUT / 'code/r3b.py'
    if target.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, target)
    return {'driver': str(target), 'status': 'PREPARED; NOT SUBMITTED'}


def interval(values, level=.95):
    import numpy as np
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'lo': None, 'hi': None, 'n_valid': 0}
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {'lo': float(lo), 'hi': float(hi), 'n_valid': len(values)}


def human_check(C, D, MT, rng):
    import numpy as np
    predictions = JUDGE / 'selected_validation_predictions_20260827_173300.jsonl'
    metadata = JUDGE / 'results_20260827_173300.json'
    if not predictions.exists() or not metadata.exists():
        return {'status': 'INCOMPLETE', 'reason': 'reference document mapping absent'}
    config = json.loads(metadata.read_text())
    splits = config.get('split_question_ids', {})
    if 'locked_test' not in splits or 'validation' not in splits:
        return {'status': 'INCOMPLETE', 'reason': 'locked judge test IDs unavailable'}
    rows = [json.loads(line) for line in predictions.open()]
    docs = sorted({r['question_id'] for r in rows})
    if set(docs) & set(splits['locked_test']) or not set(docs) <= set(splits['validation']):
        raise ValueError('reference validation docs overlap the locked judge test')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['question_id']].append(row)
    matrices, pairs, motif_counts, motif_opportunities = {}, {}, {}, {}
    for label, field in [('gold', 'gold_label'), ('judge', 'predicted_label')]:
        matrices[label] = np.zeros((len(docs), 7, 7))
        pairs[label] = []
        motif_counts[label] = np.zeros(len(docs))
        motif_opportunities[label] = np.zeros((len(docs), 7, 7))
        for d, doc in enumerate(docs):
            part = sorted(grouped[doc], key=lambda r: r['unit_id'])
            if len({r['unit_id'] for r in part}) != len(part):
                raise ValueError('duplicate reference sentence identity')
            for before, after in zip(part, part[1:]):
                if after['unit_id'] != before['unit_id'] + 1:
                    continue
                a, b = C.CLASSES.index(before[field]), C.CLASSES.index(after[field])
                matrices[label][d, a, b] += 1
                pairs[label].append((a, b, d))
            for a, b, c in zip(part, part[1:], part[2:]):
                if b['unit_id'] == a['unit_id'] + 1 and c['unit_id'] == b['unit_id'] + 1:
                    motif_counts[label][d] += int(a[field] == c[field] != b[field])
                    motif_opportunities[label][d, C.CLASSES.index(a[field]), C.CLASSES.index(b[field])] += 1
    W = rng.multinomial(len(docs), np.full(len(docs), 1 / len(docs)), size=2000)
    output = {'status': 'COMPLETE_MEASUREMENT_CHECK', 'documents': docs, 'n_units': len(rows),
        'n_adjacent_pairs': {k: len(v) for k, v in pairs.items()},
        'locked_test_intersection': 0, 'unit': 'validation document', 'conditions': {}}
    for alpha in (.5, 0.):
        summaries = {}
        for label in ('gold', 'judge'):
            point = MT.summarize(matrices[label].sum(0) + alpha)
            boot = MT.summarize(np.einsum('bq,qij->bij', W, matrices[label]) + alpha)
            # ABA count excess over the fitted first-order transition reference.
            opportunity = motif_opportunities[label].sum(0)
            expected = float((opportunity * np.asarray(point['P']).T * ~np.eye(7, dtype=bool)).sum())
            aba = float(motif_counts[label].sum())
            opportunity_boot = np.einsum('bq,qij->bij', W, motif_opportunities[label])
            expected_boot = (opportunity_boot * np.asarray(boot['P']).transpose(0, 2, 1)
                             * ~np.eye(7, dtype=bool)).sum(axis=(1, 2))
            summaries[label] = {'point': point, 'boot': boot, 'aba': aba, 'expected_aba': expected,
                'aba_excess_rate': (aba - expected) / opportunity.sum(),
                'aba_excess_boot': (W @ motif_counts[label] - expected_boot) / opportunity_boot.sum(axis=(1, 2))}
        differences = {}
        for metric in ('switch_rate', 'EP', *('persist_' + c for c in C.CLASSES)):
            differences[metric] = {'estimate': float(summaries['judge']['point'][metric]
                                                     - summaries['gold']['point'][metric]),
                **interval(summaries['judge']['boot'][metric] - summaries['gold']['boot'][metric])}
        nulls = {}
        for label in ('gold', 'judge'):
            p = np.asarray(pairs[label], int)
            nulls[label] = {}
            for mode, cluster in [('pair_flip', None), ('document_flip', p[:, 2])]:
                null_ep = []
                for _ in range(200):
                    flip = (rng.random(len(p)) < .5 if cluster is None else
                            (rng.random(len(docs)) < .5)[cluster])
                    a, b = np.where(flip, p[:, 1], p[:, 0]), np.where(flip, p[:, 0], p[:, 1])
                    matrix = np.zeros((7, 7))
                    np.add.at(matrix, (a, b), 1)
                    null_ep.append(MT.summarize(matrix + alpha)['EP'])
                nulls[label][mode] = D.null_report(float(summaries[label]['point']['EP']), np.asarray(null_ep))
        differences['ABA_first_order_excess_rate'] = {
            'estimate': float(summaries['judge']['aba_excess_rate'] - summaries['gold']['aba_excess_rate']),
            **interval(summaries['judge']['aba_excess_boot'] - summaries['gold']['aba_excess_boot'])}
        output['conditions'][str(alpha)] = {'judge_minus_gold': differences, 'reversible_nulls': nulls,
            'gold': {'switch_rate': float(summaries['gold']['point']['switch_rate']),
                     'EP': float(summaries['gold']['point']['EP']),
                     'ABA_excess_rate': float(summaries['gold']['aba_excess_rate']),
                     'ABA_excess_rate_ci': interval(summaries['gold']['aba_excess_boot']),
                     'aba_count': summaries['gold']['aba'], 'first_order_expected_aba': summaries['gold']['expected_aba']},
            'judge': {'switch_rate': float(summaries['judge']['point']['switch_rate']),
                      'EP': float(summaries['judge']['point']['EP']),
                      'ABA_excess_rate': float(summaries['judge']['aba_excess_rate']),
                      'ABA_excess_rate_ci': interval(summaries['judge']['aba_excess_boot']),
                      'aba_count': summaries['judge']['aba'], 'first_order_expected_aba': summaries['judge']['expected_aba']}}
    output['limitation'] = 'Six-document measurement check; no seven-model validation or deconvolution.'
    return output


def composition(model, C, S, rng, *, exclude_capped=False, excluded_questions=()):
    import numpy as np
    import pandas as pd
    population = C.verify_json(ROOT / 'forum/tests/null/manifests/population.json')
    ids = set(population[model]['A']['attempt_ids'])
    att, _ = S.load_attempts(model, 'A')
    att = att[att['attempt_id'].isin(ids) & ~att['question'].isin(excluded_questions)].reset_index(drop=True)
    if exclude_capped:
        att = att[~att['capped']].reset_index(drop=True)
    outcomes = S.load_outcomes(model, 'A', att)
    # Missing/unscored/capped attempts remain operational failures in the primary.
    correct = (outcomes['correct'].fillna(False).astype(bool).to_numpy()
               & ~att['capped'].to_numpy(bool))
    sentences = pd.read_parquet(C.DYN / model / 'A/sentences.parquet', columns=S.SENT_COLS)
    sentences = sentences[sentences['attempt_id'].isin(att['attempt_id'])].sort_values(
        ['attempt_id', 'segment', 'sentence_index'], kind='stable').reset_index(drop=True)
    ai = pd.Index(att['attempt_id']).get_indexer(sentences['attempt_id'])
    qi = pd.Index(sorted(att['question'].unique())).get_indexer(sentences['question'])
    n_q = att['question'].nunique()
    labels = sentences['cls'].to_numpy(int)
    adjacent = (ai[:-1] == ai[1:]) & (sentences['segment'].to_numpy()[:-1] == sentences['segment'].to_numpy()[1:])
    adjacent &= np.diff(sentences['sentence_index'].to_numpy()) == 1
    src = np.flatnonzero(adjacent)
    dst = src + 1
    N, m = att['sentence_units'].to_numpy(float), att['labelled_units'].to_numpy(float)
    weights = N * (N - 1) / np.maximum(m * (m - 1), 1)
    groups = [indices.to_numpy() for _, indices in sentences.groupby(['attempt_id', 'segment']).groups.items()]
    def counts(values):
        result = np.zeros((n_q, 2, 2))  # question, correctness, E->E numerator / E outgoing denominator
        mask = values[src] == C.EXPLORE
        q, y, w = qi[src[mask]], correct[ai[src[mask]]].astype(int), weights[ai[src[mask]]]
        np.add.at(result[:, :, 1], (q, y), w)
        same = values[dst[mask]] == C.EXPLORE
        np.add.at(result[:, :, 0], (q[same], y[same]), w[same])
        return result
    observed = counts(labels)
    shuffled = []
    for _ in range(200):
        values = labels.copy()
        for group in groups:
            values[group] = rng.permutation(values[group])
        shuffled.append(counts(values))
    reference = np.mean(shuffled, axis=0)
    W = rng.multinomial(n_q, np.full(n_q, 1 / n_q), size=2000)
    def delta(values):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = values[..., 0] / values[..., 1]
        return ratio[..., 1] - ratio[..., 0]
    point = float(delta(observed.sum(0)) - delta(reference.sum(0)))
    boot = delta(np.einsum('bq,qyi->byi', W, observed)) - delta(np.einsum('bq,qyi->byi', W, reference))
    finite = boot[np.isfinite(boot)]
    p = min(1., 2 * min((1 + (finite <= 0).sum()) / (len(finite) + 1),
                        (1 + (finite >= 0).sum()) / (len(finite) + 1))) if len(finite) else None
    return {'model': model, 'estimate': point, 'ci95': interval(boot),
        'simultaneous_ci': interval(boot, 1 - .05 / 7), 'p_bootstrap': p,
        'observed_correct_minus_wrong_P_EE': float(delta(observed.sum(0))),
        'composition_reference_delta': float(delta(reference.sum(0))),
        'n_questions': int(n_q), 'attempts': int(len(att)), 'adjacent_pairs': int(len(src)),
        'sparse_gaps_joined': 0, 'shuffles': 200, 'bootstraps': 2000,
        'null': 'within attempt/segment shuffles of observed labels; preserves counts and sparse positions',
        'bootstrap': 'shared question weights for observed and composition reference',
        'Monte_Carlo': '200-shuffle mean is a finite approximation; no equivalence claim'}


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('R3-B requires a CPU Slurm allocation')
    started = time.monotonic()
    sys.path.insert(0, str(OUT / 'code'))
    import numpy as np
    from rk1 import common as C, describe as D, metrics as MT, sparse as S
    prerequisite = C.verify_json(ROOT / 'reasoning-kinematics/rk1c/results.json')
    family_path = REPO / 'report/experimental-resume-v1/family-freeze.json'
    family = C.verify_json(family_path)
    excluded = {q for f in family['new_parent_pools']['confirm_connected_excluded']
                for q in family['new_parent_pools']['families'][f]}
    inputs = [family_path, ROOT / 'reasoning-kinematics/rk1c/FROZEN.json',
        ROOT / 'reasoning-kinematics/rk1c/results.json', ROOT / 'forum/tests/null/manifests/population.json',
        JUDGE / 'results_20260827_173300.json', JUDGE / 'selected_validation_predictions_20260827_173300.jsonl']
    for model in MODELS:
        inputs.extend([C.V3R2 / model / 'A/attempts.parquet', C.DYN / model / 'A/sentences.parquet',
                       C.DYN / model / 'A/extract_meta.csv'])
    freeze = {'version': 'r3b-v1', 'inputs': {str(p): sha(p) for p in inputs if p.exists()},
        'absent': [str(p) for p in inputs if not p.exists()],
        'code': {str(p.relative_to(OUT)): sha(p) for p in sorted((OUT / 'code').rglob('*.py'))},
        'seed': SEED, 'n_boot': 2000, 'n_shuffle': 200, 'ceiling_cpu_core_h': 3,
        'population': 'same 522 dev+tune IDs across every model; separate confirm-family exclusion sensitivity',
        'family': 'Holm seven primary model composition contrasts; cap/family sensitivities descriptive',
        'prerequisite_seal': prerequisite['sha256']}
    path = OUT / 'FROZEN.json'
    if path.exists() and {k: v for k, v in C.verify_json(path).items() if k != 'sha256'} != freeze:
        raise ValueError('R3-B inputs changed; new protocol version required')
    if not path.exists():
        C.write_json(path, freeze)
    rng = np.random.default_rng(SEED)
    output = {'human_measurement': human_check(C, D, MT, rng), 'models': {}}
    for model in MODELS:
        output['models'][model] = {'primary': composition(model, C, S, rng),
            'capped_excluded': composition(model, C, S, rng, exclude_capped=True),
            'confirm_family_excluded': composition(model, C, S, rng, excluded_questions=excluded)}
        C.log(model + ' composition controls complete')
    ordered = sorted((r['primary']['p_bootstrap'], model) for model, r in output['models'].items()
                     if r['primary']['p_bootstrap'] is not None)
    previous = 0.
    for i, (p, model) in enumerate(ordered):
        previous = max(previous, min(1., (7 - i) * p))
        output['models'][model]['primary']['p_holm'] = previous
    output['dense_pilot'] = {'source': str(ROOT / 'reasoning-kinematics/rk1c/results/dense_sensitivity.json'),
        'sha256': sha(ROOT / 'reasoning-kinematics/rk1c/results/dense_sensitivity.json'),
        'status': 'VERIFIED REUSE; trace-heterogeneous first-order null, no new order search'}
    output['meta'] = {'seconds': time.monotonic() - started, 'job_id': os.environ['SLURM_JOB_ID'],
                      'frozen_sha256': C.verify_json(path)['sha256']}
    C.write_json(OUT / 'estimates.json', output)
    lines = ['# R3-B dynamics measurement and composition controls', '',
        'The reference-label check is confined to the non-locked validation documents.',
        'Clean observational comparisons exclude confirm IDs across every model. Historical exposure remains disclosed.', '',
        '| Model | Composition-adjusted correct−wrong Explore persistence | 95% CI | Holm p |',
        '|---|---:|---|---:|']
    for model, item in output['models'].items():
        r = item['primary']
        lines.append(f"| {model} | {r['estimate']:.5g} | [{r['ci95']['lo']}, {r['ci95']['hi']}] | {r.get('p_holm')} |")
    lines += ['', 'Caps and confirm-connected family exclusions are separately reported sensitivities.',
        'Seven simultaneous intervals are conservative Bonferroni intervals beside Holm p values.',
        'Imprecise or nonsignificant order differences do not establish occupancy-only equivalence or explain difficulty.',
        'No deconvolution, new judge calls, or large text-model fitting occurred.']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    if sys.argv[1:] == ['--prepare']:
        print(json.dumps(prepare(), indent=1))
    elif not sys.argv[1:]:
        run()
    else:
        raise SystemExit('usage: r3b.py [--prepare]')

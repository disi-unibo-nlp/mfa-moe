"""Offline adapters to the frozen boundary, strict and blinded J1 contracts."""
import csv
import json
from pathlib import Path

import native_finalization_v1 as N
import operator_panel_outcomes_v1 as O


def extract(index, loader=N.U.sealed):
    rows = []
    for item in index['records']:
        a = item['assignment']
        row = {**a, 'execution_status': item['status'], 'finish': None, 'tokens': None,
            'reasoning_tokens': None, 'answer_tokens': None, 'closure': None,
            'operational_correct': None, 'gradeable': False, 'boundary_status': 'MISSING_EXECUTION',
            'observed_tokens': None, 'completion_token_ids': None, 'receipt_sha256': item['receipt_sha256'],
            'fired': False, 'side_queries': 0, 'side_prompt_tokens': 0, 'side_generated_tokens': 0, 'side_seconds': 0.}
        if item['status'] == 'MISSING':
            N.require(item['receipt_path'] is None and item['receipt_sha256'] is None, 'missing receipt metadata')
            rows.append(row); continue
        r = loader(Path(item['receipt_path']))
        N.require(r['assignment'] == a and r['sha256'] == item['receipt_sha256'] and
                  r['status'] == item['status'] and r['manifest_sha256'] == index['manifest_sha256'], 'receipt identity differs')
        ids = r['token_ids']; row.update(observed_tokens=len(ids), finish=r['finish'], completion_token_ids=ids)
        if r['status'] == 'COMMITTED_GENERATION':
            N.require(r['maximum_tokens'] == N.CAP and r['qualification_only'] is False and
                r['error'] is None and r['routing'] == 'native' and r['finish'] in ('length', 'stop') and
                (r['finish'] != 'length' or len(ids) == N.CAP) and N.U.digest(ids) == r['token_ids_sha256'], 'invalid scientific receipt')
            N.require(len(r['captures']) == 2 and {c['rank'] for c in r['captures']} == {0, 1}, 'capture rank coverage')
            for capture in r['captures']:
                N.require(N.U.file_sha(Path(capture['path'])) == capture['file_sha256'], 'route bytes changed')
            row.update(O.lengths(ids, r['finish']))
            row.update(tokens=len(ids), gradeable=r['finish'] == 'stop',
                       operational_correct=False if r['finish'] == 'length' else None)
        else:
            N.require(r['status'] == 'GENERATION_ERROR' and bool(r['error']), 'invalid error')
            row.update(operational_correct=False, boundary_status='GENERATION_ERROR')
        rows.append(row)
    return rows


def paired(rows, families):
    lookup = {(r['family'], r['seed'], r['arm']): r for r in rows}
    N.require(len(rows) == len(lookup) == 4 * len(families) and
        set(lookup) == {(f, s, a) for f in families for s in (0, 1) for a in N.ARMS}, 'incomplete/duplicate factorial')
    matrix, bounds = [], []
    for family in families:
        values, limits = [], []
        for endpoint in N.ENDPOINTS:
            maximum = 1. if endpoint == 'operational_correct' else float(N.CAP)
            diffs, lows, highs = [], [], []
            for seed in (0, 1):
                left = lookup[family, seed, 'finalization'][endpoint]
                right = lookup[family, seed, 'original'][endpoint]
                for value in (left, right):
                    N.require(value is None or 0 <= float(value) <= maximum, 'endpoint outside bounds')
                diffs.append(None if left is None or right is None else float(left) - float(right))
                lows.append((0. if left is None else float(left)) - (maximum if right is None else float(right)))
                highs.append((maximum if left is None else float(left)) - (0. if right is None else float(right)))
            values.append(None if None in diffs else sum(diffs) / 2)
            limits.append([sum(lows) / 2, sum(highs) / 2])
        matrix.append(values); bounds.append(limits)
    return matrix, bounds


def inference(rows, families, replicates=50000, seed=20261005):
    import numpy as np
    matrix, bounds = paired(rows, families)
    result = {'comparison_order': list(N.ENDPOINTS), 'direction': 'finalization minus original',
        'families': len(families), 'paired_seed_cells': len(families) * 2,
        'identification_bounds': np.asarray(bounds).mean(0).tolist(),
        'point_estimates': None, 'simultaneous_95_intervals': None, 'small_sample': len(families) < 24,
        'bootstrap_replicates': replicates, 'bootstrap_seed': seed,
        'claim_limit': 'No equivalence or accuracy-retention assertion.'}
    if any(v is None for row in matrix for v in row):
        result['status'] = 'INCOMPLETE_ENDPOINTS'; return result
    x = np.asarray(matrix); n = len(x); estimate = x.mean(0)
    result['point_estimates'] = estimate.tolist()
    if n < 2:
        result['status'] = 'INSUFFICIENT_FAMILIES'; return result
    se = x.std(0, ddof=1) / np.sqrt(n); rng = np.random.default_rng(seed); maxima = []
    for start in range(0, replicates, 1000):
        draw = x[rng.integers(0, n, (min(1000, replicates - start), n))].mean(1)
        z = np.divide(np.abs(draw - estimate), se, out=np.zeros_like(draw), where=se > 0)
        maxima.extend(z.max(1).tolist())
    critical = float(np.quantile(maxima, .95, method='higher'))
    maximum = np.array([1., N.CAP, N.CAP])
    result.update(status='COMPLETE_EXPLORATORY_DISCOVERY', critical_value=critical,
        simultaneous_95_intervals=np.stack([np.maximum(-maximum, estimate-critical*se),
                                            np.minimum(maximum, estimate+critical*se)], 1).tolist(),
        zero_variance_components=np.flatnonzero(se == 0).tolist(), degenerate_intervals=bool(np.any(se == 0)),
        interval_warning='Eight families are a small sample. Degenerate bootstrap intervals do not prove zero effect.')
    return result


def prepare(index, out):
    from utility_outcomes_v3 import scoring_modules
    score, engine = scoring_modules()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(engine.snapshot_path(), local_files_only=True)
    manifest = N.U.sealed(N.DOC / 'MANIFEST.json')
    N.require(index['manifest_sha256'] == manifest['sha256'], 'grading manifest rebound')
    measurement = N.U.sealed(N.U.REPO / 'report/experimental-resume-v1/UTILITY_OUTCOME_MEASUREMENT_PLAN_v3.json')
    N.require(all(N.U.file_sha(Path(p)) == h for p, h in measurement['code_files'].items()), 'frozen grading changed')
    questions = N.U.sealed(N.U.QUESTIONS)
    N.require(questions['sha256'] == measurement['question_table_sha256'], 'offline references changed')
    rows = extract(index); bundle, mapping = [], []
    for row in rows:
        blind = 'nf-blind-v1|' + N.U.digest([index['sha256'], row['uid']])[:32]
        mapping.append({'assignment_uid': row['uid'], 'blind_uid': blind if row['gradeable'] else None})
        if row['gradeable']:
            q = questions['questions'][row['question']]
            bundle.append({'uid': blind, 'question': row['question'], 'dataset': q['dataset'],
                'problem': q['problem'], 'gold': q['gold'],
                'content_text': tokenizer.decode(row['completion_token_ids'], skip_special_tokens=False),
                'finish_reason': 'stop', 'natural_stop': True, 'cumulative_tokens': row['tokens']})
    score.assert_blind(bundle, 'native finalization blind answers')
    strict = score.strict_rows(bundle, workers=2)
    N.require({r['uid'] for r in strict} == {r['uid'] for r in bundle}, 'strict coverage')
    items = score.build_j1_items(bundle, {r['uid']: r for r in strict}, corpus='native-finalization-v1')
    score.assert_blind(items, 'native finalization J1 items')
    out.mkdir(parents=True, exist_ok=True)
    files = {name: score.write_jsonl(out / filename, data) for name, filename, data in
        [('blind', 'blind.jsonl.gz', bundle), ('strict', 'strict.jsonl.gz', strict),
         ('items', 'items.jsonl', items), ('map', 'arm-map.jsonl.gz', mapping)]}
    for row in rows:
        row.pop('completion_token_ids')
    return N.save(out / 'GRADE_PREP.json', {'schema': 'native-finalization-grading-v1',
        'index_sha256': index['sha256'], 'manifest_sha256': manifest['sha256'],
        'measurement_sha256': measurement['sha256'], 'family_order': index['family_order'],
        'assigned_rows': rows, 'files': files, 'J1_items': len(items)})


def analyze(prep, verdicts_path, out, accounting):
    from utility_outcomes_v3 import scoring_modules
    score, _ = scoring_modules()
    data = O.read_preparation(N.ROOT / 'grading', prep)
    strict = {r['uid']: r for r in data['strict']}
    item_of = {r['problem_id']: r['item_id'] for r in data['items']}
    verdicts, stats = score.load_verdicts(verdicts_path, set(item_of.values()))
    blind_for = {r['assignment_uid']: r['blind_uid'] for r in data['map']}
    rows = []
    for original in prep['assigned_rows']:
        row = dict(original)
        if row['gradeable']:
            uid = blind_for[row['uid']]
            row['operational_correct'], row['grade_status'] = O.adjudicated(
                strict[uid]['strict_correct'], 'stop', verdicts.get(item_of.get(uid)), score)
        rows.append(row)
    effect = inference(rows, prep['family_order']); summary = O.summarize(rows)
    out.mkdir(parents=True, exist_ok=True)
    fields = ['uid', 'family', 'seed', 'arm', 'execution_status', 'finish', 'operational_correct',
              'reasoning_tokens', 'answer_tokens', 'tokens', 'boundary_status']
    with (out / 'assignments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    with (out / 'effects.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['endpoint', 'estimate', 'simultaneous_95_low',
            'simultaneous_95_high', 'missing_low', 'missing_high'])
        for i, name in enumerate(N.ENDPOINTS):
            writer.writerow([name, effect['point_estimates'][i] if effect['point_estimates'] else None,
                *(effect['simultaneous_95_intervals'][i] if effect['simultaneous_95_intervals'] else [None, None]),
                *effect['identification_bounds'][i]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for arm in N.ARMS:
        block = [r for r in rows if r['arm'] == arm and r['operational_correct'] is not None]
        for ax, field in zip(axes, ('reasoning_tokens', 'tokens')):
            usable = [r for r in block if r[field] is not None]
            ax.scatter([r[field] for r in usable], [int(r['operational_correct']) for r in usable],
                       label=arm, alpha=.6, s=40)
            ax.set(xlabel=field.replace('_', ' '), ylabel='Operational correctness', yticks=[0, 1])
            ax.legend()
    fig.suptitle('Native finalization discovery: assigned outputs (unknowns omitted from scatter)')
    for ext in ('png', 'pdf'):
        fig.savefig(out / ('accuracy-versus-length.' + ext), dpi=160)
    plt.close(fig)
    estimates = effect['point_estimates']
    if estimates is None:
        recommendation = 'Effects remain unknown. Resolve missing execution/grading and inspect paired traces before choosing another hypothesis.'
    elif estimates[1] < 0 and estimates[0] >= 0:
        recommendation = 'Lower reasoning length with non-worse observed accuracy is a discovery candidate requiring independent confirmation.'
    elif estimates[1] < 0 and estimates[0] < 0:
        recommendation = 'Shorter reasoning with lower observed accuracy is a tradeoff.'
    else:
        recommendation = 'No clear finalization benefit established; use paired traces for a discriminating next hypothesis. Do not expand automatically.'
    artifacts = {p.name: N.U.file_sha(p) for p in out.iterdir() if p.suffix in ('.csv', '.png', '.pdf')}
    return N.save(out / 'ANALYSIS.json', {'schema': 'native-finalization-analysis-v1',
        'grade_preparation_sha256': prep['sha256'], 'assigned_rows': rows, 'effect': effect,
        'arm_summary': summary, 'verdict_coverage': stats, 'accounting': accounting,
        'recommendation': recommendation, 'artifacts': artifacts, 'status': effect['status']})

"""Offline, arm-blind strict/J1 grading and paired family utility inference.

Gold answers are materialized only here, after generation. The original v1
utility convention is retained: length caps and committed generation errors
count operationally wrong. Missing executions/grades remain unknown.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import utility_scout_v1 as u
import rate_overnight_semantics_v2 as storage

DOC = u.REPO / 'report/experimental-resume-v1'
BASE = u.STAGE / 'code/s1-9a61e32f48c04c24'
PLAN = DOC / 'UTILITY_OUTCOME_MEASUREMENT_PLAN_v3.json'
CORPUS = 'routing-utility-scout-v3'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scoring_modules():
    """Pin once, then verify cached modules rather than pinning them again."""
    from diagnose_mechanism_validation_v3 import pin_qualified_worker
    overlay = u.STAGE / 'addenda/ordered/9727c10299b71e7a/moe_exp_src'
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    if 'moe_exp' not in sys.modules:
        pin_qualified_worker(overlay)
    else:
        cached = sys.modules['moe_exp']
        require(Path(cached.__file__).resolve() == (overlay / 'moe_exp/__init__.py').resolve(),
                'cached scoring package is outside the qualified overlay')
    from moe_steer import score, engine
    require(Path(score.__file__).resolve() == (BASE / 'moe_steer/score.py').resolve() and
            Path(engine.__file__).resolve() == (BASE / 'moe_steer/engine.py').resolve() and
            Path(score.answer_equivalence.__file__).resolve() ==
            (overlay / 'moe_exp/correlation_pipeline/answer_equivalence.py').resolve(),
            'cached scorer, engine or J1 source is not the frozen qualified implementation')
    return score, engine


def source_files():
    score, engine = scoring_modules()
    from moe_exp.correlation_pipeline import outcomes, scoring
    from moe_exp.models import token_replay
    paths = [Path(__file__), Path(__file__).with_suffix('.sbatch'), Path(u.__file__),
             Path(score.__file__), Path(engine.__file__), Path(score.answer_equivalence.__file__),
             Path(outcomes.__file__), Path(scoring.__file__), Path(token_replay.__file__)]
    return {str(p.resolve()): u.file_sha(p) for p in paths}


def freeze():
    plan = u.sealed(DOC / 'UTILITY_SCOUT_PLAN_v1.json')
    body = {'schema': 'utility-outcome-measurement-plan-v3', 'utility_plan_sha256': plan['sha256'],
        'question_table_sha256': u.sealed(u.QUESTIONS)['sha256'], 'code_files': source_files(),
        'superseded_plan_sha256': u.sealed(DOC / 'UTILITY_OUTCOME_MEASUREMENT_PLAN_v2.json')['sha256'],
        'amendment': 'Idempotent verified scoring imports and canonical vocabulary fingerprint validation. V2 pinned twice and compared tokenizer file bytes to the vocabulary digest. No grading, estimator or endpoint changes.',
        'population': 'All 96 frozen families, two arms and two seeds; imported pilot assignments retain canonical UIDs.',
        'primary_accuracy': 'Naturally stopped answers: frozen strict math_verify and arm-blind J1. Length caps and committed generation errors operationally wrong, as in utility_scout_v1. Missing executions or unfinished J1 remain unknown, not deleted.',
        'primary_tokens': 'Emitted plus injected generator tokens; no injections in v2. Error/missing executions have unknown complete-execution token endpoint and identification bounds. Actual observed partial tokens remain in accounting.',
        'inference': {'estimator': 'Equal-family mean of within-seed policy minus native, averaged over two seeds.',
                      'replicates': 50000, 'seed': 20261004, 'alpha': .05,
                      'multiplicity_family': ['operational_accuracy_change', 'generator_token_change'],
                      'method': 'Paired family bootstrap, simultaneous max absolute standardized mean deviation, fixed original standard errors; joint rectangular 95% region. Zero-variance components reported explicitly; no noninferiority/equivalence claim.'},
        'unknowns': 'Report family-aggregated identification bounds; no complete effect estimate or confidence region until both endpoints are observed for all assigned cells.',
        'grading': 'J1 requested only for naturally stopped strict-rejected answers; exact frozen scorer rubric, model, parser and votes. Gold/problem available only in offline blind bundle.',
        'claim_limit': '16k original-prompt exploratory utility scout. No accuracy-retention assertion without a separately selected noninferiority margin.'}
    value = storage.save(PLAN, body)
    print('FROZEN', value['sha256'])


def validate_index(index):
    require(index['schema'] == 'utility-production-reconciled-index-v1' and index['assigned'] == 384,
            'unexpected reconciled utility index')
    plan = u.sealed(Path(index['plan_path'])); u.validate_plan(plan)
    require(plan['sha256'] == index['plan_sha256'] and
            [r['assignment'] for r in index['records']] == plan['assignments'],
            'index skips, duplicates, reorders or changes an assigned cell')
    production = u.sealed(Path(index['production_manifest_path']))
    require(production['sha256'] == index['production_manifest_sha256'], 'production manifest changed')
    policy = u.sealed(Path(index['policy_path']))
    require(policy['sha256'] == index['policy_sha256'], 'selected policy changed')
    measurement = u.sealed(PLAN)
    require(measurement['utility_plan_sha256'] == plan['sha256'] and measurement['code_files'] == source_files(),
            'frozen measurement sources or enrollment changed')
    return plan, measurement


def extract_rows(index, receipt_loader=u.sealed):
    """Pure receipt adaptation: exact assignments, caps and incomplete cells."""
    rows = []
    for item in index['records']:
        assignment = item['assignment']
        status = item['status']
        row = {**assignment, 'execution_status': status, 'completion_token_ids': None,
               'finish': None, 'observed_tokens': None, 'tokens': None,
               'operational_correct': None, 'gradeable': False, 'fired': None,
               'side_generated_tokens': None, 'side_prompt_tokens': None,
               'receipt_sha256': item['receipt_sha256']}
        if status == 'MISSING':
            require(item['receipt_path'] is None and item['receipt_sha256'] is None, 'missing cell has a receipt')
            rows.append(row); continue
        require(status in ('COMMITTED_GENERATION', 'GENERATION_ERROR'), 'unknown execution status')
        receipt = receipt_loader(Path(item['receipt_path']))
        require(receipt['sha256'] == item['receipt_sha256'] and receipt['assignment'] == assignment and
                receipt['status'] == status and receipt['binding_sha256'] == item['source_binding_sha256'],
                'receipt assignment, seal, binding or status differs')
        result = receipt.get('result')
        if result is not None:
            require(result['assignment'] == assignment, 'backend assignment changed')
            ctl = result['controller']; state = ctl['state']; ids = state['completion_token_ids']
            require(state['uid'] == assignment['uid'] and state['prompt_token_ids_sha256'] == assignment['prompt_token_ids_sha256'] and
                    isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids) and
                    len(ids) <= 16384 and state['token_sources'] == ['emitted'] * len(ids) and
                    ctl['observed_generator_tokens'] == len(ids), 'controller identity/token accounting differs')
            require(assignment['arm'] != 'native' or ctl['episode'] is None, 'native cell has an intervention')
            row.update(completion_token_ids=ids, finish=state['finish'], observed_tokens=len(ids),
                       fired=ctl['episode'] is not None, side_generated_tokens=ctl['side_generated_tokens'],
                       side_prompt_tokens=ctl['side_prompt_tokens'])
        if status == 'COMMITTED_GENERATION':
            require(result is not None and row['finish'] in ('stop', 'length') and receipt.get('error') is None and
                    not result.get('qualification_only', False) and result.get('maximum_tokens') == 16384,
                    'committed receipt is not a 16k scientific completion')
            require(row['finish'] != 'length' or row['observed_tokens'] == 16384, 'short length stop')
            require(item['routed_array_sha256'] == receipt['routed_array_sha256'] and item['routed_path'] and
                    u.file_sha(Path(item['routed_path'])) == receipt['routed_array_sha256'], 'route artifact changed')
            row['tokens'] = row['observed_tokens']
            row['gradeable'] = row['finish'] == 'stop'
            row['operational_correct'] = None if row['gradeable'] else False
        else:
            require(bool(receipt.get('error')), 'generation error lacks failure evidence')
            row['operational_correct'] = False
        rows.append(row)
    return rows


def prepare(index_path, out):
    index = u.sealed(index_path); plan, measurement = validate_index(index)
    rows = extract_rows(index)
    questions = u.sealed(u.QUESTIONS)
    require(questions['sha256'] == measurement['question_table_sha256'], 'offline reference table changed')
    score, engine = scoring_modules()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(engine.snapshot_path(), local_files_only=True)
    from moe_exp.models.token_replay import tokenizer_fingerprint
    require(tokenizer_fingerprint(tokenizer) == plan['tokenizer_sha256'], 'actual decoder vocabulary fingerprint differs')
    prompts = u.sealed(u.PROMPTS)
    require(prompts['tokenizer_sha256'] == plan['tokenizer_sha256'] and
            u.file_sha(u.PROMPTS) == plan['sources']['prompt_table_file_sha256'], 'original tokenizer contract changed')
    bundle, mapping = [], []
    for row in rows:
        uid = 'utility-grade-v3|' + u.digest([index['sha256'], row['uid']])[:32]
        mapping.append({'assignment_uid': row['uid'], 'blind_uid': uid if row['gradeable'] else None})
        if row['gradeable']:
            question = questions['questions'][row['question']]
            bundle.append({'uid': uid, 'question': row['question'], 'dataset': question['dataset'],
                'problem': question['problem'], 'gold': question['gold'],
                'content_text': tokenizer.decode(row['completion_token_ids'], skip_special_tokens=False),
                'finish_reason': 'stop', 'natural_stop': True, 'cumulative_tokens': row['tokens']})
    score.assert_blind(bundle, 'utility offline answer bundle')
    strict = score.strict_rows(bundle, workers=2)
    strict_by_uid = {r['uid']: r for r in strict}
    require(len(strict_by_uid) == len(bundle), 'strict scoring coverage differs')
    items = score.build_j1_items(bundle, strict_by_uid, corpus=CORPUS)
    score.assert_blind(items, 'utility offline J1 items')
    require(len(items) <= 384, 'J1 maximum exceeds assigned population')
    require(not out.exists(), 'grading output already exists; preserve partial attempts and choose a new bound preparation path')
    out.mkdir(parents=True, exist_ok=False)
    files = {name: score.write_jsonl(out / filename, data) for name, filename, data in
        [('blind', 'blind.jsonl.gz', bundle), ('strict', 'strict.jsonl.gz', strict),
         ('items', 'items.jsonl', items), ('map', 'arm-map.jsonl.gz', mapping)]}
    for row in rows:
        row.pop('completion_token_ids')
    value = storage.save(out / 'GRADE_PREP.json', {'schema': 'utility-grade-preparation-v3',
        'index_path': str(index_path.resolve()), 'index_sha256': index['sha256'],
        'measurement_plan_sha256': measurement['sha256'], 'plan_sha256': plan['sha256'],
        'decoder_tokenizer_file_sha256': u.file_sha(Path(engine.snapshot_path()) / 'tokenizer.json'),
        'decoder_vocabulary_sha256': tokenizer_fingerprint(tokenizer),
        'assigned_rows': rows, 'files': files, 'blind_count': len(bundle), 'J1_items': len(items),
        'status': 'PENDING_J1' if items else 'STRICT_COMPLETE', 'job_id': os.environ['SLURM_JOB_ID']})
    print('PREPARED', value['sha256'], len(items), 'blind J1 items')


def paired_values(rows, family_order):
    import numpy as np
    lookup = {(r['family'], r['seed'], r['arm']): r for r in rows}
    require(len(lookup) == len(rows) == 4 * len(family_order), 'duplicate/incomplete factorial')
    bounds, values = [], []
    for family in family_order:
        cells, limits = [], []
        for field, maximum in (('operational_correct', 1.), ('tokens', 16384.)):
            diffs, low, high = [], [], []
            for seed in (0, 1):
                left = lookup[(family, seed, 'frozen_policy')][field]
                right = lookup[(family, seed, 'native')][field]
                diffs.append(None if left is None or right is None else float(left) - float(right))
                low.append((0. if left is None else float(left)) - (maximum if right is None else float(right)))
                high.append((maximum if left is None else float(left)) - (0. if right is None else float(right)))
            cells.append(None if any(x is None for x in diffs) else sum(diffs) / 2)
            limits.append([sum(low) / 2, sum(high) / 2])
        values.append(cells); bounds.append(limits)
    return values, np.asarray(bounds, dtype=float)


def inference(rows, family_order, *, replicates=50000, seed=20261004):
    import numpy as np
    values, bounds = paired_values(rows, family_order)
    result = {'families': len(family_order), 'identification_bounds': bounds.mean(axis=0).tolist(),
              'outcome_order': ['operational_accuracy_change', 'generator_token_change'],
              'point_estimates': None, 'simultaneous_95_intervals': None, 'joint_region': None}
    if any(x is None for row in values for x in row):
        result['status'] = 'INCOMPLETE_ENDPOINTS'; return result
    matrix = np.asarray(values, dtype=float); n = len(matrix)
    require(n > 1, 'family bootstrap requires multiple families')
    estimate = matrix.mean(axis=0); se = matrix.std(axis=0, ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(seed); maxima = []
    for start in range(0, replicates, 1000):
        sample = matrix[rng.integers(0, n, size=(min(1000, replicates-start), n))].mean(axis=1)
        standardized = np.divide(np.abs(sample-estimate), se,
            out=np.zeros_like(sample), where=se > 0)
        maxima.extend(standardized.max(axis=1).tolist())
    critical = float(np.quantile(maxima, .95, method='higher'))
    lower = np.maximum(estimate-critical*se, [-1, -16384]); upper = np.minimum(estimate+critical*se, [1, 16384])
    result.update(status='COMPLETE_EXPLORATORY_UTILITY', point_estimates=estimate.tolist(),
        simultaneous_95_intervals=np.stack([lower, upper], axis=1).tolist(),
        joint_region='Cartesian product of simultaneous intervals for accuracy and generator tokens.',
        zero_variance_components=np.flatnonzero(se == 0).tolist(), bootstrap_replicates=replicates,
        bootstrap_seed=seed, critical_value=critical,
        interpretation='Paired family resampling; a similar accuracy estimate does not establish noninferiority or equivalence.')
    return result


def analyze(prep_path, verdicts_path, out):
    import gzip
    prep = u.sealed(prep_path); index = u.sealed(Path(prep['index_path'])); plan, measurement = validate_index(index)
    require(prep['index_sha256'] == index['sha256'] and prep['measurement_plan_sha256'] == measurement['sha256'], 'grading source changed')
    score, _ = scoring_modules()
    directory = prep_path.parent
    names = {'blind':'blind.jsonl.gz', 'strict':'strict.jsonl.gz', 'items':'items.jsonl', 'map':'arm-map.jsonl.gz'}
    data = {}
    for name, filename in names.items():
        path = directory / filename
        require(u.file_sha(path) == prep['files'][name]['file_sha256'], 'grading file changed: ' + filename)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            data[name] = [json.loads(line) for line in stream if line.strip()]
    items = data['items']; item_of = {r['problem_id']: r['item_id'] for r in items}
    verdicts, statistics = score.load_verdicts(verdicts_path, set(item_of.values()))
    require(set(verdicts) == set(item_of.values()) and set(verdicts.values()) <= {'EQUIVALENT', 'NOT_EQUIVALENT', 'UNCERTAIN'}, 'J1 verdict coverage is incomplete or malformed')
    strict = {r['uid']: r for r in data['strict']}; answers = {}
    for row in data['blind']:
        answers[row['uid']] = score.adjudicate_uid(strict[row['uid']]['strict_correct'], 'stop',
            verdicts.get(item_of.get(row['uid'])))[2]
    blind_for = {r['assignment_uid']: r['blind_uid'] for r in data['map']}
    rows = []
    for original in prep['assigned_rows']:
        row = dict(original)
        if row['gradeable']:
            row['operational_correct'] = bool(answers[blind_for[row['uid']]])
        rows.append(row)
    result = inference(rows, plan['family_order'])
    out.mkdir(parents=True, exist_ok=True)
    summary = storage.save(out / 'ANALYSIS.json', {'schema': 'utility-outcome-analysis-v3',
        'measurement_plan_sha256': measurement['sha256'], 'index_sha256': index['sha256'],
        'grade_preparation_sha256': prep['sha256'], 'verdicts_file_sha256': u.file_sha(verdicts_path) if verdicts_path else None,
        'verdict_coverage': statistics, 'execution_counts': dict(Counter(r['execution_status'] for r in rows)),
        'finish_counts': dict(Counter(str(r['finish']) for r in rows)), 'assigned_rows': rows,
        'effect': result, 'job_id': os.environ['SLURM_JOB_ID'],
        'cost': 'Use the reconciled index allocation accounting for all four GPUs; side token counts are diagnostic and are not added twice to GPU cost.'})
    if result['point_estimates'] is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        accuracy, tokens = result['point_estimates']; intervals = result['simultaneous_95_intervals']
        fig, ax = plt.subplots(figsize=(6,4))
        ax.add_patch(Rectangle((intervals[1][0], intervals[0][0]*100), intervals[1][1]-intervals[1][0],
            (intervals[0][1]-intervals[0][0])*100, facecolor='tab:blue', alpha=.2, edgecolor='tab:blue'))
        ax.plot(tokens, accuracy*100, 'o'); ax.axhline(0, color='grey', linewidth=.7); ax.axvline(0, color='grey', linewidth=.7)
        ax.set(xlabel='Generator token change (policy − native)', ylabel='Accuracy change (percentage points)',
               title='16k utility scout: joint 95% family-bootstrap region')
        ax.autoscale(); fig.tight_layout(); fig.savefig(out/'accuracy_token_region.pdf'); fig.savefig(out/'accuracy_token_region.svg'); plt.close(fig)
    print('ANALYZED', summary['sha256'], result['status'])


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    p = sub.add_parser('prepare'); p.add_argument('--index', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('analyze'); p.add_argument('--preparation', type=Path, required=True); p.add_argument('--verdicts', type=Path); p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze': freeze(); return
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz', 'offline utility measurements require CPU Slurm')
    if args.command == 'prepare': prepare(args.index, args.out)
    else: analyze(args.preparation, args.verdicts, args.out)


if __name__ == '__main__':
    main()

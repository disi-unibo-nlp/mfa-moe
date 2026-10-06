"""Verified reasoning boundaries, blind strict/J1 grading and nine paired effects."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

import operator_panel_v1 as P


def boundary_contract(tokenizer, fingerprint):
    from moe_exp.models.token_replay import tokenizer_fingerprint
    P.require(tokenizer_fingerprint(tokenizer) == fingerprint and
              tokenizer.encode('</think>', add_special_tokens=False) == [P.THINK_END_ID] and
              tokenizer.decode([P.THINK_END_ID], skip_special_tokens=False) == '</think>',
              'reasoning marker or tokenizer vocabulary differs')
    return {'closing_marker': '</think>', 'closing_token_id': P.THINK_END_ID,
            'vocabulary_sha256': fingerprint, 'marker_excluded': True}


def lengths(ids, finish, *, prompt_reasoning_open=True, cap=16384):
    P.require(isinstance(ids, (tuple, list)) and all(type(x) is int and x >= 0 for x in ids) and
              len(ids) <= cap, 'invalid generated token history')
    if not prompt_reasoning_open:
        return {'reasoning_tokens': None, 'answer_tokens': None, 'boundary_status': 'PREFIX_ALREADY_CLOSED',
                'closure': False, 'closing_markers': ids.count(P.THINK_END_ID)}
    if P.THINK_END_ID in ids:
        at = ids.index(P.THINK_END_ID)
        return {'reasoning_tokens': at, 'answer_tokens': len(ids) - at - 1,
                'boundary_status': 'CLOSED' if ids.count(P.THINK_END_ID) == 1 else 'MULTIPLE_MARKERS_FIRST_CLOSURE',
                'closure': True, 'closing_markers': ids.count(P.THINK_END_ID)}
    return {'reasoning_tokens': len(ids) if finish == 'length' else None,
            'answer_tokens': 0 if finish == 'length' else None,
            'boundary_status': 'UNCLOSED_CAPPED' if finish == 'length' else 'MISSING_MARKER_' + str(finish).upper(),
            'closure': False, 'closing_markers': 0}


def extract(index, loader=P.U.sealed):
    rows = []
    for item in index['records']:
        a = item['assignment']; status = item['status']
        row = {**a, 'execution_status': status, 'finish': None, 'observed_tokens': None,
               'tokens': None, 'reasoning_tokens': None, 'answer_tokens': None, 'closure': None,
               'boundary_status': 'MISSING_EXECUTION', 'operational_correct': None,
               'gradeable': False, 'fired': None, 'side_generated_tokens': None,
               'side_prompt_tokens': None, 'side_queries': None, 'side_seconds': None,
               'completion_token_ids': None, 'receipt_sha256': item['receipt_sha256']}
        if status == 'MISSING':
            P.require(item['receipt_path'] is None and item['receipt_sha256'] is None, 'missing execution has receipt')
            rows.append(row); continue
        receipt = loader(Path(item['receipt_path']))
        P.require(receipt['sha256'] == item['receipt_sha256'] and receipt['assignment'] == a and
                  receipt['status'] == status and receipt['binding_sha256'] == item['source_binding_sha256'],
                  'panel receipt identity differs')
        result = receipt.get('result')
        if result is not None:
            ctl = result['controller']; state = ctl['state']; ids = state['completion_token_ids']
            P.require(result['assignment'] == a and state['uid'] == a['uid'] and
                      state['prompt_token_ids_sha256'] == a['prompt_token_ids_sha256'] and
                      state['token_sources'] == ['emitted'] * len(ids) and ctl['observed_generator_tokens'] == len(ids) and
                      (a['arm'] != 'native' or (ctl['episode'] is None and not ctl['queries'])),
                      'panel token, assignment or native behavior differs')
            episode = ctl['episode']
            if episode:
                P.require(episode['selected_arm'] == a['arm'] and episode['relative_slots'] == [0] and
                          episode['pulse_width'] == 256 and episode['stop_at_reasoning_closure'] is True,
                          'operator binding or pulse differs')
            row.update(completion_token_ids=ids, finish=state['finish'], observed_tokens=len(ids),
                       fired=episode is not None, side_generated_tokens=ctl['side_generated_tokens'],
                       side_prompt_tokens=ctl['side_prompt_tokens'], side_queries=len(ctl['queries']),
                       side_seconds=ctl['side_elapsed_seconds'])
            row.update(lengths(ids, state['finish']))
        if status == 'COMMITTED_GENERATION':
            P.require(result is not None and result['maximum_tokens'] == 16384 and
                      result['qualification_only'] is False and receipt['error'] is None and
                      row['finish'] in ('stop', 'length') and
                      (row['finish'] != 'length' or row['observed_tokens'] == 16384),
                      'scientific completion/cap differs')
            P.require(item['routed_path'] and P.U.file_sha(Path(item['routed_path'])) == receipt['routed_array_sha256'],
                      'panel routed bytes changed')
            row['tokens'] = row['observed_tokens']; row['gradeable'] = row['finish'] == 'stop'
            row['operational_correct'] = None if row['gradeable'] else False
        else:
            P.require(status == 'GENERATION_ERROR' and bool(receipt['error']), 'invalid generation error')
            row['reasoning_tokens'] = row['answer_tokens'] = None
            row['operational_correct'] = False
        rows.append(row)
    return rows


def paired_values(rows, families):
    lookup = {(r['family'], r['seed'], r['arm']): r for r in rows}
    P.require(len(lookup) == len(rows) == len(families) * 8 and
              set(lookup) == {(f, s, a) for f in families for s in (0, 1) for a in P.ARMS},
              'paired factorial incomplete or duplicated')
    matrix, bounds = [], []
    for family in families:
        values, limits = [], []
        for op in P.ARMS[1:]:
            for field in P.ENDPOINTS:
                maximum = 1. if field == 'operational_correct' else 16384.
                diff, low, high = [], [], []
                for seed in (0, 1):
                    left, right = lookup[family, seed, op][field], lookup[family, seed, 'native'][field]
                    diff.append(None if left is None or right is None else float(left) - float(right))
                    low.append((0. if left is None else float(left)) - (maximum if right is None else float(right)))
                    high.append((maximum if left is None else float(left)) - (0. if right is None else float(right)))
                values.append(None if None in diff else sum(diff) / 2)
                limits.append([sum(low) / 2, sum(high) / 2])
        matrix.append(values); bounds.append(limits)
    return matrix, bounds


def inference(rows, families, replicates=50000, seed=20261005):
    import numpy as np
    matrix, bounds = paired_values(rows, families)
    result = {'families': len(families), 'comparison_order': [op + ':' + field for op in P.ARMS[1:] for field in P.ENDPOINTS],
        'identification_bounds': np.asarray(bounds).mean(axis=0).tolist(), 'point_estimates': None,
        'simultaneous_95_intervals': None, 'small_sample': len(families) < 24,
        'claim_limit': 'No equivalence, noninferiority or accuracy-retention assertion.'}
    if any(v is None for row in matrix for v in row):
        result['status'] = 'INCOMPLETE_ENDPOINTS'; return result
    x = np.asarray(matrix, float); n = len(x); estimate = x.mean(axis=0)
    if n < 2:
        result.update(status='INSUFFICIENT_FAMILIES', point_estimates=estimate.tolist()); return result
    se = x.std(axis=0, ddof=1) / np.sqrt(n); rng = np.random.default_rng(seed); maxima = []
    for start in range(0, replicates, 1000):
        sample = x[rng.integers(0, n, (min(1000, replicates - start), n))].mean(axis=1)
        z = np.divide(np.abs(sample - estimate), se, out=np.zeros_like(sample), where=se > 0)
        maxima.extend(z.max(axis=1).tolist())
    critical = float(np.quantile(maxima, .95, method='higher'))
    maximum = np.array([1., 16384., 16384.] * 3)
    intervals = np.stack([np.maximum(-maximum, estimate - critical * se),
                          np.minimum(maximum, estimate + critical * se)], axis=1)
    result.update(status='COMPLETE_EXPLORATORY_PANEL', point_estimates=estimate.tolist(),
        simultaneous_95_intervals=intervals.tolist(), critical_value=critical, bootstrap_seed=seed,
        bootstrap_replicates=replicates, zero_variance_components=np.flatnonzero(se == 0).tolist(),
        degenerate_intervals=bool(np.any(se == 0)),
        interval_warning='Small-sample and zero-variance bootstrap intervals can be unreliable; a degenerate interval does not prove zero effect.')
    return result


def adjudicated(strict, finish, verdict, score):
    if finish != 'stop':
        return False, 'OPERATIONALLY_WRONG'
    if strict is True:
        return True, 'STRICT_CORRECT'
    if verdict is None:
        return None, 'J1_PENDING'
    P.require(verdict in ('EQUIVALENT', 'NOT_EQUIVALENT', 'UNCERTAIN'), 'unknown J1 verdict')
    value, status, operational = score.adjudicate_uid(strict, finish, verdict)
    # Completed UNCERTAIN votes retain the frozen operational convention;
    # absent/unfinished votes remain unknown and never become false by coercion.
    return bool(operational), status


def scoring():
    import utility_j1_entry_v2
    utility_j1_entry_v2.install()
    import utility_outcomes_v3
    return utility_outcomes_v3.scoring_modules()


def prepare(index_path, out, measurement):
    index = P.U.sealed(index_path)
    P.require(index['schema'] == 'panel-reconciled-index-v1' and index['plan_sha256'] == P.U.sealed(P.PLAN)['sha256'],
              'grading panel index changed')
    plan = P.U.sealed(P.PLAN)
    P.require([r['assignment'] for r in index['records']] == P.assignments(plan, index['family_order']),
              'grading index factorial differs')
    rows = extract(index); score, engine = scoring()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(engine.snapshot_path(), local_files_only=True)
    boundary = boundary_contract(tokenizer, plan['tokenizer_sha256'])
    questions = P.U.sealed(P.U.QUESTIONS)
    P.require(questions['sha256'] == measurement['question_table_sha256'], 'offline gold source changed')
    bundle, mapping = [], []
    for row in rows:
        uid = 'panel-blind-v1|' + P.U.digest([index['sha256'], row['uid']])[:32]
        mapping.append({'assignment_uid': row['uid'], 'blind_uid': uid if row['gradeable'] else None})
        if row['gradeable']:
            q = questions['questions'][row['question']]
            bundle.append({'uid': uid, 'question': row['question'], 'dataset': q['dataset'],
                'problem': q['problem'], 'gold': q['gold'],
                'content_text': tokenizer.decode(row['completion_token_ids'], skip_special_tokens=False),
                'finish_reason': 'stop', 'natural_stop': True, 'cumulative_tokens': row['tokens']})
    score.assert_blind(bundle, 'operator panel blind answers')
    strict = score.strict_rows(bundle, workers=2)
    P.require({r['uid'] for r in strict} == {r['uid'] for r in bundle}, 'strict grading coverage differs')
    items = score.build_j1_items(bundle, {r['uid']: r for r in strict}, corpus='operator-panel-v1')
    score.assert_blind(items, 'operator panel J1 prompts')
    out.mkdir(parents=True, exist_ok=False)
    files = {name: score.write_jsonl(out / filename, data) for name, filename, data in
             [('blind', 'blind.jsonl.gz', bundle), ('strict', 'strict.jsonl.gz', strict),
              ('items', 'items.jsonl', items), ('map', 'arm-map.jsonl.gz', mapping)]}
    for row in rows:
        row.pop('completion_token_ids')
    return P.save(out / 'GRADE_PREP.json', {'schema': 'panel-grade-preparation-v1', 'index_path': str(index_path),
        'index_sha256': index['sha256'], 'measurement_sha256': measurement['sha256'],
        'family_order': index['family_order'], 'boundary': boundary, 'assigned_rows': rows,
        'files': files, 'J1_items': len(items), 'blind_count': len(bundle)})


def read_preparation(root, prep):
    data = {}
    for name, filename in [('blind', 'blind.jsonl.gz'), ('strict', 'strict.jsonl.gz'),
                            ('items', 'items.jsonl'), ('map', 'arm-map.jsonl.gz')]:
        path = root / filename
        P.require(P.U.file_sha(path) == prep['files'][name]['file_sha256'], 'grading bytes changed')
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            data[name] = [json.loads(line) for line in stream if line.strip()]
    return data


def analysis(prep_path, verdicts_path, out, accounting):
    prep = P.U.sealed(prep_path); data = read_preparation(prep_path.parent, prep)
    score, _ = scoring()
    strict = {r['uid']: r for r in data['strict']}
    item_of = {r['problem_id']: r['item_id'] for r in data['items']}
    verdicts, stats = score.load_verdicts(verdicts_path, set(item_of.values()))
    blind_for = {r['assignment_uid']: r['blind_uid'] for r in data['map']}
    rows = []
    for original in prep['assigned_rows']:
        row = dict(original)
        if row['gradeable']:
            uid = blind_for[row['uid']]
            row['operational_correct'], row['grade_status'] = adjudicated(strict[uid]['strict_correct'], 'stop',
                verdicts.get(item_of.get(uid)), score)
        rows.append(row)
    result = inference(rows, prep['family_order'])
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    artifacts = write_tables_figures(rows, result, out)
    return P.save(out / 'ANALYSIS.json', {'schema': 'operator-panel-analysis-v1',
        'grade_preparation_sha256': prep['sha256'], 'measurement_sha256': prep['measurement_sha256'],
        'assigned_rows': rows, 'effect': result, 'arm_summary': summary, 'accounting': accounting,
        'verdict_coverage': stats, 'disagreements': disagreements(rows, prep['family_order']),
        'artifacts': artifacts, 'status': result['status']})


def summarize(rows):
    import numpy as np
    result = {}
    for arm in sorted({r['arm'] for r in rows}):
        block = [r for r in rows if r['arm'] == arm]; n = len(block)
        details = {'assigned': n, 'execution_counts': dict(Counter(r['execution_status'] for r in block)),
                   'boundary_counts': dict(Counter(r['boundary_status'] for r in block))}
        for field, predicate in [('natural_stop', lambda r: r['finish'] == 'stop'),
            ('cap', lambda r: r['finish'] == 'length'), ('error', lambda r: r['execution_status'] == 'GENERATION_ERROR'),
            ('firing', lambda r: r['fired'] is True), ('reasoning_closure', lambda r: r['closure'] is True)]:
            details[field + '_count'] = sum(predicate(r) for r in block)
            details[field + '_rate_assigned_denominator'] = details[field + '_count'] / n
        details['unknown_correctness'] = sum(r['operational_correct'] is None for r in block)
        details['known_correct'] = sum(r['operational_correct'] is True for r in block)
        details['accuracy_bounds'] = [details['known_correct'] / n,
            (details['known_correct'] + details['unknown_correctness']) / n]
        for field in ('reasoning_tokens', 'answer_tokens', 'tokens'):
            values = [r[field] for r in block if r[field] is not None]
            details[field] = {'observed': len(values), 'mean': float(np.mean(values)) if values else None,
                             'median': float(np.median(values)) if values else None}
        details['side_work'] = {field: sum(r[field] or 0 for r in block) for field in
            ('side_generated_tokens', 'side_prompt_tokens', 'side_queries', 'side_seconds')}
        details['descriptive_conditioned_reasoning'] = {}
        for field in ('operational_correct', 'fired'):
            for flag in (True, False):
                values = [r['reasoning_tokens'] for r in block if r[field] is flag and r['reasoning_tokens'] is not None]
                details['descriptive_conditioned_reasoning'][field + '=' + str(flag)] = {
                    'count': len(values), 'mean': float(np.mean(values)) if values else None,
                    'interpretation': 'Descriptive; conditioning follows treatment.'}
        result[arm] = details
    return result


def disagreements(rows, families):
    lookup = {(r['family'], r['seed'], r['arm']): r for r in rows}; result = {}
    for arm in P.ARMS[1:]:
        counts = Counter()
        for family in families:
            for seed in (0, 1):
                left, right = lookup[family, seed, arm]['operational_correct'], lookup[family, seed, 'native']['operational_correct']
                counts['unknown' if left is None or right is None else f'{int(left)}:{int(right)}'] += 1
        result[arm] = dict(counts)
    return result


def write_tables_figures(rows, effect, out):
    fields = ['uid', 'family', 'seed', 'arm', 'execution_status', 'finish', 'operational_correct',
              'reasoning_tokens', 'answer_tokens', 'tokens', 'boundary_status', 'fired', 'side_queries',
              'side_prompt_tokens', 'side_generated_tokens', 'side_seconds']
    with (out / 'assignments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    with (out / 'effects.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['comparison', 'estimate', 'simultaneous_95_low',
            'simultaneous_95_high', 'missing_bound_low', 'missing_bound_high'])
        for i, name in enumerate(effect['comparison_order']):
            ci = effect['simultaneous_95_intervals'][i] if effect['simultaneous_95_intervals'] else [None, None]
            value = effect['point_estimates'][i] if effect['point_estimates'] else None
            writer.writerow([name, value, *ci, *effect['identification_bounds'][i]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, color in zip(P.ARMS[1:], ('tab:blue', 'tab:orange', 'tab:green')):
        i = P.ARMS[1:].index(arm) * 3
        for offset, ax in zip((1, 2), axes):
            if effect['point_estimates'] is not None:
                estimates = effect['point_estimates']; ax.plot(estimates[i + offset], estimates[i] * 100, 'o', color=color, label=arm)
                if effect['simultaneous_95_intervals']:
                    ci = effect['simultaneous_95_intervals']; x, y = ci[i + offset], ci[i]
                    ax.add_patch(Rectangle((x[0], 100 * y[0]), x[1] - x[0], 100 * (y[1] - y[0]), color=color, alpha=.15))
            else:
                ax.text(.5, .5, 'Incomplete endpoints: see identification bounds', ha='center', transform=ax.transAxes)
    for ax, label in zip(axes, ('Generated reasoning token change', 'Total generated token change')):
        ax.axhline(0, color='grey', linewidth=.7); ax.axvline(0, color='grey', linewidth=.7)
        ax.set(xlabel=label + ' (operator − native)', ylabel='Accuracy change (percentage points)')
        if effect['point_estimates'] is not None: ax.legend()
    fig.suptitle('16,384-token panel: simultaneous 95% intervals across nine effects')
    fig.tight_layout(); fig.savefig(out / 'accuracy_length.pdf'); fig.savefig(out / 'accuracy_length.svg'); plt.close(fig)
    return {p.name: P.U.file_sha(p) for p in out.iterdir() if p.suffix in ('.csv', '.svg', '.pdf')}

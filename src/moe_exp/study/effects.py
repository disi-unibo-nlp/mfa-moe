"""Intervention effects of the reduced design (CPU): lexical state endpoints, efficiency, inference.

Endpoints per continuation (window [pulse, H) of continuation tokens, H = 16·Q75, as `endpoints.occupancy`):
  occupancy of each class (hard lexical labels; soft = classifier probability mass), fraction ended,
  class of the first complete sentence, generated/total tokens, strict accuracy with unknown bounds.
Contrasts are paired within (model, family, class): targeted vs matched random (primary; deactivation sign
flipped so positive = intended), targeted vs native, random vs native. Pooled estimands weight models and
classes equally within a family and benchmarks equally across families. Inference resamples the shared
question families (10,000 multinomial draws, the same draws for every contrast) and reports percentile
intervals plus a max-|t| simultaneous envelope over each pre-declared family of contrasts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .common import BENCHMARKS, CLASSES, MODELS, digest, load, publish, require, verified
from .driver import DRIVER, ROOT, binding, bulk, log

PRIMARY = ('activate:first', 'activate:occupancy', 'deactivate:first', 'deactivate:occupancy')
SEMANTIC_PREFIXES = {
    'occupancy': 'occ_',
    'first': 'first_',
    'soft_occupancy': 'occs_',
    'pulse_occupancy': 'pulse_occ_',
    'full_occupancy': 'full_occ_',
    'first_post': 'first_post_',
    'first_overlap': 'first_overlap_',
    'semantic_first': 'semantic_first_',
}


def endpoints(outcome, sentences):
    """Return fixed-window point estimates and explicit missingness bounds.

    The original endpoint counted uncovered positions as zero and therefore
    gave an outcome with no sentence rows a perfect null.  The aliases
    ``occupancy_hard``/``first_hard`` are retained for existing consumers; the
    new ``*_lower`` and ``*_upper`` fields are the values used for inference.
    """
    pulse, horizon = int(outcome['pulse_tokens']), int(outcome['horizon'])
    end = outcome['reasoning_end']
    natural = outcome['generated_tokens'] if outcome['finish_reason'] == 'stop' else None
    ends = [x for x in (end, natural) if x is not None and x == x]
    stop = min(ends) if ends else None
    nonreasoning = set(json.loads(outcome['nonreasoning']) if isinstance(outcome['nonreasoning'], str)
                       else outcome['nonreasoning'])
    spans = sorted(sentences, key=lambda s: s['token_start'])

    def window(start, finish):
        n = max(0, finish-start)
        hard, soft = np.zeros(7), np.zeros(7)
        ended = unknown = covered = nonreasoning_count = 0
        j = 0
        for p in range(start, finish):
            if stop is not None and p >= stop:
                ended += 1
                continue
            if p in nonreasoning:
                nonreasoning_count += 1
                continue
            while j < len(spans) and spans[j]['token_end'] <= p:
                j += 1
            if j < len(spans) and spans[j]['token_start'] <= p:
                s = spans[j]
                hard[CLASSES.index(s['lexical_label'])] += 1
                soft += np.array([s['p_'+c] for c in CLASSES])
                covered += 1
            else:
                unknown += 1
        if not n:
            zero = np.zeros(7)
            return {'n': 0, 'hard': zero, 'soft': zero, 'lower': zero, 'upper': zero,
                    'soft_lower': zero, 'soft_upper': zero, 'ended': 0., 'unknown': 0.,
                    'covered': 0., 'nonreasoning': 0., 'available': False}
        require(int(hard.sum())+ended+unknown+nonreasoning_count == n, 'occupancy mass is not conserved')
        return {'n': n, 'hard': hard/n, 'soft': soft/n,
                'lower': hard/n, 'upper': (hard+unknown)/n,
                'soft_lower': soft/n, 'soft_upper': (soft+unknown)/n,
                'ended': ended/n, 'unknown': unknown/n, 'covered': covered/n,
                'nonreasoning': nonreasoning_count/n, 'available': True}

    post = window(pulse, horizon)
    full = window(0, horizon)
    pulse_window = window(0, pulse)
    complete = [s for s in spans if s['complete']]
    first = next((s for s in complete if not s.get('straddles_prefix', False)
                  and 0 <= s['token_start'] < horizon), None)
    first_post = next((s for s in complete if s['token_start'] >= pulse), None)
    first_overlap = next((s for s in spans
                          if s['token_start'] < pulse and s['token_end'] > 0), None)
    if first_overlap is None:
        first_overlap = first_post

    def first_values(sentence):
        if sentence is None:
            # A stopped or naturally terminated continuation has no observed
            # complete sentence; a capped/truncated one is unknown.
            known = stop is not None or natural is not None
            point = np.zeros(7)
            return {'point': point, 'hard': np.full(7, np.nan),
                    'lower': point, 'upper': point if known else np.ones(7),
                    'soft': point, 'soft_lower': point, 'soft_upper': point if known else np.ones(7),
                    'available': False}
        index = CLASSES.index(sentence['lexical_label'])
        point = np.eye(7)[index]
        probs = np.array([sentence['p_'+c] for c in CLASSES])
        return {'point': point, 'hard': point, 'lower': point, 'upper': point,
                'soft': probs, 'soft_lower': probs, 'soft_upper': probs, 'available': True}

    first_values_all = first_values(first)
    first_values_post = first_values(first_post)
    first_values_overlap = first_values(first_overlap)
    dyn = dynamics_metrics(spans)
    return {
        'occupancy_hard': post['hard'], 'occupancy_soft': post['soft'],
        'occupancy_lower': post['lower'], 'occupancy_upper': post['upper'],
        'occupancy_soft_lower': post['soft_lower'], 'occupancy_soft_upper': post['soft_upper'],
        'ended': post['ended'], 'uncovered': post['unknown'], 'covered': post['covered'],
        'available': post['available'],
        'pulse_occupancy_hard': pulse_window['hard'], 'pulse_occupancy_lower': pulse_window['lower'],
        'pulse_occupancy_upper': pulse_window['upper'], 'pulse_available': pulse_window['available'],
        'full_occupancy_hard': full['hard'], 'full_occupancy_lower': full['lower'],
        'full_occupancy_upper': full['upper'], 'full_ended': full['ended'], 'full_unknown': full['unknown'],
        'full_nonreasoning': full['nonreasoning'],
        'first_hard': first_values_all['hard'], 'first_lower': first_values_all['lower'],
        'first_upper': first_values_all['upper'], 'first_soft': first_values_all['soft'],
        'first_soft_lower': first_values_all['soft_lower'], 'first_soft_upper': first_values_all['soft_upper'],
        'first_available': first_values_all['available'],
        'first_post_hard': first_values_post['hard'], 'first_post_lower': first_values_post['lower'],
        'first_post_upper': first_values_post['upper'], 'first_post_available': first_values_post['available'],
        'first_overlap_hard': first_values_overlap['hard'], 'first_overlap_lower': first_values_overlap['lower'],
        'first_overlap_upper': first_values_overlap['upper'], 'first_overlap_available': first_values_overlap['available'],
        **{'dyn_'+k: v for k, v in dyn.items()},
    }


def _undefined_dynamics(reason: str) -> dict:
    """Leave trajectory metrics undefined when the observed sequence is incomplete."""
    metrics = ('nsent', 'ntrans', 'selfrate', 'clsdist', 'entropy', 'runmean', 'nruns',
               'target_first', 'target_occupancy_sent', 'target_persistence', 'target_entry',
               'target_exit', 'target_runmean', 'target_runs')
    return {**dict.fromkeys(metrics, np.nan), 'qualified': False, 'reason': reason}


def dynamics_metrics(sentences, target=None):
    """Trajectory descriptors and target-specific dynamics used as secondary outcomes."""
    complete = sorted(sentences, key=lambda s: s.get('sentence_index', 0))
    if not complete:
        return _undefined_dynamics('no_observed_sequence')
    if any(not sentence.get('complete') or sentence.get('lexical_label') not in CLASSES for sentence in complete):
        return _undefined_dynamics('incomplete_or_unlabelled_sentence')
    indices = [s.get('sentence_index', i) for i, s in enumerate(complete)]
    if any(second != first+1 for first, second in zip(indices, indices[1:])):
        return _undefined_dynamics('nonadjacent_sentence_indices')
    labels = [s['lexical_label'] for s in complete]
    pairs = [(a, b) for a, b, i, j in zip(labels, labels[1:], indices, indices[1:]) if j == i+1]
    runs = []
    start = 0
    for i in range(1, len(labels)+1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((labels[start], i-start))
            start = i
    counts = np.bincount([CLASSES.index(x) for x in labels], minlength=7)
    probs = counts/counts.sum()
    entropy = float(-(probs[probs > 0]*np.log(probs[probs > 0])).sum())
    value = {'qualified': True, 'reason': None, 'nsent': len(labels), 'ntrans': len(pairs),
            'selfrate': sum(a == b for a, b in pairs)/len(pairs) if pairs else np.nan,
            'clsdist': len(set(labels)), 'entropy': entropy,
            'runmean': float(np.mean([n for _, n in runs])), 'nruns': len(runs)}
    if target in CLASSES:
        target_pairs = [(a, b) for a, b in pairs if a == target]
        entry_pairs = [(a, b) for a, b in pairs if a != target]
        target_runs = [n for label, n in runs if label == target]
        value.update({
            'target_first': float(labels[0] == target),
            'target_occupancy_sent': float(np.mean([x == target for x in labels])),
            'target_persistence': (sum(b == target for _, b in target_pairs)/len(target_pairs)
                                   if target_pairs else np.nan),
            'target_entry': (sum(b == target for _, b in entry_pairs)/len(entry_pairs)
                             if entry_pairs else np.nan),
            'target_exit': (sum(b != target for _, b in target_pairs)/len(target_pairs)
                            if target_pairs else np.nan),
            'target_runmean': float(np.mean(target_runs)) if target_runs else 0.,
            'target_runs': len(target_runs),
        })
    return value


def analysis_assignments(model: str, design: str) -> list[dict]:
    """Return the frozen analysis population, including unavailable main prefixes.

    Args:
        model: Model owning the frozen assignments.
        design: Main, exploratory strong-dose, or rescue design.

    Returns:
        Original assignments; no new continuation or random draw is created.
    """
    from .driver import MANIFESTS
    manifest = verified(DRIVER/model/MANIFESTS[design])
    rows = list(manifest['records'])
    if design == 'main' and manifest.get('unavailable'):
        original = verified(ROOT/model/'ASSIGNMENTS-full.json')
        require(original['sha256'] == manifest['source_assignments_sha256'], 'original analysis assignments changed')
        unavailable = set(manifest['unavailable'])
        restored = [row for row in original['records'] if row['uid'] in unavailable]
        require({row['uid'] for row in restored} == unavailable, 'unavailable analysis assignments are absent')
        rows.extend(restored)
    if design == 'strong':
        families = {row['family'] for row in rows}
        rows = [{**row, 'arm': row['arm'].removeprefix('strong_')} for row in rows]
        main = analysis_assignments(model, 'main')
        rows.extend(row for row in main if row['arm'] == 'native' and row['family'] in families)
        rows = [{**row, 'pulse_tokens': 0} for row in rows]
    return rows


def missing_outcome(assignment: dict) -> dict:
    """Keep an unavailable assigned continuation unknown, without inventing token counts."""
    source = assignment.get('source') or {}
    action = (assignment.get('actions') or [assignment.get('action')])[0] or {}
    return {
        **assignment,
        'layer': action.get('layer'),
        'layer_regime': assignment.get('layer_regime'),
        'operator': action.get('operator', 'hard_select'),
        'generated_tokens': np.nan,
        'prefix_tokens': source.get('prefix_tokens', np.nan),
        'reasoning_tokens': np.nan,
        'reasoning_end': None,
        'finish_reason': 'unknown',
        'strict_correct': None,
        'accuracy_lower': 0.,
        'accuracy_upper': 1.,
        'segments': None,
        'nonreasoning': '[]',
        'outcome_status': ('unavailable_prefix' if assignment.get('prefix_status') != 'enrolled'
                           else 'outcome_unavailable'),
    }


def restore_outcomes(outcomes, assignments: list[dict]):
    """Join extraction to every planned UID and retain missing outcomes as bounds.

    Args:
        outcomes: Extracted outcomes for the specified design.
        assignments: Frozen planned rows, including unavailable prefixes.

    Returns:
        One outcome per assignment; missing generated-token counts remain NaN.
    """
    import pandas as pd
    expected = {row['uid']: row for row in assignments}
    require(len(expected) == len(assignments), 'duplicate planned continuation UID')
    observed = {row['uid']: row for row in outcomes.to_dict('records')}
    require(len(observed) == len(outcomes), 'duplicate extracted continuation UID')
    require(set(observed) <= set(expected), 'extracted outcome is outside its frozen assignments')
    rows = []
    for uid, assignment in expected.items():
        if uid not in observed:
            rows.append(missing_outcome(assignment))
            continue
        outcome = observed[uid]
        for key in ('model', 'family', 'benchmark', 'fold', 'arm', 'class'):
            actual = outcome[key]
            require((pd.isna(actual) and assignment[key] is None) or actual == assignment[key],
                    f'{uid}: extracted {key} differs from its assignment')
        rows.append({**outcome, 'outcome_status': 'observed'})
    return pd.DataFrame(rows)


def _read_outcome_table(folder: Path, kind: str, design: str):
    """Read one design's shards, retaining its original native comparison."""
    import pandas as pd
    pattern = kind+'-[0-9]*.parquet' if design != 'rescue' else kind+'-rescue-*.parquet'
    frames = [pd.read_parquet(path) for path in sorted(folder.glob(pattern))]
    if design == 'rescue':
        require(frames, f'{folder}: rescue {kind} table is absent')
    main = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if design != 'strong':
        return main
    frames = [pd.read_parquet(path) for path in sorted(folder.glob(kind+'-strong-*.parquet'))]
    strong = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not main.empty:
        main = main[main.arm == 'native']
    if not strong.empty:
        strong['arm'] = strong.arm.str.replace('strong_', '', regex=False)
    return pd.concat([main, strong], ignore_index=True)


def continuation_table(models, design='main', *, cohort=None, classifier=None, closeout=False):
    """design 'strong': post-hoc strong-dose arms (renamed to target/random_activate) plus the main-design native rows."""
    import pandas as pd
    frames = []
    cohort_value, semantic, cohort_tables = None, {}, {}
    if design == 'rescue':
        require(cohort is not None, 'rescue analysis requires --cohort')
        from .cohort import read
        from .rescue_reader import semantic_labels, verified_tasks
        cohort_value = read(Path(cohort))
        task_path = Path(cohort_value['paths']['control'])/'READER_TASKS.json'
        if task_path.exists() or not closeout:
            semantic = semantic_labels(cohort_value, verified_tasks(cohort_value))
        extraction_ready = all((Path(cohort_value['paths']['extraction'])/model/'EXTRACTION.json').exists() for model in models)
        cohort_tables = (_cohort_tables(cohort_value, models) if extraction_ready or not closeout else
                         {model: (pd.DataFrame(columns=['uid']), pd.DataFrame()) for model in models})
    for model in models:
        if cohort_value is not None:
            from .cohort import assignments as cohort_assignments
            assignments = cohort_assignments(cohort_value, model)
        else:
            assignments = analysis_assignments(model, design)
        folder = bulk()/model/'outcomes'
        outcomes, sentences = (cohort_tables[model] if cohort_value is not None else
                              (_read_outcome_table(folder, 'outcomes', design),
                               _read_outcome_table(folder, 'sentences', design)))
        if design == 'strong':  # routing forced over the whole window: measure [0, H) for strong and native rows
            outcomes = outcomes.assign(pulse_tokens=0)
        outcomes = restore_outcomes(outcomes, assignments)
        if not sentences.empty:
            from .lexical import LEXICAL, predict
            readout = cohort_value['classifier'] if cohort_value is not None else classifier
            sentences = predict(sentences, Path(readout['root']) if readout else LEXICAL/'readout-v1', frozen=readout)
        groups = {u: g.to_dict('records') for u, g in sentences.groupby('uid')} if not sentences.empty else {}
        rows = []
        for o in outcomes.to_dict('records'):
            sentences_for_uid = groups.get(o['uid'], [])
            e = endpoints(o, sentences_for_uid)
            row = {k: o[k] for k in ('uid', 'model', 'arm', 'class', 'family', 'fold', 'benchmark', 'layer',
                                      'generated_tokens', 'prefix_tokens', 'reasoning_tokens', 'finish_reason',
                                      'strict_correct', 'accuracy_lower', 'accuracy_upper', 'segments', 'outcome_status')}
            row.update({key: o[key] for key in ('layer_regime', 'operator') if key in o})
            row.update(ended=e['ended'], uncovered=e['uncovered'], covered=e['covered'],
                       available=e['available'], pulse_available=e['pulse_available'])
            row.update(full_ended=e['full_ended'], full_unknown=e['full_unknown'], full_nonreasoning=e['full_nonreasoning'])
            semantic_bounds = semantic_endpoint(semantic.get(o['uid'], {'status': 'unknown'}))
            if o['outcome_status'] != 'observed':
                row['ended'] = np.nan
            for i, c in enumerate(CLASSES):
                row['occ_'+c], row['occs_'+c] = e['occupancy_hard'][i], e['occupancy_soft'][i]
                row['occ_lower_'+c], row['occ_upper_'+c] = e['occupancy_lower'][i], e['occupancy_upper'][i]
                row['occs_lower_'+c], row['occs_upper_'+c] = e['occupancy_soft_lower'][i], e['occupancy_soft_upper'][i]
                row['pulse_occ_'+c] = e['pulse_occupancy_hard'][i]
                row['pulse_occ_lower_'+c], row['pulse_occ_upper_'+c] = e['pulse_occupancy_lower'][i], e['pulse_occupancy_upper'][i]
                row['full_occ_'+c] = e['full_occupancy_hard'][i]
                row['full_occ_lower_'+c], row['full_occ_upper_'+c] = e['full_occupancy_lower'][i], e['full_occupancy_upper'][i]
                row['first_'+c] = np.nan_to_num(e['first_hard'][i], nan=0.)
                row['firsts_'+c] = np.nan_to_num(e['first_soft'][i], nan=0.)
                row['first_lower_'+c], row['first_upper_'+c] = e['first_lower'][i], e['first_upper'][i]
                row['first_post_'+c], row['first_post_lower_'+c], row['first_post_upper_'+c] = (
                    np.nan_to_num(e['first_post_hard'][i], nan=0.), e['first_post_lower'][i], e['first_post_upper'][i])
                row['first_overlap_'+c], row['first_overlap_lower_'+c], row['first_overlap_upper_'+c] = (
                    np.nan_to_num(e['first_overlap_hard'][i], nan=0.), e['first_overlap_lower'][i], e['first_overlap_upper'][i])
                row['semantic_first_'+c] = float((semantic_bounds[0][i]+semantic_bounds[1][i])/2)
                row['semantic_first_lower_'+c] = semantic_bounds[0][i]
                row['semantic_first_upper_'+c] = semantic_bounds[1][i]
            for available, prefixes in ((e['available'], ('occ_', 'occs_')),
                                        (e['pulse_available'], ('pulse_occ_',))):
                if not available:
                    row.update({key: np.nan for key in row if key.startswith(prefixes)})
            row.update({k: e[k] for k in e if k.startswith('dyn_')})
            dyn = dynamics_metrics(sentences_for_uid, o.get('class'))
            row.update({'dyn_'+k: v for k, v in dyn.items()})
            if o['outcome_status'] != 'observed':
                row.update({key: np.nan for key in row if key.startswith('dyn_')})
            rows.append(row)
        frames.append(pd.DataFrame(rows))
        log({'event': 'continuations', 'model': model, 'rows': len(rows), 'sentences': len(sentences)})
    return pd.concat(frames, ignore_index=True)


def semantic_endpoint(annotation: dict) -> tuple:
    """Distinguish observed reader labels, genuine no-event and unknown outcomes.

    Args:
        annotation: Exact assignment/context join from the reader inventory.

    Returns:
        Seven class-specific lower and upper outcome vectors.
    """
    if annotation['status'] == 'no_event':
        return np.zeros(7), np.zeros(7)
    if annotation['status'] == 'reader_task' and annotation.get('label') in CLASSES:
        observed = np.eye(7)[CLASSES.index(annotation['label'])]
        return observed, observed
    return np.zeros(7), np.ones(7)


def _cohort_tables(value: dict, models: list[str]) -> dict:
    """Adapt verified terminal replays to the maintained effect-table interface."""
    import pandas as pd
    from .cohort import assignments, generation_index
    from .common import check_files
    index = generation_index(value)
    tables = {}
    for model in models:
        folder = Path(value['paths']['extraction'])/model
        receipt = verified(folder/'EXTRACTION.json')
        require(receipt['cohort_sha256'] == value['sha256']
                and index['extraction_sha256'][model] == receipt['sha256'], 'analysis extraction differs')
        check_files(receipt['files'])
        roster = {row['uid']: row for row in assignments(value, model)}
        outcomes, sentences = [], []
        for replay in pd.read_parquet(folder/'terminal-replay.parquet').to_dict('records'):
            row = roster[replay['uid']]
            record = json.loads(replay['replay_json'])
            action = (row.get('actions') or [row.get('action')])[0] or {}
            outcomes.append({
                **row, 'layer': action.get('layer'), 'operator': action.get('operator', 'native'),
                **{key: record[key] for key in ('generated_tokens', 'prefix_tokens', 'reasoning_tokens',
                   'reasoning_end', 'finish_reason', 'strict_correct')},
                'accuracy_lower': record['accuracy_bounds'][0], 'accuracy_upper': record['accuracy_bounds'][1],
                'segments': 1, 'nonreasoning': json.dumps(record['nonreasoning']),
            })
            for sentence in record['sentences']:
                sentences.append({
                    **{key: row[key] for key in ('uid', 'model', 'arm', 'class', 'family', 'fold')},
                    **{key: sentence[key] for key in ('sentence_index', 'token_start', 'token_end', 'complete',
                                                     'first_complete', 'straddles_prefix')},
                    'sentence_text': sentence['inputs']['sentence'],
                    'prev_sentence': sentence['inputs']['previous_sentence'] or '<START OF RESPONSE>',
                    'next_sentence': sentence['inputs']['next_sentence'] or '<END OF RESPONSE>',
                })
        tables[model] = (pd.DataFrame(outcomes, columns=list(outcomes[0]) if outcomes else ['uid']), pd.DataFrame(sentences))
    return tables


def paired(table):
    """One row per (model, family, class, layer regime) of paired contrasts."""
    import pandas as pd
    t = table.copy()
    t['layer_regime'] = t['layer_regime'].fillna('central') if 'layer_regime' in t else 'central'
    t['log_tokens'] = np.log(t.generated_tokens.clip(lower=1))
    t['log_total'] = np.log((t.generated_tokens+t.prefix_tokens).clip(lower=1))
    t['log_reasoning'] = np.log(t.reasoning_tokens.clip(lower=1))
    t['correct'] = t.strict_correct.astype(float)
    t['correct'] = t['correct'].fillna((t.accuracy_lower+t.accuracy_upper)/2)
    t['correct_lower'] = t.accuracy_lower.astype(float)
    t['correct_upper'] = t.accuracy_upper.astype(float)
    native = t[t.arm == 'native'].set_index(['model', 'family', 'layer_regime'])
    require(native.index.is_unique, 'native rescue/main rows are not unique')
    rows = []
    observed = t[~t.arm.isin(('native', 'native_repeat'))]
    for (model, family, c, regime), g in observed.groupby(['model', 'family', 'class', 'layer_regime']):
        require(g.arm.is_unique, f'{model}/{family}/{c}/{regime}: duplicate intervention arm')
        arms = {r['arm']: r for r in g.to_dict('records')}
        if (model, family, regime) in native.index:
            nat = native.loc[(model, family, regime)].to_dict()
        elif (model, family, 'native') in native.index:
            nat = native.loc[(model, family, 'native')].to_dict()
        else:
            nat = None
        out = {'model': model, 'family': family, 'class': c, 'layer_regime': regime,
               'benchmark': g.benchmark.iloc[0], 'layer': g.layer.iloc[0]}
        for direction, sign in (('activate', 1), ('deactivate', -1)):
            ta, ra = arms.get('target_'+direction), arms.get('random_'+direction)
            for kind, base in SEMANTIC_PREFIXES.items():
                if not ta or not ra or base+c not in ta or base+c not in ra:
                    out[f'{direction}:{kind}'] = np.nan
                    out[f'{direction}:{kind}:lower'] = np.nan
                    out[f'{direction}:{kind}:upper'] = np.nan
                    out[f'{direction}:{kind}:vs_native'] = np.nan
                    out[f'{direction}:{kind}:vs_native:lower'] = np.nan
                    out[f'{direction}:{kind}:vs_native:upper'] = np.nan
                    continue
                lower_key, upper_key = base+'lower_', base+'upper_'
                # Reversing the contrast also reverses its interval endpoints.
                if sign < 0:
                    lower_key, upper_key = upper_key, lower_key
                out[f'{direction}:{kind}'] = sign*(ta[base+c]-ra[base+c])
                out[f'{direction}:{kind}:lower'] = sign*(ta[lower_key+c]-ra[upper_key+c])
                out[f'{direction}:{kind}:upper'] = sign*(ta[upper_key+c]-ra[lower_key+c])
                out[f'{direction}:{kind}'] = (out[f'{direction}:{kind}:lower']+out[f'{direction}:{kind}:upper'])/2
                if nat:
                    out[f'{direction}:{kind}:vs_native'] = sign*(ta[base+c]-nat[base+c])
                    out[f'{direction}:{kind}:vs_native:lower'] = sign*(ta[lower_key+c]-nat[upper_key+c])
                    out[f'{direction}:{kind}:vs_native:upper'] = sign*(ta[upper_key+c]-nat[lower_key+c])
                else:
                    out[f'{direction}:{kind}:vs_native'] = np.nan
                    out[f'{direction}:{kind}:vs_native:lower'] = np.nan
                    out[f'{direction}:{kind}:vs_native:upper'] = np.nan
            for metric in ('log_tokens', 'log_total', 'log_reasoning', 'correct', 'correct_lower',
                           'correct_upper', 'generated_tokens', 'ended'):
                if ta and ra:
                    if metric == 'correct_lower':
                        out[f'{direction}:{metric}'] = ta['accuracy_lower']-ra['accuracy_upper']
                        out[f'{direction}:{metric}:vs_native'] = ta['accuracy_lower']-nat['accuracy_upper'] if nat else np.nan
                        out[f'random_{direction}:{metric}:vs_native'] = ra['accuracy_lower']-nat['accuracy_upper'] if nat else np.nan
                    elif metric == 'correct_upper':
                        out[f'{direction}:{metric}'] = ta['accuracy_upper']-ra['accuracy_lower']
                        out[f'{direction}:{metric}:vs_native'] = ta['accuracy_upper']-nat['accuracy_lower'] if nat else np.nan
                        out[f'random_{direction}:{metric}:vs_native'] = ra['accuracy_upper']-nat['accuracy_lower'] if nat else np.nan
                    else:
                        out[f'{direction}:{metric}'] = ta[metric]-ra[metric]
                        out[f'{direction}:{metric}:vs_native'] = ta[metric]-nat[metric] if nat else np.nan
                        out[f'random_{direction}:{metric}:vs_native'] = ra[metric]-nat[metric] if nat else np.nan
                else:
                    out[f'{direction}:{metric}'] = np.nan
                    out[f'{direction}:{metric}:vs_native'] = np.nan
                    out[f'random_{direction}:{metric}:vs_native'] = np.nan
            for comparison in (direction, 'random_'+direction):
                for suffix in ('', ':vs_native'):
                    if comparison.startswith('random_') and not suffix:
                        continue
                    for bound in ('lower', 'upper'):
                        out[f'{comparison}:correct{suffix}:{bound}'] = out[f'{comparison}:correct_{bound}{suffix}']
            for metric in ('dyn_selfrate', 'dyn_clsdist', 'dyn_entropy', 'dyn_target_first',
                           'dyn_target_occupancy_sent', 'dyn_target_persistence', 'dyn_target_entry',
                           'dyn_target_exit', 'dyn_target_runmean', 'dyn_target_runs'):
                if ta and ra and metric in ta and metric in ra:
                    out[f'{direction}:{metric}'] = sign*(ta[metric]-ra[metric])
                    out[f'{direction}:{metric}:vs_native'] = sign*(ta[metric]-nat[metric]) if nat else np.nan
                else:
                    out[f'{direction}:{metric}'] = np.nan
                    out[f'{direction}:{metric}:vs_native'] = np.nan
            for c2 in CLASSES:  # response matrix: targeted vs random on every observed class
                if ta and ra:
                    prefix = 'full_occ_' if ta.get('operator') == 'score_override' else 'occ_'
                    out[f'{direction}:response:{c2}'] = ta[prefix+c2]-ra[prefix+c2]
                    out[f'{direction}:response:{c2}:lower'] = ta[prefix+'lower_'+c2]-ra[prefix+'upper_'+c2]
                    out[f'{direction}:response:{c2}:upper'] = ta[prefix+'upper_'+c2]-ra[prefix+'lower_'+c2]
                    out[f'{direction}:response:{c2}'] = (out[f'{direction}:response:{c2}:lower']+
                                                          out[f'{direction}:response:{c2}:upper'])/2
                else:
                    out[f'{direction}:response:{c2}'] = np.nan
                    out[f'{direction}:response:{c2}:lower'] = np.nan
                    out[f'{direction}:response:{c2}:upper'] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


class Bootstrap:
    """Shared family draws within benchmarks; benchmark-balanced family-level means."""
    def __init__(self, families, benchmarks, draws=10000, seed=20261015):
        self.families = list(families)
        self.index = {f: i for i, f in enumerate(self.families)}
        self.benchmark = np.array([benchmarks[f] for f in self.families])
        require(set(self.benchmark) <= set(BENCHMARKS), 'unknown bootstrap benchmark')
        rng = np.random.default_rng(seed)
        self.weights = np.zeros((draws, len(self.families)), dtype=int)
        # Each declared benchmark retains its family count in every shared
        # replicate, including a benchmark represented by a single family.
        for benchmark in BENCHMARKS:
            indices = np.flatnonzero(self.benchmark == benchmark)
            if len(indices):
                self.weights[:, indices] = rng.multinomial(
                    len(indices), np.full(len(indices), 1/len(indices)), size=draws)

    def estimate(self, frame, column):
        """frame rows (family, value); value averaged within family first (equal model/class weights)."""
        sub = frame[['family', column]]
        if sub.empty:
            return None
        require(set(sub.family) <= set(self.index), 'analysis family is outside the bootstrap population')
        missing = int((~np.isfinite(sub[column].to_numpy(dtype=float))).sum())
        counts = {
            'families': int(sub.family.nunique()),
            'cells': int(len(sub)),
            'benchmarks': len({self.benchmark[self.index[f]] for f in sub.family}),
            'missing_cells': missing,
        }
        if missing:
            return {'status': 'unqualified_missing_outcomes', **counts,
                    'reason': 'Unbounded or undefined outcomes remain; no complete-case interval is reported.'}
        means = sub.groupby('family')[column].mean()
        v = np.full(len(self.families), np.nan)
        for f, x in means.items():
            v[self.index[f]] = x
        point, boot, used = 0., 0., 0
        for b in BENCHMARKS:
            ids = np.where((self.benchmark == b) & ~np.isnan(v))[0]
            if not len(ids):
                continue
            point += v[ids].mean()
            w = self.weights[:, ids]
            denominator = w.sum(1)
            if np.any(denominator == 0):
                return {'status': 'unqualified_sparse_benchmark', **counts,
                        'reason': 'The comparison omits planned families and has empty benchmark replicates.'}
            boot = boot+(w*v[ids]).sum(1)/denominator
            used += 1
        point, boot = point/used, boot/used
        return {'status': 'qualified', 'estimate': float(point), 'se': float(boot.std(ddof=1)),
                'ci': np.quantile(boot, [.025, .975]).tolist(), **counts, 'draws': boot}


def summarize(boot, frame, columns, family_name, *, column_masks=None):
    """Joint intervals, including available bounds; undefined metrics stay unqualified."""
    found, out, unknown_cells = {}, {}, {}
    for column in columns:
        selected = frame.loc[column_masks[column]] if column_masks else frame
        point = boot.estimate(selected, column)
        if point is None:
            continue
        if point['status'] != 'qualified':
            out[column] = point
            continue
        values = [point]
        if column+':lower' in selected and column+':upper' in selected:
            values.extend(boot.estimate(selected, column+':'+bound) for bound in ('lower', 'upper'))
            unknown_cells[column] = int((selected[column+':upper'] > selected[column+':lower']).sum())
        if any(value['status'] != 'qualified' for value in values):
            out[column] = {'status': 'unqualified_missing_bounds', 'family_of_contrasts': family_name}
        else:
            found[column] = values
    if not found:
        return out
    t = np.stack([np.abs(v['draws']-v['estimate'])/max(v['se'], 1e-12)
                  for values in found.values() for v in values], 1)
    critical = float(np.quantile(t.max(1), .95))
    for c, values in found.items():
        v = values[0]
        out[c] = {k: v[k] for k in ('status', 'estimate', 'se', 'ci', 'families', 'cells', 'benchmarks', 'missing_cells')}
        out[c]['simultaneous'] = [v['estimate']-critical*v['se'], v['estimate']+critical*v['se']]
        out[c]['critical'] = critical
        out[c]['family_of_contrasts'] = family_name
        if len(values) == 3:
            lower, upper = values[1:]
            out[c].update(
                lower_bound=lower['estimate'], upper_bound=upper['estimate'],
                bound_ci=[lower['ci'][0], upper['ci'][1]],
                bound_simultaneous=[lower['estimate']-critical*lower['se'],
                                    upper['estimate']+critical*upper['se']],
                bound_components=3, unknown_cells=unknown_cells[c],
                point_identified=unknown_cells[c] == 0,
                point_role='compatibility_only' if unknown_cells[c] else 'observed_endpoint')
            if unknown_cells[c]:
                out[c]['status'] = 'qualified_bounds'
    return out


def bounded_summary(boot, frame, names, family_name):
    """Joint point and missingness-bound inference for named contrasts.

    Each name has point, lower-bound and upper-bound columns.  The critical
    value is shared across all three components, so an unknown outcome widens
    the interval rather than becoming a zero contrast.
    """
    for name in names:
        require(name+':lower' in frame and name+':upper' in frame, f'{name}: outcome bounds are absent')
    return summarize(boot, frame, names, family_name)


def arm_levels(boot, table, model):
    """Benchmark-balanced family means of accuracy and token counts for each arm of one model, with a
    simultaneous envelope over all its arm/metric cells (descriptive levels behind the paired contrasts)."""
    t = table.assign(correct=table.strict_correct.astype(float), total_tokens=table.generated_tokens+table.prefix_tokens,
                     stopped=(table.finish_reason == 'stop').astype(float))
    t['correct'] = t.correct.fillna((t.accuracy_lower+t.accuracy_upper)/2)
    t.loc[t.outcome_status != 'observed', 'stopped'] = np.nan
    columns, masks = [], {}
    for arm in sorted(t.arm.unique()):
        for metric in ('correct', 'reasoning_tokens', 'generated_tokens', 'total_tokens', 'stopped'):
            t[f'{arm}:{metric}'] = t[metric].where(t.arm == arm)
            columns.append(f'{arm}:{metric}')
            masks[f'{arm}:{metric}'] = t.arm == arm
        t[f'{arm}:correct:lower'] = t.accuracy_lower
        t[f'{arm}:correct:upper'] = t.accuracy_upper
    return summarize(boot, t, columns, f'levels:{model}', column_masks=masks)


def _analysis_metadata(models: list[str], design: str, table, pairs) -> dict:
    """Describe the planned population without conflating lexical and reader evidence."""
    result = {
        'models': models,
        'design': design,
        'continuations': int(len(table)),
        'pairs': int(len(pairs)),
        'readout': 'lexical-readout-v1 (cross-fitted; no judge on intervention text)',
        'analysis_version': 'planned-denominators-v1',
        'bootstrap': 'shared family draws stratified by the frozen benchmark roster',
    }
    result['availability'] = [
        {'model': model, 'benchmark': benchmark, 'status': status, 'continuations': int(count)}
        for (model, benchmark, status), count in table.groupby(['model', 'benchmark', 'outcome_status']).size().items()
    ]
    return result


def _qualified_efficiency_link(prereg: Path, pairs, boot) -> dict:
    """Keep the fixed correlation test unqualified when its outcomes remain unknown."""
    frozen = verified(prereg)
    incomplete = (not np.isfinite(pairs['activate:log_reasoning']).all() or
                  (pairs['activate:correct:lower'] != pairs['activate:correct:upper']).any())
    result = ({'status': 'unqualified_missing_outcomes',
               'reason': 'The preregistered link cannot omit unavailable families or impute censored accuracy.'}
              if incomplete else efficiency_link(frozen, pairs, boot))
    result['prereg_sha256'] = frozen['sha256']
    return result


def _rescue_summary(cohort_path: Path, primary_columns: list[str], pairs, boot, *, closeout=False) -> dict:
    """Join semantic evidence, audit diagnostics and the frozen six-contrast family."""
    result = {}
    from .cohort import read
    from .rescue_reader import verified_tasks, context_labels
    from .measurement import disagreement
    value = read(cohort_path)
    if closeout and not (Path(value['paths']['control'])/'READER_TASKS.json').exists():
        names = primary_columns+['activate:semantic_first', 'deactivate:semantic_first']
        return {
            'cohort_sha256': value['sha256'],
            'joint_six_contrasts': bounded_summary(boot, pairs, names, 'joint-6'),
            'reader_disagreement': {'status': 'unavailable', 'reason': 'reader task preparation did not complete'},
            'conclusion': {'category': 'bounded_null_or_inconclusive', 'equivalence_established': False,
                           'reason': 'unqualified or unavailable evidence remains unknown in the planned population'},
        }
    tasks = verified_tasks(value)
    semantic_columns = ['activate:semantic_first', 'deactivate:semantic_first']
    result['joint_six_contrasts'] = bounded_summary(boot, pairs, primary_columns+semantic_columns, 'joint-6')
    result['cohort_sha256'] = value['sha256']
    result['semantic_endpoint_frozen'] = tasks['semantic_endpoint_frozen']
    result['readout'] = 'frozen C10 lexical trajectories and exact-context semantic first-complete labels'
    result['reader_disagreement'] = disagreement(tasks['audit'], context_labels(value, tasks), boot)
    result['null_diagnostic'] = null_diagnostic()
    semantic_support = [name for name in semantic_columns
                        if (result['joint_six_contrasts'].get(name, {}).get('bound_simultaneous', [-1, 1])[0] > 0
                            or result['joint_six_contrasts'].get(name, {}).get('bound_simultaneous', [-1, 1])[1] < 0)]
    result['conclusion'] = {
        'category': 'reader_defined_steering' if semantic_support else 'bounded_null_or_inconclusive',
        'supported_semantic_contrasts': semantic_support,
        'semantic_directions': {name: ('intended' if result['joint_six_contrasts'][name]['bound_simultaneous'][0] > 0
                                        else 'opposite') for name in semantic_support},
        'equivalence_established': False,
        'measurement_disagreement': 'see probability-sampled HT estimates and their separate uncertainty components',
        'delivered_intervention_and_generic_disruption': 'assess online dose diagnostics with native/random accuracy and token costs',
    }
    return result


def analyze(models, out, design='main', *, cohort=None, classifier=None, closeout=False):
    """Write fresh effect artifacts while retaining planned unknown outcomes."""
    import pandas as pd
    require(not any((out/name).exists() for name in ('EFFECTS.json', 'continuations.parquet', 'pairs.parquet')),
            'corrected analysis requires a fresh output directory; preserve historical results')
    allocation = load(ROOT/'ALLOCATION.json')['records']
    benchmarks = {r['family']: r['benchmark'] for r in allocation}
    table = continuation_table(models, design, cohort=cohort, classifier=classifier, closeout=closeout)
    pairs = paired(table)
    boot = Bootstrap(sorted(table.family.unique()), benchmarks)
    primary_columns = (list(PRIMARY) if design != 'rescue' else
                       ['activate:full_occupancy', 'deactivate:full_occupancy',
                        'activate:first_overlap', 'deactivate:first_overlap'])
    result = _analysis_metadata(models, design, table, pairs)
    result['primary_columns'] = primary_columns
    result['primary_pooled'] = bounded_summary(boot, pairs, primary_columns, 'primary-4')
    if design == 'rescue':
        result.update(_rescue_summary(Path(cohort), primary_columns, pairs, boot, closeout=closeout))
    result['primary_by_model'] = {m: bounded_summary(boot, pairs[pairs.model == m], primary_columns, f'primary-4:{m}')
                                  for m in models}
    result['primary_by_class'] = {c: bounded_summary(boot, pairs[pairs['class'] == c], primary_columns, f'primary-4:{c}')
                                  for c in CLASSES}
    result['primary_by_regime'] = {
        str(regime): bounded_summary(boot, pairs[pairs['layer_regime'] == regime], primary_columns,
                                     'primary-4:'+str(regime))
        for regime in sorted(pairs.layer_regime.unique())}
    metrics = ('log_tokens', 'log_total', 'log_reasoning', 'correct', 'ended')
    efficiency = [f'{d}:{m}' for d in ('activate', 'deactivate') for m in metrics]
    efficiency += [f'{d}:{m}:vs_native' for d in ('activate', 'deactivate') for m in metrics]
    efficiency += [f'random_{d}:{m}:vs_native' for d in ('activate', 'deactivate') for m in metrics]
    result['efficiency_pooled'] = summarize(boot, pairs, efficiency, 'efficiency')
    result['efficiency_by_class'] = {c: summarize(boot, pairs[pairs['class'] == c], efficiency, f'efficiency:{c}')
                                     for c in CLASSES}
    result['efficiency_by_model'] = {m: summarize(boot, pairs[pairs.model == m], efficiency, f'efficiency:{m}')
                                     for m in models}
    result['accuracy_pooled'] = {key: value for key, value in result['efficiency_pooled'].items() if ':correct' in key}
    result['levels_by_model'] = {m: arm_levels(boot, table[table.model == m], m) for m in models}
    result['efficiency_by_model_class'] = {f'{m}|{c}': summarize(boot, pairs[(pairs.model == m) & (pairs['class'] == c)],
                                           ['activate:log_tokens', 'activate:correct', 'activate:log_tokens:vs_native',
                                            'activate:correct:vs_native'], f'efficiency:{m}:{c}')
                                           for m in models for c in CLASSES}
    secondary = [f'{d}:{k}' for d in ('activate', 'deactivate') for k in ('soft_occupancy', 'occupancy:vs_native',
                                                                            'first:vs_native')]
    result['secondary_pooled'] = summarize(boot, pairs, secondary, 'secondary')
    bounded = [f'{d}:{k}' for d in ('activate', 'deactivate')
               for k in ('pulse_occupancy', 'full_occupancy', 'first_post', 'first_overlap')]
    result['bounded_secondary'] = bounded_summary(boot, pairs, bounded, 'bounded-secondary')
    result['dynamics_pooled'] = summarize(
        boot, pairs, [f'{d}:dyn_{k}' for d in ('activate', 'deactivate') for k in
                      ('selfrate', 'clsdist', 'entropy', 'target_first', 'target_occupancy_sent',
                       'target_persistence', 'target_entry', 'target_exit', 'target_runmean', 'target_runs')],
        'dynamics')
    result['response_matrix'] = {c: {c2: (boot.estimate(pairs[pairs['class'] == c], f'activate:response:{c2}') or {}).get('estimate')
                                     for c2 in CLASSES} for c in CLASSES}
    result['response_matrix_bounds'] = {
        c: {c2: {k: (boot.estimate(pairs[pairs['class'] == c], f'activate:response:{c2}:{k}') or {}).get('estimate')
                 for k in ('lower', 'upper')} | {
            'estimate': (boot.estimate(pairs[pairs['class'] == c], f'activate:response:{c2}') or {}).get('estimate')}
            for c2 in CLASSES} for c in CLASSES}
    result['directional_response_matrices'] = {
        direction: {target: bounded_summary(boot, pairs[pairs['class'] == target],
                    [f'{direction}:response:{observed}' for observed in CLASSES],
                    f'response:{direction}:{target}') for target in CLASSES}
        for direction in ('activate', 'deactivate')}
    result['rogan_gladen_sensitivity'] = {
        'status': 'unqualified_calibration_uncertainty',
        'reason': 'Model/class calibration and joint family-bootstrap uncertainty in J remain required.',
    }
    prereg = DRIVER/'EFFICIENCY_LINK_PREREG.json'
    if prereg.exists() and design == 'main':  # the pre-registration excludes other strengths
        result['efficiency_link'] = _qualified_efficiency_link(prereg, pairs, boot)
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out/'continuations.parquet', index=False)
    pairs.to_parquet(out/'pairs.parquet', index=False)
    return publish(out/'EFFECTS.json', result)


def null_diagnostic(repetitions: int = 100) -> dict:
    """Report a bounded fixed-seed six-contrast null simulation diagnostic.

    Args:
        repetitions: Fixed finite number of synthetic panels.

    Returns:
        Observed diagnostic coverage, never a universal coverage guarantee.
    """
    import pandas as pd
    rng = np.random.default_rng(20261020)
    families = [str(index) for index in range(64)]
    benchmarks = {family: BENCHMARKS[index % len(BENCHMARKS)] for index, family in enumerate(families)}
    boot = Bootstrap(families, benchmarks, draws=1000)
    covered = 0
    for _ in range(repetitions):
        noise = rng.normal(size=(64, 6))
        frame = pd.DataFrame({'family': families})
        for index in range(6):
            frame[str(index)] = noise[:, index]
            frame[str(index)+':lower'] = noise[:, index]-.1
            frame[str(index)+':upper'] = noise[:, index]+.1
        summary = bounded_summary(boot, frame, list(map(str, range(6))), 'synthetic-null-6')
        covered += all(row['bound_simultaneous'][0] <= 0 <= row['bound_simultaneous'][1] for row in summary.values())
    return {'seed': 20261020, 'panels': repetitions, 'bootstrap_draws_per_panel': 1000,
            'joint_zero_coverage': covered/repetitions, 'role': 'bounded diagnostic, not a coverage guarantee'}


def efficiency_link(prereg, pairs, boot):
    """The pre-registered test (EFFICIENCY_LINK_PREREG.json 'test'): frozen beta(m,c) vs measured Delta(m,c) =
    family mean of targeted-minus-random activation (log reasoning tokens; strict accuracy). Spearman rho and the
    sign-agreement rate on signed cells, each with a 95% percentile interval from a family-cluster bootstrap of
    the intervention side (2,000 draws, seed 20261017, betas fixed), plus an exact binomial test. Secondary:
    tok_total / acc_excl_trunc / between-question (dataset_fe) predictors and the sign-robust restricted rho."""
    from scipy.stats import binomtest, spearmanr
    link = Bootstrap(boot.families, dict(zip(boot.families, boot.benchmark)), draws=2000, seed=20261017)

    def test(cells, outcome, column, restrict=False):
        beta = {(r['model'], r['class']): r[outcome] for r in cells if r.get(outcome) and r[outcome].get('beta') is not None}
        if restrict:
            beta = {k: v for k, v in beta.items() if v.get('sign') in ('+', '-') and not v.get('opposite_sign_in')}
        means = {}
        for k in sorted(beta):
            sub = pairs[(pairs.model == k[0]) & (pairs['class'] == k[1])][['family', column]].dropna()
            if len(sub):
                means[k] = sub.groupby('family')[column].mean()
        keys = sorted(means)
        if len(keys) < 4:
            return {'status': 'insufficient_cells', 'cells': len(keys)}
        cols = {k: np.array([link.index[f] for f in means[k].index]) for k in keys}
        xs = np.array([beta[k]['beta'] for k in keys])
        signed = np.array([beta[k].get('sign') in ('+', '-') for k in keys])
        predicted = np.array([1 if beta[k].get('sign') == '+' else -1 for k in keys])

        def stats(weights):
            if weights is None:
                ys = np.array([means[k].mean() for k in keys])
            else:
                ys = np.array([(weights[cols[k]]*means[k].to_numpy()).sum()/max(weights[cols[k]].sum(), 1) for k in keys])
            agree = ((np.sign(ys) == predicted) & (ys != 0))[signed]
            return spearmanr(xs, ys).statistic, (agree.mean() if signed.any() else np.nan), ys
        rho, rate, ys = stats(None)
        draws = np.array([stats(link.weights[i])[:2] for i in range(len(link.weights))])
        agree = int(round(rate*signed.sum())) if signed.any() else 0
        return {'spearman': float(rho), 'spearman_ci': np.nanquantile(draws[:, 0], [.025, .975]).tolist(),
                'sign_agreement_rate': float(rate) if signed.any() else None,
                'sign_agreement_ci': np.nanquantile(draws[:, 1], [.025, .975]).tolist() if signed.any() else None,
                'signed_cells': int(signed.sum()), 'sign_agreement': agree,
                'binomial_p': float(binomtest(agree, int(signed.sum()), .5).pvalue) if signed.any() else None,
                'supported': bool(np.nanquantile(draws[:, 0], .025) > 0), 'cells': len(keys),
                'bootstrap_draws': len(link.weights), 'seed': 20261017,
                'cells_detail': [{'model': k[0], 'class': k[1], 'beta': float(beta[k]['beta']),
                                  'predicted_sign': beta[k].get('sign'), 'delta': float(y)} for k, y in zip(keys, ys)]}
    cells = prereg['cells']
    out = {'tokens': test(cells, 'tok_reason', 'activate:log_reasoning'),
           'accuracy': test(cells, 'acc', 'activate:correct'),
           'secondary': {'tok_total': test(cells, 'tok_total', 'activate:log_total'),
                         'acc_excl_trunc': test(cells, 'acc_excl_trunc', 'activate:correct'),
                         'tokens_restricted': test(cells, 'tok_reason', 'activate:log_total', restrict=True),
                         'accuracy_restricted': test(cells, 'acc', 'activate:correct', restrict=True)}}
    fe = (prereg.get('sensitivity') or {}).get('dataset_fe', {}).get('cells')
    if fe:
        out['secondary']['dataset_fe_tokens'] = test(fe, 'tok_reason', 'activate:log_total')
        out['secondary']['dataset_fe_accuracy'] = test(fe, 'acc', 'activate:correct')
    return out


def main(argv=None):
    """Run a fresh analysis, requiring accepted cohort ownership for rescue output."""
    ap = argparse.ArgumentParser(prog='moe_exp.study.effects')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--out', type=Path)
    ap.add_argument('--design', choices=['main', 'strong', 'rescue'], default='main')
    ap.add_argument('--cohort', type=Path)
    args = ap.parse_args(argv)
    models = args.models.split(',')
    require(models and len(set(models)) == len(models), 'analysis model roster is empty or duplicated')
    if args.design == 'rescue':
        from .cohort import assignments, identity, read
        from .ownership import worker
        require(args.cohort is not None, 'rescue analysis requires --cohort')
        cohort_value = read(args.cohort)
        root = Path(cohort_value['paths']['analysis'])
        out = args.out or root/'rescue'
        require(out.resolve() == root/'rescue', 'rescue analysis destination differs from the cohort binding')
        execution = identity(cohort_value, 'cpu', 'analysis', [row['uid'] for row in assignments(cohort_value)])
        with worker(root, execution):
            value = analyze(models, out, args.design, cohort=args.cohort)
    else:
        value = analyze(models, args.out or DRIVER/'analysis-corrected-v1', args.design)
    log({'event': 'effects', 'primary': value['primary_pooled']})


if __name__ == '__main__':
    main()

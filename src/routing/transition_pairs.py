"""Transition estimates completed with the labels of the sentences that follow Read and Verify.

The native-dynamics panel estimates, per source state, the probability of each next state from the pairs of
adjacent judge-labelled sentences. The 100,000 judged sentences of a model are a random sample of its
sentences, so a Read or Verify sentence has a labelled successor only when the successor happened to be drawn
too. That is rare (GPT-OSS 31%, Qwen3.5 8%), and for some models and benchmarks it never happens, which leaves
the Read and Verify rows unestimated. This module labels the missing successors.

The rule, identical for all eight models: for every judge-labelled sentence of class Read or Verify, if the
sentence index+1 of the same attempt and reasoning segment exists, owns a reasoning token and was never judged,
judge it with the model's original judge setup (same prompt program, decoding and context layout; the sentence,
its neighbours and the problem statement). The new labels enter only the next-state estimates, and only as the
destination of a pair whose source is an original sampled label. Occupancy, dwell, re-entry and label coverage
read the original sample alone, and the added rows carry `pair_completion=True` so nothing else can use them.
A sentence the judge already saw but could not label stays unknown; nothing is redrawn.

Inputs: the saved traces and original judge records of the seven captured models (`routing.sentence_table`), LFM2.5's
own sources and tasks in the study folder, the published panels in the capture record
`report/research-campaign-v3-20261014-r5-r2`.
The source classes are Read and Verify (`read-verify`); `--source-classes Read,Verify,Plan,Explore,Monitor` widens the
rule (`read-verify-plan-explore-monitor`), and a widened plan reuses the labels of the narrower one.

Steps (one launcher runs them all: scripts/routing/transition_pairs.sh):

    python -m routing.transition_pairs plan --model M                CPU  build the judge contexts of the missing
                                         successors, audit them against the contexts of the original labels, write counts
    python -m routing.transition_pairs judge --model M --shard I/N   GPU  judge one shard (durable per-context results,
                                         resumable mid-generation)
    python -m routing.transition_pairs summary --model M             CPU  merge the labels, recompute the native-dynamics
                                         panel, write receipts
    python -m routing.transition_pairs summary --model M --original-sample-only
                                    CPU  the same panel without added labels (reproduces the published panel)
    python -m routing.transition_pairs collect                       CPU  join the eight panels with the published
                                         ones and list the changes

Outputs: the work files (`PAIR_TASKS.json`, `PAIR_SENTENCES.parquet`, the label store) under the bulk root's
`transition-pairs/<model>/`, and the receipts `PAIR_PLAN.json`, `PAIR_LABELS.json`, `NATIVE_DYNAMICS_SUMMARY.json`
under `report/final-routing-20261015-r1/transition-pairs/<model>/<variant>/` and `driver/TRANSITION_PAIRS.<variant>.json`.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from generation.client import slurm_deadline
from labeling.contexts import context_id
from labeling.lfm_inputs import judge_inputs
from routing.dynamics import trajectories
from routing.sentence_table import (CAMPAIGN, PAIR_SOURCES, attempt_layout, campaign, campaign_sources,
                                    original_label_records, pairs_dir, receipt_dir, sentence_extent, trace_key,
                                    write_table)
from stats import bootstrap_panel
from steering.pulse import ADMISSION_MARGIN_HOURS, annotate_rows, log, parse_shard
from store import CLASSES, MODELS, STUDY, Units, binding, digest, file_sha, load, publish, require, verified

NEXT_COLUMN = ':next:'
SUCCESSOR_COLUMNS = ('attempt_id', 'question', 'dataset', 'sentence_index', 'source_index', 'source_class', 'segment',
                     'n_tokens', 'tok_start', 'tok_end', 'n_reasoning', 'annotation_uid')
SKIP_REASONS = ('successor_labelled', 'last_sentence', 'next_segment', 'already_judged', 'no_reasoning_tokens')
ORIGINAL_SAMPLE_SUMMARY = 'NATIVE_DYNAMICS_SUMMARY.original-sample.json'
# The estimands of a reproduced panel must equal the published ones to double precision.
REPRODUCTION_TOLERANCE = 1e-12


# ----------------------------------------------------------------------------- model inputs

def variant(classes: tuple[str, ...]) -> str:
    """Folder name of a source-class choice: the Read/Verify rule is `read-verify`."""
    return '-'.join(c.lower() for c in classes)


def model_sources(model: str) -> list[dict]:
    """The attempts of a model: the captured sources, or LFM2.5's own discovery sources."""
    return verified(STUDY/'lfm25/SOURCES.json')['records'] if model == 'lfm25' else campaign_sources(model)


def original_table_path(model: str) -> Path:
    """Parquet of the model's original labelled sentences (Qwen3.6: the table rebuilt on all its attempts)."""
    if model == 'lfm25':
        inputs = verified(STUDY/'lfm25/DISCOVERY_INPUTS.json')
        require(file_sha(inputs['sentences']) == inputs['sentences_sha256'], 'LFM derived sentences changed')
        return Path(inputs['sentences'])
    if model == 'qwen36':
        receipt = verified(receipt_dir(model)/'SENTENCES.json')
        require(file_sha(receipt['path']) == receipt['sentences_sha256'], 'rebuilt Qwen3.6 sentences changed')
        return Path(receipt['path'])
    return Path(campaign()['saved_inputs'][model]['sentences'])


def original_sentences(model: str) -> pd.DataFrame:
    """The judged random sample's labelled sentences, with the columns the estimators use."""
    columns = ['attempt_id', 'question', 'dataset', 'sentence_index', 'cls', 'segment', 'n_tokens',
               'tok_start', 'tok_end']
    return pd.read_parquet(original_table_path(model), columns=columns)


def sentence_records(frame: pd.DataFrame, family_of_attempt: dict[str, str]) -> list[dict]:
    """Table rows as estimator records: benchmark, family and class name, flagged as original-sample rows."""
    return [{'attempt_id': r.attempt_id, 'question': r.question, 'benchmark': r.dataset,
             'family': family_of_attempt[r.attempt_id], 'class': CLASSES[int(r.cls)],
             'sentence_index': int(r.sentence_index), 'segment': int(r.segment), 'n_tokens': int(r.n_tokens),
             'pair_completion': False} for r in frame.itertuples()]


def judge_history(model: str, sources: list[dict], sentences: pd.DataFrame,
                  classes: tuple[str, ...]) -> tuple[dict, dict]:
    """What the original judge saw: {attempt: {sentence index: span or None}} and {attempt: {index: inputs}}.

    The second result holds the judge inputs of each original sentence of a source class, for the audit of the
    rebuilt contexts. LFM2.5's sampled sentences carry no offsets (None): its derivation required each sampled
    text to equal the sentence at its index.
    """
    source_pairs = sentences[sentences.cls.isin([CLASSES.index(c) for c in classes])]
    wanted = set(zip(source_pairs.attempt_id, source_pairs.sentence_index.astype(int)))
    judged, inputs = defaultdict(dict), defaultdict(dict)
    if model == 'lfm25':
        bulk = Path(binding()['config']['bulk_root'])
        context = {t['uid']: t['inputs'] for t in verified(STUDY/'lfm25/ANNOTATION_TASKS.json')['records']}
        for r in verified(bulk/'lfm25/SAMPLED_SENTENCES.json')['records']:
            judged[r['attempt_id']][r['sentence_index']] = None
            if (r['attempt_id'], r['sentence_index']) in wanted:
                inputs[r['attempt_id']][r['sentence_index']] = context[r['annotation_uid']]
        return judged, inputs
    records = original_label_records(model, classes)
    for source in sources:
        for index, start, end, _label, judge_input in records.get(trace_key(source), []):
            judged[source['attempt_id']][index] = (start, end)
            if (source['attempt_id'], index) in wanted and judge_input is not None:
                inputs[source['attempt_id']][index] = judge_input
    return judged, inputs


def captured_tokens(model: str, source: dict) -> int | None:
    """Completion tokens whose routing LFM2.5's replay captured (its sentence table drops later ones)."""
    if model != 'lfm25':
        return None
    bulk = Path(binding()['config']['bulk_root'])
    return verified(bulk/'lfm25/replay'/(source['attempt_id']+'.json'))['captured_completion_tokens']


# ----------------------------------------------------------------------------- pair completion

def pair_candidates(sentences: list[dict], layout: dict, judged: dict, captured: int | None = None,
                    classes: tuple[str, ...] = PAIR_SOURCES) -> tuple[list[dict], Counter]:
    """The unlabelled successors of one attempt's original sentences of the source classes.

    Args:
        sentences: The attempt's original labelled sentences (estimator records).
        layout: `attempt_layout` of its trace.
        judged: {sentence index: span or None} for every sentence the original judge saw.
        captured: Completion tokens whose routing was captured, when the table drops later tokens.
        classes: Source classes (default Read and Verify).

    Returns:
        (candidates, skipped): one candidate per successor to judge, and why the others need no label.
    """
    labelled = {s['sentence_index'] for s in sentences}
    units, segments = layout['units'], layout['segments']
    candidates, skipped = [], Counter()
    for source in sentences:
        if source['class'] not in classes:
            continue
        i = source['sentence_index']
        j = i+1
        if j in labelled:
            skipped['successor_labelled'] += 1
        elif j >= len(units):
            skipped['last_sentence'] += 1
        elif segments[j] != segments[i]:
            skipped['next_segment'] += 1
        elif j in judged and judged[j] in (None, (units[j]['start'], units[j]['end'])):
            skipped['already_judged'] += 1
        elif (extent := sentence_extent(layout, j, captured)) is None:
            skipped['no_reasoning_tokens'] += 1
        else:
            candidates.append({'attempt_id': source['attempt_id'], 'question': source['question'],
                               'dataset': source['benchmark'], 'sentence_index': j, 'source_index': i,
                               'source_class': source['class'], 'segment': segments[j], **extent,
                               'n_reasoning': layout['n_reasoning'],
                               'inputs': judge_inputs(layout['value'], units, j)})
    return candidates, skipped


def context_parity(sentences: list[dict], layout: dict, inputs: dict) -> dict:
    """Rebuild the judge contexts of original source-class sentences and compare them with the recorded ones.

    `inputs` maps sentence index to the inputs the original judge was given (only for source-class sentences).
    """
    checked = mismatched = 0
    for s in sentences:
        recorded = inputs.get(s['sentence_index'])
        if recorded is not None:
            checked += 1
            mismatched += judge_inputs(layout['value'], layout['units'], s['sentence_index']) != recorded
    return {'checked': checked, 'mismatched': mismatched}


def _plan_attempt(task: tuple) -> dict:
    """Worker: candidates, skip reasons and context parity of one attempt."""
    source, sentences, judged, inputs, captured, classes = task
    layout = attempt_layout(source)
    candidates, skipped = pair_candidates(sentences, layout, judged, captured, classes)
    return {'candidates': candidates, 'skipped': skipped, 'parity': context_parity(sentences, layout, inputs)}


def plan(model: str, workers: int, limit: int | None = None, classes: tuple[str, ...] = PAIR_SOURCES) -> dict:
    """Build the judge contexts of one model's missing successors and publish the plan receipt.

    Writes, in the model's private directory under the variant's folder, PAIR_TASKS.json (one judge task per
    distinct context, sealed) and PAIR_SENTENCES.parquet (one row per successor sentence), and in the report
    directory PAIR_PLAN.json with the counts, the context-parity audit and the annotator setup. `classes` are the
    source states whose successors are judged (the chosen rule: Read and Verify). `limit` is a debug option
    (first N attempts, nothing published).
    """
    sources = model_sources(model)[:limit]
    family_of_attempt = {s['attempt_id']: s['family'] for s in sources}
    frame = original_sentences(model)
    frame = frame[frame.attempt_id.isin(family_of_attempt)]
    judged, inputs = judge_history(model, sources, frame, classes)
    by_attempt = {attempt: sentence_records(group, family_of_attempt) for attempt, group in frame.groupby('attempt_id')}
    tasks = [(s, by_attempt[s['attempt_id']], judged.get(s['attempt_id'], {}), inputs.get(s['attempt_id'], {}),
              captured_tokens(model, s), classes)
             for s in sources if s['attempt_id'] in by_attempt]
    with multiprocessing.get_context('fork').Pool(workers) as pool:
        parts = list(pool.imap_unordered(_plan_attempt, tasks, chunksize=4))
    candidates = [c for part in parts for c in part['candidates']]
    skipped = sum((part['skipped'] for part in parts), Counter())
    parity = {k: sum(part['parity'][k] for part in parts) for k in ('checked', 'mismatched')}
    annotator = load(Path(binding()['source_campaign'])/'CAMPAIGN.json')['annotator']
    for c in candidates:
        c['annotation_uid'] = context_id(c['inputs'], annotator)
    contexts = {c['annotation_uid']: {'uid': c['annotation_uid'], 'inputs': c['inputs'], 'model': model}
                for c in candidates}
    successors = pd.DataFrame([{k: v for k, v in c.items() if k != 'inputs'} for c in candidates],
                              columns=SUCCESSOR_COLUMNS)
    successors = successors.sort_values(['attempt_id', 'sentence_index'], ignore_index=True)
    source_rows = frame[frame.cls.isin([CLASSES.index(c) for c in classes])]
    summary = {'model': model, 'attempts': len(sources), 'original_sentences': len(frame),
               'source_classes': list(classes),
               'source_sentences': {c: int((source_rows.cls == CLASSES.index(c)).sum()) for c in classes},
               'successors_to_judge': len(candidates), 'distinct_contexts': len(contexts),
               'successors_to_judge_by_benchmark': successors.groupby('dataset').size().to_dict(),
               'successors_to_judge_by_source_class': successors.groupby('source_class').size().to_dict(),
               'no_label_needed': {reason: skipped[reason] for reason in SKIP_REASONS},
               'context_parity_with_original_labels': parity}
    if limit:
        return summary
    require(parity['mismatched'] == 0, f"{model}: rebuilt judge contexts differ from the recorded ones: {parity}")
    folder = pairs_dir(model)/variant(classes)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    tasks_receipt = publish(folder/'PAIR_TASKS.json', {'records': sorted(contexts.values(), key=lambda r: r['uid']),
                                                       'model': model})
    write_table(successors, folder/'PAIR_SENTENCES.parquet')
    table = original_table_path(model)
    return publish(receipt_dir(model)/variant(classes)/'PAIR_PLAN.json', {
        **summary, 'tasks': str(folder/'PAIR_TASKS.json'), 'tasks_sha256': tasks_receipt['sha256'],
        'successors': str(folder/'PAIR_SENTENCES.parquet'),
        'successors_sha256': file_sha(folder/'PAIR_SENTENCES.parquet'),
        'original_table': str(table), 'original_table_sha256': file_sha(table),
        'annotator': {k: annotator['config'][k] for k in ('judge_model', 'program_sha256', 'reasoning_effort',
                                                           'max_tokens', 'temperature', 'top_p', 'top_k')},
        'module_sha256': file_sha(Path(__file__))}, provenance=('module_sha256',))


# ----------------------------------------------------------------------------- judge and merge

def judge(model: str, shard: tuple[int, int], classes: tuple[str, ...] = PAIR_SOURCES) -> dict:
    """Judge one shard of a model's planned contexts on the allocated GPUs (resumable, never redrawn).

    The label store is shared by every source-class variant of the model, keyed by context, so a context that
    two plans both contain is judged once.
    """
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'),
            'judging requires a Slurm GPU allocation')
    folder = pairs_dir(model)
    tasks = verified(folder/variant(classes)/'PAIR_TASKS.json')['records']
    return annotate_rows(tasks, shard, slurm_deadline(ADMISSION_MARGIN_HOURS), units_root=folder/'annotations',
                         progress_root=folder/'progress', jobs_root=folder/'jobs', stage='pair-judge-'+model)


def completion_records(model: str, family_of_attempt: dict[str, str],
                       classes: tuple[str, ...] = PAIR_SOURCES) -> tuple[list[dict], dict]:
    """The judged successors as estimator records flagged `pair_completion`, plus the label receipt body.

    Every context must have a terminal result. A successor whose judge outcome is unknown is left out (it
    stays unknown, and is counted).
    """
    plan_receipt = verified(receipt_dir(model)/variant(classes)/'PAIR_PLAN.json')
    folder = pairs_dir(model)
    require(file_sha(plan_receipt['successors']) == plan_receipt['successors_sha256'], 'planned successors changed')
    units = Units(folder/'annotations')
    tasks = verified(folder/variant(classes)/'PAIR_TASKS.json')['records']
    states = {task['uid']: units.state(task['uid']) for task in tasks}
    terminal = ('complete', 'unknown')
    unresolved = [uid for uid, state in states.items() if state is None or state['status'] not in terminal]
    require(not unresolved, f'{model}: {len(unresolved)} contexts without a terminal judge result')
    successors = pd.read_parquet(plan_receipt['successors'])
    records, unknown, destinations = [], 0, Counter()
    for r in successors.itertuples():
        label = states[r.annotation_uid].get('label')
        if label not in CLASSES:
            unknown += 1
            continue
        destinations[r.source_class, label] += 1
        records.append({'attempt_id': r.attempt_id, 'question': r.question, 'benchmark': r.dataset,
                        'family': family_of_attempt[r.attempt_id], 'class': label,
                        'sentence_index': int(r.sentence_index), 'segment': int(r.segment),
                        'n_tokens': int(r.n_tokens), 'pair_completion': True})
    body = {'successors': len(successors), 'labelled': len(records), 'unknown': unknown,
            'destination_by_source_class': {f'{a}:{b}': n for (a, b), n in sorted(destinations.items())},
            'labels_sha256': digest(sorted((r['attempt_id'], r['sentence_index'], r['class']) for r in records))}
    return records, body


# ----------------------------------------------------------------------------- estimands

def completed_transitions(original: list[dict], added: list[dict]) -> list[dict]:
    """Adjacent pairs whose source is an original sampled label; the destination may be an added label.

    A pair that starts at an added label is dropped: its source is not part of the random sample.
    """
    flagged = {(r['attempt_id'], r['sentence_index']) for r in added}
    pairs = trajectories(original+added)['transitions']
    return [t for t in pairs if (t['attempt_id'], t['boundary']-1) not in flagged]


def native_dynamics_rows(sources: list[dict], original: list[dict], added: list[dict] = ()) -> list[dict]:
    """Per-attempt estimands of the native-dynamics panel, with added successor labels in next-state rates only.

    Occupancy, dwell, re-entry and label coverage use `original` alone, as the published panel does; the rate
    of each next state among the pairs that start in a state uses `completed_transitions`.
    """
    dynamics = trajectories(original)
    pairs_by_attempt = defaultdict(list)
    for t in completed_transitions(original, list(added)):
        pairs_by_attempt[t['attempt_id']].append(t)
    grouped = defaultdict(lambda: defaultdict(list))
    for key, rows in (('sentences', original), ('dwell', dynamics['dwell']), ('transitions', dynamics['transitions']),
                      ('reentry', dynamics['reentry'])):
        for r in rows:
            grouped[r['attempt_id']][key].append(r)
    by_attempt = {s['attempt_id']: s for s in sources}
    rows = []
    for attempt, group in grouped.items():
        if not group['sentences']:
            continue
        source = by_attempt[attempt]
        labelled_tokens = sum(s['n_tokens'] for s in group['sentences'])
        row = {'benchmark': source['benchmark'], 'family': source['family'], 'attempt_id': attempt,
               'observed_labelled_tokens': labelled_tokens,
               'label_coverage': labelled_tokens/source['completion_tokens'] if source['completion_tokens'] else None}
        for c in CLASSES:
            runs = [d for d in group['dwell'] if d['class'] == c]
            uncensored = [d['tokens'] for d in runs if not d['left_censored'] and not d['right_censored']]
            observed_boundaries = [d for d in group['transitions'] if d['source'] == c]
            next_pairs = [d for d in pairs_by_attempt[attempt] if d['source'] == c]
            in_class_tokens = sum(s['n_tokens'] for s in group['sentences'] if s['class'] == c)
            row[c+':occupancy'] = in_class_tokens/labelled_tokens if labelled_tokens else None
            row[c+':observed_dwell_tokens'] = float(np.mean([d['tokens'] for d in runs])) if runs else None
            row[c+':uncensored_dwell_tokens'] = float(np.mean(uncensored)) if uncensored else None
            reentries = sum(d['class'] == c for d in group['reentry'])
            row[c+':reentry_per_observed_boundary'] = (
                reentries/len(observed_boundaries) if observed_boundaries else None)
            for destination in CLASSES:
                row[c+NEXT_COLUMN+destination] = (
                    sum(d['destination'] == destination for d in next_pairs)/len(next_pairs) if next_pairs else None)
        rows.append(row)
    return rows


def panel(model: str, rows: list[dict]) -> dict:
    """The model's 10,000-draw whole-family bootstrap panel with simultaneous intervals (published settings)."""
    columns = sorted({k for r in rows for k in r if k not in ('benchmark', 'family', 'attempt_id')})
    return bootstrap_panel(rows, columns, panel='native-dynamics:'+model)


def published_estimands_path(model: str) -> Path | None:
    """Per-attempt estimands of the published panel, when this model has one on the same attempts."""
    if model == 'lfm25':
        return Path(binding()['config']['bulk_root'])/'lfm25/derived/native_dynamics_estimands.parquet'
    return None if model == 'qwen36' else CAMPAIGN/model/'native_dynamics_estimands.parquet'


def published_panel_path(model: str) -> Path:
    """The published NATIVE_DYNAMICS_SUMMARY.json the paper reads."""
    if model == 'lfm25':
        return STUDY/'lfm25/NATIVE_DYNAMICS_SUMMARY.json'
    return CAMPAIGN/model/'NATIVE_DYNAMICS_SUMMARY.json'


def reproduction(rows: list[dict], published: Path) -> dict:
    """Compare per-attempt estimands with the published parquet: same attempts, columns, gaps and values."""
    stored = pd.read_parquet(published).set_index('attempt_id').sort_index()
    new = pd.DataFrame(rows).set_index('attempt_id').sort_index()
    require(list(stored.index) == list(new.index), 'reproduced attempts differ from the published ones')
    numeric = [c for c in stored.columns if c not in ('benchmark', 'family')]
    require(set(numeric) == set(new.columns)-{'benchmark', 'family'},
            'reproduced columns differ from the published ones')
    stored, new = stored[numeric].astype(float), new[numeric].astype(float)
    gaps_differ = int((stored.isna().to_numpy() != new.isna().to_numpy()).sum())
    largest = float(np.nanmax(np.abs(stored.to_numpy()-new[stored.columns].to_numpy())))
    return {'attempts': len(stored), 'columns': len(numeric), 'gaps_differ': gaps_differ, 'largest_difference': largest,
            'reproduced': bool(gaps_differ == 0 and largest <= REPRODUCTION_TOLERANCE)}


def summary(model: str, original_sample_only: bool = False, classes: tuple[str, ...] = PAIR_SOURCES) -> dict:
    """Recompute a model's native-dynamics panel, with or without the added successor labels, and publish it.

    The panel is first built from the original sample alone and, where a published per-attempt table exists,
    required to equal it, which shows that the estimator here is the published one. With added labels, only
    the next-state columns can differ from that baseline.
    """
    sources = model_sources(model)
    family_of_attempt = {s['attempt_id']: s['family'] for s in sources}
    original = sentence_records(original_sentences(model), family_of_attempt)
    baseline = native_dynamics_rows(sources, original)
    published = published_estimands_path(model)
    check = reproduction(baseline, published) if published else None
    require(check is None or check['reproduced'],
            f'{model}: original-sample estimands differ from the published ones: {check}')
    table_sha256 = file_sha(original_table_path(model))
    if original_sample_only:
        return publish(receipt_dir(model)/ORIGINAL_SAMPLE_SUMMARY,
                       {**panel(model, baseline), 'reproduction': check, 'table_sha256': table_sha256})
    added, labels = completion_records(model, family_of_attempt, classes)
    rows = native_dynamics_rows(sources, original, added)
    unchanged = [c for c in rows[0] if NEXT_COLUMN not in c and c not in ('benchmark', 'family', 'attempt_id')]
    by_attempt = {r['attempt_id']: r for r in baseline}
    require(all(r[c] == by_attempt[r['attempt_id']][c] for r in rows for c in unchanged),
            f'{model}: a non-transition estimand changed with the added labels')
    publish(receipt_dir(model)/variant(classes)/'PAIR_LABELS.json', {'model': model, **labels})
    return publish(receipt_dir(model)/variant(classes)/'NATIVE_DYNAMICS_SUMMARY.json', {
        **panel(model, rows),
        'pair_completion': {'labelled_successors': labels['labelled'], 'labels_sha256': labels['labels_sha256']},
        'reproduction_of_published_original_sample': check, 'table_sha256': table_sha256})


def without_estimate(panel_value: dict) -> list[str]:
    return sorted(k for k, v in panel_value['results'].items() if NEXT_COLUMN in k and v['estimate'] is None)


def without_interval(panel_value: dict) -> list[str]:
    return sorted(k for k, v in panel_value['results'].items() if NEXT_COLUMN in k and v['interval'] is None)


def collect(classes: tuple[str, ...] = PAIR_SOURCES) -> dict:
    """Join the eight completed panels with the published ones: what became estimable, what moved.

    For every model: next-state cells without an estimate or an interval before and after, the largest change
    of any other estimate (zero by construction, except Qwen3.6 whose cohort is rebuilt) and of any other
    interval (the simultaneous critical value is shared by the panel's columns), and the family support of the
    Read and Verify rows per benchmark.
    """
    models = {}
    for model in MODELS:
        published = load(published_panel_path(model))
        new = verified(receipt_dir(model)/variant(classes)/'NATIVE_DYNAMICS_SUMMARY.json')
        other = [k for k in new['results'] if NEXT_COLUMN not in k]
        estimate_shift = max(abs(new['results'][k]['estimate']-published['results'][k]['estimate']) for k in other
                             if None not in (new['results'][k]['estimate'], published['results'][k]['estimate']))
        interval_shift = max(abs(a-b) for k in other if new['results'][k]['interval'] and published['results'][k]['interval']
                             for a, b in zip(new['results'][k]['interval'], published['results'][k]['interval']))
        models[model] = {
            'without_estimate': {'published': without_estimate(published), 'completed': without_estimate(new)},
            'without_interval': {'published': without_interval(published), 'completed': without_interval(new)},
            'other_estimates_largest_change': estimate_shift, 'other_intervals_largest_change': interval_shift,
            'critical_value': {'published': published['critical_value'], 'completed': new['critical_value']},
            'source_family_support': {c: new['results'][f'{c}:next:Read']['benchmark_family_support'] for c in CLASSES},
            'pair_completion': new['pair_completion'], 'panel_sha256': new['sha256']}
    return publish(STUDY/'driver'/f'TRANSITION_PAIRS.{variant(classes)}.json',
                   {'source_classes': list(classes), 'models': models})


def main(argv: list[str] | None = None) -> None:
    """Run one step for one model (`collect` for all eight) and log the counts that identify its receipt."""
    parser = argparse.ArgumentParser(prog='routing.transition_pairs', description=__doc__.split('\n\n')[0])
    parser.add_argument('command', choices=['plan', 'judge', 'summary', 'collect'])
    parser.add_argument('--model')
    parser.add_argument('--shard', default='0/1', type=parse_shard)
    parser.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    parser.add_argument('--limit', type=int, help='plan debug: first N attempts, nothing published')
    parser.add_argument('--original-sample-only', action='store_true')
    parser.add_argument('--source-classes', default=','.join(PAIR_SOURCES),
                        help='comma-separated states whose successors are judged (default: the chosen rule)')
    args = parser.parse_args(argv)
    classes = tuple(args.source_classes.split(','))
    require(set(classes) <= set(CLASSES), 'unknown source class')
    if args.command == 'collect':
        result = collect(classes)
    elif args.command == 'plan':
        result = plan(args.model, args.workers, args.limit, classes)
    elif args.command == 'judge':
        result = judge(args.model, args.shard, classes)
    else:
        result = summary(args.model, args.original_sample_only, classes)
    shown = ('sha256', 'successors_to_judge', 'distinct_contexts', 'no_label_needed',
             'context_parity_with_original_labels')
    log({'event': args.command, 'model': args.model, **{k: v for k, v in result.items() if k in shown}})


if __name__ == '__main__':
    main()

"""Extraction and bounded analysis of whole-reasoning rows.

extract(root, model) replays and scores the finished rows of one model of a release with
pulse_outcomes.record (GPT token replay, the threaded math_verify helper) and writes
root/MODEL/extraction/. combine(root) joins the extractions of every release listed in
root/SOURCES.json by family, keeps every planned family and cell, shares all 10,000
benchmark-stratified bootstrap draws across models and contrasts, and writes
root/analysis-combined/EFFECTS_WHOLE.json (the paper's whole-reasoning results).
Run both through python -m steering.whole.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import shutil
from collections import Counter
import json
import math
import os
from pathlib import Path
from threading import local

import numpy as np
import pandas as pd

from store import CLASSES, digest, load, publish, require, verified
from steering import pulse as driver
from steering.whole import (
    DOSES,
    DRAWS,
    MODELS,
    SEED,
    bound_store,
    efficiency_directed,
    link_receipt,
    manifest,
    repetition_rate,
)

_EXTRACTOR = local()


def unit_bounds(row: dict) -> dict:
    """Bound log1p reasoning and fixed-budget strict accuracy for one assigned row.

    Args:
        row: Assignment joined to extraction; missing outcomes retain their allowance.

    Returns:
        Point compatibility values and separate lower/upper bound components.
    """
    allowance = int(row['total_allowance'])
    observed = row.get('outcome_status') == 'observed'
    cap = row.get('finish_reason') == 'length'
    tokens = row.get('reasoning_tokens', np.nan)
    lower = math.log1p(tokens) if observed and np.isfinite(tokens) else 0.
    reasoning_closed = row.get('reasoning_end') is not None and not pd.isna(row.get('reasoning_end'))
    upper = math.log1p(allowance) if not observed or (cap and not reasoning_closed) else lower
    strict = row.get('strict_correct')
    if cap and observed:
        acc_lower = acc_upper = 0.
    elif observed and strict is not None and not pd.isna(strict):
        acc_lower = acc_upper = float(strict)
    else:
        acc_lower, acc_upper = 0., 1.
    return {'log_reasoning': (lower+upper)/2, 'log_reasoning:lower': lower, 'log_reasoning:upper': upper,
            'accuracy': (acc_lower+acc_upper)/2, 'accuracy:lower': acc_lower, 'accuracy:upper': acc_upper}


def _extract_one(row: dict, *, b: dict, model: str, tokenizer, close, store, inputs):
    """Replay and score one completed draw, recording a per-unit extraction failure."""
    from steering.pulse_outcomes import record
    try:
        if row.get('nomination_status', 'nominated') != 'nominated':
            return None
        if (store.state(row['uid']) or {}).get('status') != 'complete':
            return None
        folder = store.folder(row['uid'])
        raw = verified(folder/'OUTPUT.json')
        require(raw['assignment_sha256'] == digest(row) and raw['total_allowance'] == row['total_allowance'],
                'whole output assignment/allowance join differs')
        require(raw['profile_sha256'] == row['profile_sha256'] and raw['segments'][0]['seed'] == row['seed'],
                'whole output profile/seed join differs')
        # A length-limit row has its own prefix (cut native tokens + closing suffix): read it from the row.
        rec = record(b, model, row, raw, tokenizer, close,
                     frozen_input=None if row.get('budget_prefix') else inputs[row['family']])
        capped = raw['finish_reason'] == 'length'
        strict = False if capped else rec['strict_correct']
        outcome = {**{k: v for k, v in row.items() if k not in ('actions', 'source', 'budget_prefix')},
                   **{k: rec[k] for k in ('generated_tokens', 'prefix_tokens', 'reasoning_tokens',
                      'reasoning_end', 'finish_reason', 'answer_present', 'reasoning_fallback')},
                   'total_allowance': raw['total_allowance'], 'strict_correct': strict,
                   'repetition_16gram': repetition_rate(raw['token_ids']), 'outcome_status': 'observed',
                   'dose_json': json.dumps(verified(folder/'CAPTURE.json')['stats']['dose'])}
        sentences = [{**{k: row[k] for k in ('uid', 'model', 'family', 'fold', 'class', 'arm')},
                      **{k: s[k] for k in ('sentence_index', 'token_start', 'token_end', 'complete')},
                      'sentence_text': s['inputs']['sentence'],
                      'prev_sentence': s['inputs']['previous_sentence'] or '<START OF RESPONSE>',
                      'next_sentence': s['inputs']['next_sentence'] or '<END OF RESPONSE>'} for s in rec['sentences']]
        return outcome, sentences, None
    except Exception as error:
        return None, [], {'uid': row['uid'], 'error': f'{type(error).__name__}: {error}'}


def _initialize_extractor(local_path: str, model: str) -> None:
    """Give every thread its own fast tokenizer and closure sequences."""
    from transformers import AutoTokenizer
    from steering.runtime import closure
    _EXTRACTOR.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    _EXTRACTOR.close = closure(_EXTRACTOR.tokenizer, model)


def _extract_with_worker(row: dict, *, b: dict, model: str, store, inputs):
    """Use only the current worker's tokenizer while sharing immutable prompt inputs."""
    return _extract_one(row, b=b, model=model, tokenizer=_EXTRACTOR.tokenizer,
                        close=_EXTRACTOR.close, store=store, inputs=inputs)


def extract(root: Path, model: str) -> dict:
    """Extract whatever completed, preserving missing rows for the subsequent planned join."""
    from transformers import AutoTokenizer
    from steering.runtime import prompt_prefix
    value = manifest(root, model)
    store = bound_store(root, model, value['dose'])
    b = driver.binding()
    tokenizer = AutoTokenizer.from_pretrained(b['models'][model]['local_path'], local_files_only=True)
    rows = value['records']
    # Every row of a family shares its original prompt; a sweep manifest has no native row to read it from.
    first = {}
    for row in rows:
        first.setdefault(row['family'], row)
    # The family prompt never includes a length-limit row's own prefix.
    inputs = {family: prompt_prefix(b, model, {k: v for k, v in row.items() if k != 'budget_prefix'}, tokenizer)
              for family, row in first.items()}

    outcomes, sentences, failures = [], [], []
    # Calling score_completion_detailed from these threads selects its process helper.
    workers = int(os.environ.get('MOE_EXP_SCORING_PROCESSES', '8'))
    with ThreadPoolExecutor(max_workers=workers, initializer=_initialize_extractor,
                           initargs=(b['models'][model]['local_path'], model)) as executor:
        from functools import partial
        extract_one = partial(_extract_with_worker, b=b, model=model, store=store, inputs=inputs)
        for result in executor.map(extract_one, rows):
            if result is None:
                continue
            outcome, spans, failure = result
            if outcome is not None:
                outcomes.append(outcome)
                sentences.extend(spans)
            if failure:
                failures.append(failure)
    folder = root/model/'extraction'
    require(not folder.exists(), 'whole extraction requires a fresh destination')
    # Written aside and renamed last, so a job that dies half-way leaves nothing for combine to read.
    partial = folder.with_name('extraction.partial')
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True, mode=0o700)
    pd.DataFrame(outcomes, columns=list(outcomes[0]) if outcomes else ['uid']).to_parquet(partial/'outcomes.parquet', index=False)
    pd.DataFrame(sentences, columns=list(sentences[0]) if sentences else ['uid']).to_parquet(partial/'sentences.parquet', index=False)
    receipt = publish(partial/'EXTRACTION.json', {'model': model, 'manifest_sha256': value['sha256'],
                      'planned': len(rows), 'extracted': len(outcomes), 'failures': failures, 'sentences': len(sentences)})
    partial.rename(folder)
    return receipt


def load_table(sources: list[dict]) -> pd.DataFrame:
    """Join every manifest row and fixed-denominator lexical occupancy with bounds.

    Args:
        sources: {model, root, dose} entries to join (SOURCES.json). Sources are read in parallel
            processes, one per CPU.
    """
    workers = min(len(sources), int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    if workers <= 1:
        return pd.concat(map(_source_table, sources), ignore_index=True)
    # Spawned single-threaded workers: forked BLAS pools oversubscribe the allocation.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        return pd.concat(pool.map(_source_table, sources), ignore_index=True)


def _extracted(root: Path, model: str, value: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one release's extracted outcomes and lexically labelled sentences for one manifest."""
    from routing.lexical import LEXICAL, predict
    folder = root/model/'extraction'
    require((folder/'EXTRACTION.json').exists(), f'{root.name} has no {model} extraction')
    require(verified(folder/'EXTRACTION.json')['manifest_sha256'] == value['sha256'],
            'whole extraction belongs to another manifest')
    observed = pd.read_parquet(folder/'outcomes.parquet') if (folder/'outcomes.parquet').exists() else pd.DataFrame()
    sentences = pd.read_parquet(folder/'sentences.parquet') if (folder/'sentences.parquet').exists() else pd.DataFrame()
    if not sentences.empty:
        frozen = verified(root/'RELEASE.json')['classifier']
        # Full solutions have far more sentences than the old 512-token windows.
        sentences = pd.concat([predict(sentences.iloc[start:start+10000], LEXICAL/'readout-v1', frozen=frozen)
                               for start in range(0, len(sentences), 10000)], ignore_index=True)
    return observed, sentences


def _source_table(source: dict) -> pd.DataFrame:
    """Join one model and dose of one release (see load_table).

    A source may list top-up releases ('topups') that ran the never-started rows of its failed shards;
    their outcomes join the source's manifest by UID. It may also list complement releases
    ('complements', whole.complement_manifest) whose new cells join the source's rows by family.
    """
    model, root = source['model'], Path(source['root'])
    value = manifest(root, model, source['dose'])
    parts = [_extracted(root, model, value)]
    for topup in map(Path, source.get('topups', [])):
        planned = manifest(topup, model, source['dose'])
        require(planned['parent_manifest_sha256'] == value['sha256'], 'top-up belongs to another manifest')
        parts.append(_extracted(topup, model, planned))
    for complement in map(Path, source.get('complements', [])):
        planned = manifest(complement, model, source['dose'])
        require(planned['design'] == 'complement' and planned['parent_manifest_sha256'] == value['sha256'],
                'complement belongs to another manifest')
        parts.append(_extracted(complement, model, planned))
        value = {**value, 'records': value['records']+planned['records']}
    observed = pd.concat([part[0] for part in parts], ignore_index=True)
    sentences = pd.concat([part[1] for part in parts], ignore_index=True)
    by_uid = {r['uid']: r for r in observed.to_dict('records')}
    require(len(by_uid) == len(observed), 'duplicate whole extracted UID')
    require(set(by_uid) <= {r['uid'] for r in value['records']}, 'whole extraction UID outside manifest')
    groups = {uid: g.to_dict('records') for uid, g in sentences.groupby('uid')} if not sentences.empty else {}
    rows = []
    for assignment in value['records']:
        row = {**assignment, **by_uid.get(assignment['uid'], {
            'outcome_status': 'missing', 'generated_tokens': np.nan, 'reasoning_tokens': np.nan,
            'reasoning_end': None, 'finish_reason': 'unknown', 'strict_correct': None,
            'answer_present': np.nan, 'repetition_16gram': np.nan, 'dose_json': '{}',
        }), 'release': root.name}
        if assignment.get('nomination_status', 'nominated') != 'nominated':
            rows.append({**row, 'outcome_status': 'not_in_design'})
            continue
        row.update(unit_bounds(row))
        spans = groups.get(row['uid'], [])
        # The denominator is this request's frozen total allowance for every arm.
        # Unexecuted reasoning beyond a cap/missing row remains unknown mass.
        denominator = row['total_allowance']
        observed_tokens = (row.get('reasoning_tokens', 0) if row['outcome_status'] == 'observed' else 0)
        counts = dict.fromkeys(CLASSES, 0.)
        covered = 0
        for span in spans:
            width = max(0, min(denominator, span['token_end'])-max(0, span['token_start']))
            counts[span['lexical_label']] += width
            covered += width
        if covered > observed_tokens:
            row['unit_failure_reason'] = (f'sentence_coverage_exceeds_reasoning_ownership: '
                                          f'{covered} > {observed_tokens}')
            row['outcome_status'] = 'unknown_sentence_coverage'
            row.update(unit_bounds(row))
            counts, covered, observed_tokens = dict.fromkeys(CLASSES, 0.), 0, 0
        natural = row['outcome_status'] == 'observed' and (
            row.get('reasoning_end') is not None and not pd.isna(row.get('reasoning_end')) or
            row.get('finish_reason') == 'stop')
        unknown = max(0, observed_tokens-covered) if natural else denominator-covered
        for class_name in CLASSES:
            lower, upper = counts[class_name]/denominator, (counts[class_name]+unknown)/denominator
            row['occ_'+class_name] = (lower+upper)/2
            row['occ_'+class_name+':lower'], row['occ_'+class_name+':upper'] = lower, upper
        rows.append(row)
    # The experts of each row stay in the manifest; the analysis never reads them, and they dominate the memory.
    return pd.DataFrame(rows).drop(columns=['source', 'actions'], errors='ignore')


def contrast(target: dict, control: dict, metric: str, benefit=False) -> tuple[float, float, float]:
    """Subtract two bounded endpoints, reversing bound order for token benefit."""
    lower = target[metric+':lower']-control[metric+':upper']
    upper = target[metric+':upper']-control[metric+':lower']
    if benefit:
        lower, upper = -upper, -lower
    return (lower+upper)/2, lower, upper


def family_contrasts(table: pd.DataFrame, link: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired within-family contrasts for every model, dose and planned cell.

    Native rows are dose-free, so a sweep dose (which runs none) pairs with the native rows of the
    model's base release, family by family.

    Returns:
        One row per (model, dose, family, cell) with target-random, target-native and random-native
        bounded contrasts, and the native-repeat noise per (model, family).
    """
    cells, native_repeats, natives = [], [], {}
    for (model, family), members in table[table.operation == 'native'].groupby(['model', 'family']):
        arms = {r['arm']: r for r in members.to_dict('records')}
        require(len(members) == 2 and arms.keys() == {'native', 'native_repeat'}, 'whole native pair differs')
        natives[model, family] = arms['native']
        noise = {'family': family, 'model': model}
        for metric, name in (('log_reasoning', 'tokens'), ('accuracy', 'accuracy')):
            for suffix, value in zip(('', ':lower', ':upper'), contrast(arms['native_repeat'], arms['native'], metric)):
                noise[name+suffix] = value
        native_repeats.append(noise)
    steered = table[table.operation != 'native']
    for (model, dose, family), members in steered.groupby(['model', 'dose', 'family']):
        records = members.to_dict('records')
        require((model, family) in natives, 'steered family has no native rows in any release')
        native = natives[model, family]
        planned = sorted({(r['operation'], r['class']) for r in records},
                         key=lambda cell: (cell[0], CLASSES.index(cell[1])))
        for operation, class_name in planned:
            arms = {r['arm']: r for r in records if r['class'] == class_name and r['operation'] == operation}
            target, random = arms['target_'+operation], arms['random_'+operation]
            efficiency = efficiency_directed(model, operation, class_name, link)
            nominated = all(r.get('nomination_status', 'nominated') == 'nominated' for r in (target, random))
            require(nominated or not efficiency, 'efficiency-directed cell is not nominated')
            if not nominated:
                continue
            cell = {'family': family, 'model': model, 'dose': dose, 'class': class_name, 'operation': operation,
                    'efficiency_directed': efficiency, 'nomination_status': 'nominated'}
            for metric, name in (('log_reasoning', 'tokens'), ('accuracy', 'accuracy')):
                for comparison, control, treated in (('target_random', random, target),
                                                     ('target_native', native, target),
                                                     ('random_native', native, random)):
                    values = contrast(treated, control, metric, metric == 'log_reasoning')
                    for suffix, value in zip(('', ':lower', ':upper'), values):
                        cell[comparison+':'+name+suffix] = value
            for class_response in CLASSES:
                metric = 'occ_'+class_response
                sign = -1 if operation == 'suppress' else 1
                for comparison, control, treated in (('target_random', random, target),
                                                     ('target_native', native, target),
                                                     ('random_native', native, random)):
                    values = contrast(treated, control, metric, sign < 0)
                    for suffix, value in zip(('', ':lower', ':upper'), values):
                        cell[comparison+':response:'+class_response+suffix] = value
            cells.append(cell)
    return pd.DataFrame(cells), pd.DataFrame(native_repeats)


def primary_frame(cells: pd.DataFrame, comparison: str = 'target_random', dose: str | None = None) -> pd.DataFrame:
    """One family column per model and metric, plus the pooled equal-model average.

    Args:
        cells: family_contrasts cells.
        comparison: target_random (primary) or a native companion.
        dose: The strength to read; None reads each model's only dose (a single-release analysis).

    Returns:
        Family rows; a family a model did not run (LFM2.5 has 112, Nemotron and Qwen3-30B 128) is NaN
        there, and the pooled column averages the models that ran it.
    """
    selected = cells[cells.efficiency_directed]
    if dose is None:
        require(not selected.groupby('model').dose.nunique().gt(1).any(), 'several doses need an explicit dose')
    else:
        selected = selected[selected.dose == dose]
    if 'nomination_status' in selected:
        require((selected.nomination_status == 'nominated').all(), 'primary composite contains an unnominated cell')
    metrics = [comparison+':'+metric+suffix for metric in ('tokens', 'accuracy') for suffix in ('', ':lower', ':upper')]
    grouped = selected.groupby(['family', 'model'])[metrics].mean()
    models = [model for model in MODELS if model in set(selected.model)]
    frame = pd.DataFrame({'family': sorted(cells.family.unique())}).set_index('family')
    for metric in ('tokens', 'accuracy'):
        for suffix in ('', ':lower', ':upper'):
            for model in models:
                frame[model+':'+metric+suffix] = grouped[comparison+':'+metric+suffix].xs(model, level='model')
            frame['pooled:'+metric+suffix] = frame[[m+':'+metric+suffix for m in models]].mean(axis=1)
    return frame.reset_index()


def point_columns(frame: pd.DataFrame) -> list[str]:
    """Contrast names of a family frame, without their bound columns."""
    return [c for c in frame.columns if c != 'family' and not c.endswith((':lower', ':upper'))]


def summarize_present(boot, frame: pd.DataFrame, family_name: str) -> dict:
    """Summarize every contrast of a family frame over the families each column ran."""
    from steering.pulse_effects import summarize
    names = point_columns(frame)
    return summarize(boot, frame, names, family_name, column_masks={name: frame[name].notna() for name in names})


def dose_response(boot, cells: pd.DataFrame) -> dict:
    """Target-minus-random efficiency at every strength, and its slope on log2(strength).

    Declared separately from the full-dose primary. Every model and dose column and every slope shares
    one simultaneous envelope. The slope is the least-squares slope over the model's three strengths,
    per family and with bounds; the pooled slope averages the models that ran every strength.
    """
    frames = []
    for dose in DOSES:
        if not (cells.dose == dose).any():
            continue
        frame = primary_frame(cells, 'target_random', dose).set_index('family')
        frames.append(frame.rename(columns=lambda c: c.replace(':', '@'+dose+':', 1)))
    joined = pd.concat(frames, axis=1)
    x = {dose: math.log2(fraction) for dose, fraction in DOSES.items()}
    centered = {dose: value-np.mean(list(x.values())) for dose, value in x.items()}
    weights = {dose: value/sum(v*v for v in centered.values()) for dose, value in centered.items()}
    complete = [m for m in MODELS if all(m+'@'+dose+':tokens' in joined for dose in DOSES)]
    for metric in ('tokens', 'accuracy'):
        for model in complete:
            column = model+':slope:'+metric
            joined[column] = sum(w*joined[f'{model}@{d}:{metric}'] for d, w in weights.items())
            # A positive weight takes the lower bound of its dose for the lower slope bound, a negative the upper.
            for bound, flip in ((':lower', ':upper'), (':upper', ':lower')):
                joined[column+bound] = sum(w*joined[f'{model}@{d}:{metric}'+(bound if w > 0 else flip)]
                                           for d, w in weights.items())
        for suffix in ('', ':lower', ':upper'):
            joined['pooled:slope:'+metric+suffix] = joined[[m+':slope:'+metric+suffix for m in complete]].mean(axis=1)
    result = summarize_present(boot, joined.reset_index(), 'whole-dose-response')
    return {'contrasts': result, 'slope_models': complete, 'log2_strength': x,
            'models_by_dose': {dose: sorted(cells[cells.dose == dose].model.unique()) for dose in DOSES},
            'declared': 'secondary; separate from the full-dose primary; no best strength is selected'}


def arm_summaries(boot, table: pd.DataFrame) -> tuple[list, dict]:
    """Summarize arm levels with the shared draws and normalize additive dose counters."""
    from steering.pulse_effects import summarize
    levels_by_arm, dose_by_arm = [], {}
    table = table[table.get('nomination_status', pd.Series('nominated', index=table.index)) == 'nominated']
    # Arm levels describe the base design; complement cells appear only in the per-cell effects.
    if 'complement' in table:
        table = table[~table.complement.eq(True)]
    for (model, dose, arm), group in table.groupby(['model', 'dose', 'arm']):
        observed = group[group.outcome_status == 'observed']
        levels = group[['family', 'outcome_status', 'total_allowance']].copy()
        unknown = group.outcome_status != 'observed'
        for metric, values, upper in (
            ('cap_rate', (group.get('finish_reason') == 'length').astype(float), 1.),
            ('answer_presence', group.get('answer_present', pd.Series(np.nan, index=group.index)).astype(float), 1.),
            ('repetition_16gram', group.get('repetition_16gram', pd.Series(np.nan, index=group.index)), 1.),
            ('total_tokens', group.get('generated_tokens', pd.Series(np.nan, index=group.index)), group.total_allowance),
        ):
            levels[metric+':lower'] = values.where(~unknown, 0.)
            levels[metric+':upper'] = values.where(~unknown, upper)
            levels[metric] = (levels[metric+':lower']+levels[metric+':upper'])/2
        levels_by_arm.append({'model': model, 'dose': dose, 'arm': arm, 'planned': len(group),
                              'complete': len(observed),
                              'cap_rate': float((observed.finish_reason == 'length').mean()) if len(observed) else None,
                              'answer_presence': float(observed.answer_present.mean()) if len(observed) else None,
                              'repetition_16gram': float(observed.repetition_16gram.mean()) if len(observed) else None,
                              'mean_total_tokens': float(observed.generated_tokens.mean()) if len(observed) else None,
                              'bounded_family_bootstrap': summarize(boot, levels, ['cap_rate', 'answer_presence',
                                  'repetition_16gram', 'total_tokens'], 'whole-levels:'+model+':'+dose+':'+arm)})
        totals = {}
        for encoded in observed.get('dose_json', []):
            for layer, counters in json.loads(encoded).items():
                aggregate = totals.setdefault(layer, Counter())
                aggregate.update(counters)
        dose_by_arm[model+'|'+dose+'|'+arm] = {layer: {**dict(counters),
            'changed_membership_fraction': counters['changed_membership']/max(1, counters['active_positions']),
            'native_target_weight_mean': counters['native_target_weight']/max(1, counters['active_positions']),
            'executed_target_weight_mean': counters['executed_target_weight']/max(1, counters['active_positions']),
            'l1_mean': counters['l1']/max(1, counters['active_positions'])} for layer, counters in totals.items()}
        aggregate = Counter()
        for counters in totals.values():
            aggregate.update(counters)
        positions = max(1, aggregate['active_positions'])
        levels_by_arm[-1]['manipulation_check'] = {
            'mean_native_frequency_steered_experts': (
                float(group.steered_native_frequency.mean()) if 'steered_native_frequency' in group and
                group.steered_native_frequency.notna().any() else None),
            'changed_membership_fraction': aggregate['changed_membership']/positions,
            'l1': aggregate['l1']/positions,
            'deactivated_native_weight': aggregate['native_deactivate_weight']/positions,
            'active_layer_positions': aggregate['active_positions'],
        }
    return levels_by_arm, dose_by_arm


def bootstrap():
    """The shared benchmark-stratified family bootstrap of every whole-reasoning interval."""
    from steering.pulse_effects import Bootstrap
    benchmarks = {r['family']: r['benchmark'] for r in load(driver.STUDY/'ALLOCATION.json')['records']}
    return Bootstrap(sorted(benchmarks), benchmarks, draws=DRAWS, seed=SEED)


def failures(table: pd.DataFrame, sources: list[dict]) -> dict:
    """Extraction, sentence-coverage and design exclusions, each kept with its unit and reason."""
    extraction = []
    for source in sources:
        for root in [source['root'], *source.get('topups', []), *source.get('complements', [])]:
            path = Path(root)/source['model']/'extraction/EXTRACTION.json'
            if path.exists():
                extraction.extend({'model': source['model'], 'dose': source['dose'], **failure}
                                  for failure in verified(path)['failures'])
    drops = table[table.get('nomination_status', pd.Series('nominated', index=table.index)) != 'nominated']
    return {'extraction_failures': {'count': len(extraction), 'units': extraction},
            'unit_failures': table.loc[table.get('unit_failure_reason', pd.Series(index=table.index, dtype=object)).notna(),
                                       ['uid', 'model', 'dose', 'family', 'unit_failure_reason']].to_dict('records')
                             if 'unit_failure_reason' in table else [],
            'not_in_design': drops[['model', 'dose', 'fold', 'class', 'operation']].drop_duplicates().to_dict('records')}


def cell_summaries(boot, cells: pd.DataFrame) -> tuple[dict, dict]:
    """Per steered cell: length/accuracy effects and the signed step-type occupancy response."""
    from steering.pulse_effects import summarize
    effects, mechanism = {}, {}
    for (model, dose, operation, class_name), sub in cells.groupby(['model', 'dose', 'operation', 'class']):
        key = '|'.join((model, dose, operation, class_name))
        effects[key] = summarize(boot, sub, [comparison+':'+metric for comparison in
                                             ('target_random', 'target_native', 'random_native')
                                             for metric in ('tokens', 'accuracy')], 'whole-cell-effects')
        mechanism[key] = summarize(boot, sub, [comparison+':response:'+observed for comparison in
                                               ('target_random', 'target_native', 'random_native')
                                               for observed in CLASSES], 'whole-signed-response')
    return effects, mechanism


def verdicts(primary: dict, models) -> dict:
    """The declared decision rule: token benefit above zero and accuracy loss above -5 points."""
    result = {}
    for model in (*models, 'pooled'):
        tokens = primary.get(model+':tokens', {}).get('bound_simultaneous', [-np.inf, np.inf])
        accuracy = primary.get(model+':accuracy', {}).get('bound_simultaneous', [-np.inf, np.inf])
        result[model] = ('efficient_steering' if tokens[0] > 0 and accuracy[0] > -.05 else
                         'speed_accuracy_tradeoff' if tokens[0] > 0 and accuracy[0] <= -.05 else
                         'inconclusive; assess native companions for generic disruption')
    return result


def combine(root: Path) -> dict:
    """Analyze base and strength-sweep releases together into EFFECTS_WHOLE.json.

    Args:
        root: The release folder. Its SOURCES.json lists {model, root, dose} for each model's base release
            (with its native rows) and its sweep releases; RELINK.json, if present, names the link receipt.

    Returns:
        The pre-registered primary at the full dose for every model, its native companions, the
        separately declared dose response, per-cell effects and step-type responses at every dose, and
        breakdown (cap rate, repetition) and manipulation checks by model, dose and arm.
    """
    from steering.pulse_effects import summarize
    folder = root/'analysis-combined'
    require(not folder.exists(), 'combined whole analysis requires a fresh directory')
    link = link_receipt(root)
    sources = verified(root/'SOURCES.json')['sources']
    require(len({(s['model'], s['dose']) for s in sources}) == len(sources), 'a model and dose is listed twice')
    table = load_table(sources)
    cells, noise = family_contrasts(table, link)
    boot = bootstrap()
    models = [model for model in MODELS if model in set(table.model)]
    result = {'models': models, 'bootstrap_draws': DRAWS, 'seed': SEED,
              # The receipt whose native signs decide which cells are efficiency-directed.
              'efficiency_link': {'path': str(link), 'sha256': verified(link)['sha256']},
              'sources': [{**s, 'manifest_sha256': manifest(Path(s['root']), s['model'], s['dose'])['sha256'],
                           'release_sha256': verified(Path(s['root'])/'RELEASE.json')['sha256']} for s in sources],
              **failures(table, sources),
              'primary': summarize_present(boot, primary_frame(cells, dose='main'), 'whole-primary-full'),
              'companions': {comparison: summarize_present(boot, primary_frame(cells, comparison, 'main'),
                                                           'whole-companions-full:'+comparison)
                             for comparison in ('target_native', 'random_native')},
              'dose_response': dose_response(boot, cells),
              'native_repeat_minus_native': {model: summarize(boot, noise[noise.model == model],
                                                            ['tokens', 'accuracy'], 'whole-native-noise')
                                             for model in models}}
    result['cell_effects'], result['mechanism'] = cell_summaries(boot, cells)
    result['levels'], result['manipulation'] = arm_summaries(boot, table)
    result['verdict'] = verdicts(result['primary'], models)
    folder.mkdir(parents=True, mode=0o700)
    table.to_parquet(folder/'continuations.parquet', index=False)
    cells.to_parquet(folder/'cells.parquet', index=False)
    return publish(folder/'EFFECTS_WHOLE.json', result)



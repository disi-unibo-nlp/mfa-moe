"""Bounded efficiency and lexical mechanism inference for whole-reasoning rows.

Extraction reuses outcomes.record, including GPT token replay and the threaded
math_verify helper. Analysis keeps every planned family/cell and shares all
10,000 benchmark-stratified bootstrap draws across models and contrasts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CLASSES, digest, load, publish, require, verified
from . import driver
from .whole import ROOT, MODELS, SUPPRESS, manifest, bound_store, repetition_rate

LINK_WORDING = ('The frozen preregistered test (activation arms, central strength, log total tokens, main design) '
                'is not estimable in this design. We report an exploratory transfer of the frozen beta_tok signs '
                'to the promote arms.')


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
    upper = math.log1p(allowance) if not observed or cap else lower
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
    from .outcomes import record
    try:
        if (store.state(row['uid']) or {}).get('status') != 'complete':
            return None
        folder = store.folder(row['uid'])
        raw = verified(folder/'OUTPUT.json')
        require(raw['assignment_sha256'] == digest(row) and raw['total_allowance'] == row['total_allowance'],
                'whole output assignment/allowance join differs')
        require(raw['profile_sha256'] == row['profile_sha256'] and raw['segments'][0]['seed'] == row['seed'],
                'whole output profile/seed join differs')
        rec = record(b, model, row, raw, tokenizer, close, frozen_input=inputs[row['family']])
        capped = raw['finish_reason'] == 'length'
        strict = False if capped else rec['strict_correct']
        outcome = {**{k: v for k, v in row.items() if k not in ('actions', 'source')},
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



def extract(model: str) -> dict:
    """Extract whatever completed, preserving missing rows for the subsequent planned join."""
    from transformers import AutoTokenizer
    from .runtime import closure, prompt_prefix
    value = manifest(model)
    store = bound_store(model, value['dose'])
    b = driver.binding()
    tokenizer = AutoTokenizer.from_pretrained(b['models'][model]['local_path'], local_files_only=True)
    close = closure(tokenizer, model)
    rows = value['records']
    inputs = {row['family']: prompt_prefix(b, model, row, tokenizer) for row in rows if row['arm'] == 'native'}


    outcomes, sentences, failures = [], [], []
    # Calling score_completion_detailed from these threads selects its process helper.
    with ThreadPoolExecutor(max_workers=2) as executor:
        from functools import partial
        extract_one = partial(_extract_one, b=b, model=model, tokenizer=tokenizer, close=close, store=store, inputs=inputs)
        for result in executor.map(extract_one, rows):
            if result is None:
                continue
            outcome, spans, failure = result
            if outcome is not None:
                outcomes.append(outcome)
                sentences.extend(spans)
            if failure:
                failures.append(failure)
    folder = ROOT/model/'extraction'
    require(not folder.exists(), 'whole extraction requires a fresh destination')
    folder.mkdir(parents=True, mode=0o700)
    pd.DataFrame(outcomes, columns=list(outcomes[0]) if outcomes else ['uid']).to_parquet(folder/'outcomes.parquet', index=False)
    pd.DataFrame(sentences, columns=list(sentences[0]) if sentences else ['uid']).to_parquet(folder/'sentences.parquet', index=False)
    return publish(folder/'EXTRACTION.json', {'model': model, 'manifest_sha256': value['sha256'],
                  'planned': len(rows), 'extracted': len(outcomes), 'failures': failures, 'sentences': len(sentences)})


def load_table() -> pd.DataFrame:
    """Join every manifest row and fixed-denominator lexical occupancy with bounds."""
    from .lexical import LEXICAL, predict
    rows = []
    for model in MODELS:
        value = manifest(model)
        folder = ROOT/model/'extraction'
        observed = pd.read_parquet(folder/'outcomes.parquet') if (folder/'outcomes.parquet').exists() else pd.DataFrame()
        by_uid = {r['uid']: r for r in observed.to_dict('records')}
        require(len(by_uid) == len(observed), 'duplicate whole extracted UID')
        require(set(by_uid) <= {r['uid'] for r in value['records']}, 'whole extraction UID outside manifest')
        sentences = pd.read_parquet(folder/'sentences.parquet') if (folder/'sentences.parquet').exists() else pd.DataFrame()
        groups = {}
        if not sentences.empty:
            frozen = verified(ROOT/'RELEASE.json')['classifier']
            # Full solutions have far more sentences than the old 512-token windows.
            sentences = pd.concat([predict(sentences.iloc[start:start+10000], LEXICAL/'readout-v1', frozen=frozen)
                                   for start in range(0, len(sentences), 10000)], ignore_index=True)
            groups = {uid: g.to_dict('records') for uid, g in sentences.groupby('uid')}
        for assignment in value['records']:
            row = {**assignment, **by_uid.get(assignment['uid'], {
                'outcome_status': 'missing', 'generated_tokens': np.nan, 'reasoning_tokens': np.nan,
                'reasoning_end': None, 'finish_reason': 'unknown', 'strict_correct': None,
                'answer_present': np.nan, 'repetition_16gram': np.nan, 'dose_json': '{}',
            })}
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
            require(covered <= observed_tokens, 'whole occupancy exceeds replay reasoning ownership')
            natural = row['outcome_status'] == 'observed' and (
                row.get('reasoning_end') is not None and not pd.isna(row.get('reasoning_end')) or
                row.get('finish_reason') == 'stop')
            unknown = max(0, observed_tokens-covered) if natural else denominator-covered
            for class_name in CLASSES:
                lower, upper = counts[class_name]/denominator, (counts[class_name]+unknown)/denominator
                row['occ_'+class_name] = (lower+upper)/2
                row['occ_'+class_name+':lower'], row['occ_'+class_name+':upper'] = lower, upper
            rows.append(row)
    return pd.DataFrame(rows)


def contrast(target: dict, control: dict, metric: str, benefit=False) -> tuple[float, float, float]:
    """Subtract two bounded endpoints, reversing bound order for token benefit."""
    lower = target[metric+':lower']-control[metric+':upper']
    upper = target[metric+':upper']-control[metric+':lower']
    if benefit:
        lower, upper = -upper, -lower
    return (lower+upper)/2, lower, upper


def family_contrasts(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average fixed efficiency cells equally within model, then models 0.5/0.5."""
    cells, native_repeats = [], []
    for (model, family), members in table.groupby(['model', 'family']):
        records = members.to_dict('records')
        native = next(row for row in records if row['arm'] == 'native')
        repeat = next(row for row in records if row['arm'] == 'native_repeat')
        noise = {'family': family, 'model': model}
        for metric, name in (('log_reasoning', 'tokens'), ('accuracy', 'accuracy')):
            for suffix, value in zip(('', ':lower', ':upper'), contrast(repeat, native, metric)):
                noise[name+suffix] = value
        native_repeats.append(noise)
        for operation in ('promote', 'suppress'):
            for class_name in (CLASSES if operation == 'promote' else SUPPRESS[model]):
                arms = {r['arm']: r for r in records if r['class'] == class_name and r['operation'] == operation}
                target, random = arms['target_'+operation], arms['random_'+operation]
                cell = {'family': family, 'model': model, 'class': class_name, 'operation': operation,
                        'efficiency_directed': operation == 'suppress' or class_name in ('Verify', 'Monitor')}
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


def primary_frame(cells: pd.DataFrame, comparison: str = 'target_random') -> pd.DataFrame:
    """Return six columns whose model and pooled estimates share one envelope."""
    selected = cells[cells.efficiency_directed]
    metrics = [comparison+':'+metric+suffix for metric in ('tokens', 'accuracy') for suffix in ('', ':lower', ':upper')]
    grouped = selected.groupby(['family', 'model'])[metrics].mean()
    families = sorted(cells.family.unique())
    frame = pd.DataFrame({'family': families}).set_index('family')
    for metric in ('tokens', 'accuracy'):
        for suffix in ('', ':lower', ':upper'):
            for model in MODELS:
                frame[model+':'+metric+suffix] = grouped[comparison+':'+metric+suffix].xs(model, level='model')
            frame['pooled:'+metric+suffix] = .5*frame['gpt:'+metric+suffix]+.5*frame['qwen36:'+metric+suffix]
    return frame.reset_index()


def planning(models=MODELS) -> dict:
    """Recompute native-only family SDs; mark transported MDEs SUSPECTED."""
    from moe_exp.research_campaign_v3_r6.inputs import read_source
    from moe_exp.schemas import TraceRecord
    from moe_exp.correlation_pipeline.dynamics.classes import saved_layout
    from scipy.stats import norm
    b = driver.binding()
    results, native_residuals = {}, {}
    for model in models:
        sources = {r['attempt_id']: r for r in load(Path(b['source_campaign'])/model/'SOURCES.json')['records']}
        natives = [r for r in verified(driver.DRIVER/model/'ASSIGNMENTS.json')['records'] if r['arm'] == 'native']
        values = []
        for row in natives:
            trace = read_source(sources[row['source']['attempt_id']]['source_location'])
            layout = saved_layout(TraceRecord(**trace))
            correct = trace.get('is_correct')
            if correct is None:
                correct = trace.get('metadata', {}).get('is_correct')
            values.append({'family': row['family'], 'benchmark': row['benchmark'],
                           'log_reasoning': math.log1p(len(layout['reasoning_tokens'] or [])),
                           'correct': float(correct) if correct is not None else np.nan})
        frame = pd.DataFrame(values)
        # Remove benchmark means, retaining family variation within the declared strata.
        residual = frame[['log_reasoning', 'correct']]-frame.groupby('benchmark')[['log_reasoning', 'correct']].transform('mean')
        residual.index = frame.family
        native_residuals[model] = residual
        token_sd, accuracy_sd = residual.std(ddof=1).tolist()
        critical = float(norm.ppf(1-.05/12)+norm.ppf(.8))
        results[model] = {'source': 'main-design original native token replay; benchmark-centered family SD',
                          'families': len(frame), 'log_reasoning_sd': token_sd,
                          'accuracy_sd': accuracy_sd if np.isfinite(accuracy_sd) else None,
                          'accuracy_observed': int(frame.correct.notna().sum()),
                          'composite_claim_reproduced': False,
                          'fallback': 'native-only SDs; no invented composite treatment covariance',
                          'planning_z': critical, 'mde_log_tokens': critical*token_sd/math.sqrt(129),
                          'mde_accuracy': critical*accuracy_sd/math.sqrt(129) if np.isfinite(accuracy_sd) else None,
                          'transport_status': 'SUSPECTED', 'native_values_sha256': digest(values)}
    if set(MODELS) <= set(results):
        critical = results['gpt']['planning_z']
        results['pooled'] = _pooled_native_planning(native_residuals, critical)
    return {'models': results, 'hypothesis_cells_frozen_before_production': True,
            'note': 'Claimed composite SDs 0.19/0.16 and 0.10/0.07 are not native-only SDs; transport is unverified.'}


def _pooled_native_planning(residuals: dict, critical: float) -> dict:
    """Use paired native family covariance in the explicitly suspected planning fallback."""
    require(residuals['gpt'].index.is_unique and residuals['qwen36'].index.is_unique and
            set(residuals['gpt'].index) == set(residuals['qwen36'].index), 'native planning families differ')
    pooled = .5*residuals['gpt']+.5*residuals['qwen36']
    token_sd, accuracy_sd = pooled.std(ddof=1).tolist()
    if not pooled.correct.notna().all():
        accuracy_sd = np.nan
    return {
        'transport_status': 'SUSPECTED', 'model_weights': [.5, .5],
        'log_reasoning_sd': token_sd, 'accuracy_sd': accuracy_sd if np.isfinite(accuracy_sd) else None,
        'accuracy_observed': int(pooled.correct.notna().sum()),
        'mde_log_tokens': critical*token_sd/math.sqrt(len(pooled)),
        'mde_accuracy': critical*accuracy_sd/math.sqrt(len(pooled)) if np.isfinite(accuracy_sd) else None,
        'covariance_role': 'paired native family residuals; no independence assumption; treatment-variance transport unverified',
    }


def _total_bounds(unit: dict):
    """Bound historical log-total length using this request's actual allowance."""
    if unit['outcome_status'] != 'observed':
        return 0., math.log(unit['total_allowance'])
    value = math.log(max(1, unit['generated_tokens']))
    return value, math.log(unit['total_allowance']) if unit['finish_reason'] == 'length' else value


def exploratory_link(cells: pd.DataFrame, table: pd.DataFrame, boot) -> dict:
    """Transfer frozen beta signs to promote cells with both declared token comparands."""
    from .effects import summarize
    prereg = verified(driver.DRIVER/'EFFICIENCY_LINK_PREREG.json')
    frozen = {(r['model'], r['class']): r for r in prereg['cells'] if r['model'] in MODELS}
    details, link_cells = [], {'log_reasoning': [], 'log_total': []}
    for model in MODELS:
        for class_name in CLASSES:
            sub = cells[(cells.model == model) & (cells['class'] == class_name) & (cells.operation == 'promote')]
            row = frozen.get((model, class_name), {})
            signs = row.get('tok_reason', {})
            token_summary = summarize(boot, sub, ['target_random:tokens'], 'exploratory-link')
            if signs.get('beta') is not None:
                link_cells['log_reasoning'].append((signs, sub.assign(
                    delta=-sub['target_random:tokens'],
                    **{'delta:lower': -sub['target_random:tokens:upper'],
                       'delta:upper': -sub['target_random:tokens:lower']})))
            # Historical log(total), without the new log1p reasoning transform.
            total_rows = []
            for family, group in table[(table.model == model) & (table['class'] == class_name) &
                                       (table.operation == 'promote')].groupby('family'):
                arms = {r['arm']: r for r in group.to_dict('records')}
                target, random = arms['target_promote'], arms['random_promote']
                tl, tu = _total_bounds(target)
                rl, ru = _total_bounds(random)
                total_rows.append({'family': family, 'delta': (tl-ru+tu-rl)/2,
                                   'delta:lower': tl-ru, 'delta:upper': tu-rl})
            if signs.get('beta') is not None:
                link_cells['log_total'].append((signs, pd.DataFrame(total_rows)))
            details.append({'model': model, 'class': class_name, 'frozen_tok_beta': signs,
                            'log_reasoning_token_benefit': token_summary,
                            'log_total_target_minus_random': summarize(boot, pd.DataFrame(total_rows), ['delta'],
                                                                      'exploratory-link-total')})
    return {'wording': LINK_WORDING, 'prereg_sha256': prereg['sha256'], 'cells': details,
            'statistics': {metric: link_statistics(boot, records) for metric, records in link_cells.items()},
            'claim': 'exploratory sign transfer; no preregistered replication claim'}


def _rank_correlation(centered_predictor, values):
    """Evaluate cell-wise Spearman correlations for points or shared bootstrap rows."""
    from scipy.stats import rankdata
    ranked = rankdata(values, axis=-1)
    ranked -= ranked.mean(axis=-1, keepdims=True)
    return (ranked*centered_predictor).sum(-1)/np.sqrt((ranked**2).sum(-1)*(centered_predictor**2).sum())


def link_statistics(boot, records: list[tuple]) -> dict:
    """Compute exploratory rho and sign agreement only for point-identified cells."""
    from scipy.stats import binomtest, rankdata
    if any((frame['delta:lower'] != frame['delta:upper']).any() for _, frame in records):
        return {'status': 'unqualified_partial_identification', 'cells': len(records),
                'reason': 'Capped or missing cells retain their bounds; no midpoint correlation is observed.'}
    estimates = [boot.estimate(frame, 'delta') for _, frame in records]
    if any(not value or value['status'] != 'qualified' for value in estimates):
        return {'status': 'unqualified_missing_benchmark', 'cells': len(records)}
    predictors = np.asarray([record['beta'] for record, _ in records])
    points = np.asarray([value['estimate'] for value in estimates])
    draws = np.column_stack([value['draws'] for value in estimates])
    x = rankdata(predictors)
    x -= x.mean()
    signed = np.asarray([record.get('sign') in ('+', '-') for record, _ in records])
    predicted = np.asarray([1 if record.get('sign') == '+' else -1 for record, _ in records])
    agreements = (np.sign(points[signed]) == predicted[signed]) & (points[signed] != 0)
    agreement_draws = ((np.sign(draws[:, signed]) == predicted[signed]) & (draws[:, signed] != 0)).mean(-1)
    rho_draws = _rank_correlation(x, draws)
    return {'status': 'qualified_exploratory', 'spearman': float(_rank_correlation(x, points)),
            'spearman_ci': np.nanquantile(rho_draws, [.025, .975]).tolist(),
            'sign_agreement': float(agreements.mean()),
            'sign_agreement_ci': np.quantile(agreement_draws, [.025, .975]).tolist(),
            'signed_cells': int(signed.sum()), 'cells': len(records), 'draws': len(boot.weights),
            'binomial_p': float(binomtest(int(agreements.sum()), int(signed.sum()), .5).pvalue)}


def arm_summaries(boot, table: pd.DataFrame) -> tuple[list, dict]:
    """Summarize arm levels with the shared draws and normalize additive dose counters."""
    from .effects import summarize
    levels_by_arm, dose_by_arm = [], {}
    for (model, arm), group in table.groupby(['model', 'arm']):
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
        levels_by_arm.append({'model': model, 'arm': arm, 'planned': len(group), 'complete': len(observed),
                                 'cap_rate': float((observed.finish_reason == 'length').mean()) if len(observed) else None,
                                 'answer_presence': float(observed.answer_present.mean()) if len(observed) else None,
                                 'repetition_16gram': float(observed.repetition_16gram.mean()) if len(observed) else None,
                                 'mean_total_tokens': float(observed.generated_tokens.mean()) if len(observed) else None,
                                 'bounded_family_bootstrap': summarize(boot, levels, ['cap_rate', 'answer_presence',
                                     'repetition_16gram', 'total_tokens'], 'whole-levels:'+model+':'+arm)})
        totals = {}
        for encoded in observed.get('dose_json', []):
            for layer, counters in json.loads(encoded).items():
                aggregate = totals.setdefault(layer, Counter())
                aggregate.update(counters)
        dose_by_arm[model+'|'+arm] = {layer: {**dict(counters),
            'changed_membership_fraction': counters['changed_membership']/max(1, counters['active_positions']),
            'native_target_weight_mean': counters['native_target_weight']/max(1, counters['active_positions']),
            'executed_target_weight_mean': counters['executed_target_weight']/max(1, counters['active_positions']),
            'l1_mean': counters['l1']/max(1, counters['active_positions'])} for layer, counters in totals.items()}
    return levels_by_arm, dose_by_arm


def analyze() -> dict:
    """Publish the six primary bounded contrasts and mandatory companions from fresh files."""
    from .effects import Bootstrap, summarize
    folder = ROOT/'analysis'
    require(not folder.exists(), 'whole analysis requires a fresh directory')
    table = load_table()
    cells, noise = family_contrasts(table)
    benchmarks = {r['family']: r['benchmark'] for r in load(driver.ROOT/'ALLOCATION.json')['records']}
    boot = Bootstrap(sorted(benchmarks), benchmarks, draws=10000, seed=20261015)
    names = [model+':'+metric for model in (*MODELS, 'pooled') for metric in ('tokens', 'accuracy')]
    result = {'models': list(MODELS), 'bootstrap_draws': 10000, 'seed': 20261015,
              'release_sha256': verified(ROOT/'RELEASE.json')['sha256'],
              'manifest_sha256': {model: manifest(model)['sha256'] for model in MODELS},
              'scope': {model: manifest(model)['scope'] for model in MODELS},
              'primary': summarize(boot, primary_frame(cells), names, 'whole-primary-six'),
              'companions': {comparison: summarize(boot, primary_frame(cells, comparison), names,
                                                  'whole-companions:'+comparison)
                             for comparison in ('target_native', 'random_native')},
              'native_repeat_minus_native': {model: summarize(boot, noise[noise.model == model],
                                                            ['tokens', 'accuracy'], 'whole-native-noise')
                                             for model in MODELS}}
    mechanism = {}
    for model in MODELS:
        for operation in ('promote', 'suppress'):
            for class_name in CLASSES:
                if operation == 'suppress' and class_name not in SUPPRESS[model]:
                    mechanism['|'.join((model, operation, class_name))] = {
                        'status': 'not_in_design', 'observed_classes': list(CLASSES),
                        'reason': 'This suppress class was not assigned; no zero effect is imputed.'}
                    continue
                sub = cells[(cells.model == model) & (cells.operation == operation) & (cells['class'] == class_name)]
                columns = [comparison+':response:'+observed for comparison in
                           ('target_random', 'target_native', 'random_native') for observed in CLASSES]
                mechanism['|'.join((model, operation, class_name))] = summarize(boot, sub, columns, 'whole-signed-response')
    result['mechanism'] = mechanism
    result['exploratory_link'] = exploratory_link(cells, table, boot)
    result['levels'], result['dose'] = arm_summaries(boot, table)
    result['verdict'] = {}
    for model in (*MODELS, 'pooled'):
        tokens = result['primary'][model+':tokens'].get('bound_simultaneous', [-np.inf, np.inf])
        accuracy = result['primary'][model+':accuracy'].get('bound_simultaneous', [-np.inf, np.inf])
        result['verdict'][model] = ('efficient_steering' if tokens[0] > 0 and accuracy[0] > -.05 else
                                    'speed_accuracy_tradeoff' if tokens[0] > 0 and accuracy[0] <= -.05 else
                                    'inconclusive; assess native companions for generic disruption')
    folder.mkdir(parents=True, mode=0o700)
    table.drop(columns=['source', 'actions'], errors='ignore').to_parquet(folder/'continuations.parquet', index=False)
    cells.to_parquet(folder/'cells.parquet', index=False)
    return publish(folder/'EFFECTS.json', result)

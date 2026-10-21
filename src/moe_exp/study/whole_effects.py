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
import os
from pathlib import Path
from threading import local

import numpy as np
import pandas as pd

from .common import CLASSES, digest, load, publish, require, verified
from . import driver
from .whole import ROOT, MODELS, directions, efficiency_directed, manifest, bound_store, repetition_rate

LINK_WORDING = ('The frozen preregistered test (activation arms, central strength, log total tokens, main design) '
                'is not estimable in this design. We report an exploratory transfer of the frozen beta_tok signs '
                'to the promote arms.')
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
    from .outcomes import record
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


def _initialize_extractor(local_path: str, model: str) -> None:
    """Give every thread its own fast tokenizer and closure sequences."""
    from transformers import AutoTokenizer
    from .runtime import closure
    _EXTRACTOR.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    _EXTRACTOR.close = closure(_EXTRACTOR.tokenizer, model)


def _extract_with_worker(row: dict, *, b: dict, model: str, store, inputs):
    """Use only the current worker's tokenizer while sharing immutable prompt inputs."""
    return _extract_one(row, b=b, model=model, tokenizer=_EXTRACTOR.tokenizer,
                        close=_EXTRACTOR.close, store=store, inputs=inputs)



def extract(model: str) -> dict:
    """Extract whatever completed, preserving missing rows for the subsequent planned join."""
    from transformers import AutoTokenizer
    from .runtime import prompt_prefix
    value = manifest(model)
    store = bound_store(model, value['dose'])
    b = driver.binding()
    tokenizer = AutoTokenizer.from_pretrained(b['models'][model]['local_path'], local_files_only=True)
    rows = value['records']
    inputs = {row['family']: prompt_prefix(b, model, row, tokenizer) for row in rows if row['arm'] == 'native'}


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
            for class_name in (CLASSES if operation == 'promote' else directions(model)['suppress']):
                arms = {r['arm']: r for r in records if r['class'] == class_name and r['operation'] == operation}
                target, random = arms['target_'+operation], arms['random_'+operation]
                efficiency = efficiency_directed(model, operation, class_name)
                nominated = all(r.get('nomination_status', 'nominated') == 'nominated' for r in (target, random))
                require(nominated or not efficiency, 'efficiency-directed cell is not nominated')
                if not nominated:
                    continue
                cell = {'family': family, 'model': model, 'class': class_name, 'operation': operation,
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


def primary_frame(cells: pd.DataFrame, comparison: str = 'target_random') -> pd.DataFrame:
    """Return six columns whose model and pooled estimates share one envelope."""
    selected = cells[cells.efficiency_directed]
    if 'nomination_status' in selected:
        require((selected.nomination_status == 'nominated').all(), 'primary composite contains an unnominated cell')
    metrics = [comparison+':'+metric+suffix for metric in ('tokens', 'accuracy') for suffix in ('', ':lower', ':upper')]
    grouped = selected.groupby(['family', 'model'])[metrics].mean()
    families = sorted(cells.family.unique())
    frame = pd.DataFrame({'family': families}).set_index('family')
    for metric in ('tokens', 'accuracy'):
        for suffix in ('', ':lower', ':upper'):
            for model in MODELS:
                frame[model+':'+metric+suffix] = grouped[comparison+':'+metric+suffix].xs(model, level='model')
            # Equal model weights; a family absent for one model (LFM2.5 has 112) averages the others.
            frame['pooled:'+metric+suffix] = frame[[m+':'+metric+suffix for m in MODELS]].mean(axis=1)
    return frame.reset_index()


def planning(models=MODELS) -> dict:
    """Transport within-family native-repeat noise to the planned composite SD.

    Args:
        models: Models with saved main-study native and native_repeat outcomes.

    Returns:
        SUSPECTED planning values; absent paired evidence is disclosed and never
        gates science. Accuracy uses driver outcomes, never trace is_correct.
    """
    from .effects import _read_outcome_table
    from scipy.stats import norm
    results, native_residuals = {}, {}
    for model in models:
        folder = driver.bulk()/model/'outcomes'
        outcomes = _read_outcome_table(folder, 'outcomes', 'main')
        differences = native_repeat_differences(outcomes)
        cell_count = len(directions(model)['promote'])+len(directions(model)['suppress'])
        composite = differences/math.sqrt(cell_count)
        native_residuals[model] = composite
        difference_sd = differences.std(ddof=1)
        token_sd, accuracy_sd = composite.std(ddof=1).tolist()
        critical = float(norm.ppf(1-.05/12)+norm.ppf(.8))
        results[model] = {
            'source': str(folder),
            'accuracy_source': 'main-study driver strict_correct outcomes; legacy trace is_correct excluded',
            'noise_reference': 'native_repeat minus native within family',
            'formula': 'sigma_w = SD(d)/sqrt(2); composite_SD = sigma_w*sqrt(2/m)',
            'efficiency_cells': cell_count,
            'paired_families': differences.notna().sum().to_dict(),
            'within_family_sigma_w': (difference_sd/math.sqrt(2)).to_dict(),
            'log_reasoning_sd': token_sd if np.isfinite(token_sd) else None,
            'accuracy_sd': accuracy_sd if np.isfinite(accuracy_sd) else None,
            'planning_z': critical,
            'mde_log_tokens': critical*token_sd/math.sqrt(129) if np.isfinite(token_sd) else None,
            'mde_accuracy': critical*accuracy_sd/math.sqrt(129) if np.isfinite(accuracy_sd) else None,
            'transport_status': 'SUSPECTED',
            'paired_values_sha256': digest(differences.reset_index().to_dict('records')),
        }
    if set(MODELS) <= set(results):
        critical = results[MODELS[0]]['planning_z']
        results['pooled'] = _pooled_native_planning(native_residuals, critical)
    return {'models': results, 'hypothesis_cells_frozen_before_production': True,
            'note': 'Within-family paired noise, transported with the declared sqrt(2/m) approximation; no gate uses it.'}


def native_repeat_differences(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Return observed paired reasoning/accuracy differences from main driver outcomes."""
    native = outcomes[outcomes.arm.isin(('native', 'native_repeat'))].copy()
    require(not native.duplicated(['family', 'arm']).any(), 'main native planning identities are duplicated')
    closed = native.reasoning_end.notna()
    unknown = (native.finish_reason == 'length') & ~closed
    native['log_reasoning'] = np.log1p(native.reasoning_tokens).where(~unknown)
    native['correct'] = native.strict_correct.astype(float).where(native.finish_reason != 'length', 0.)
    columns = ['log_reasoning', 'correct']
    first = native[native.arm == 'native'].set_index('family')[columns]
    repeat = native[native.arm == 'native_repeat'].set_index('family')[columns]
    return repeat-first


def _pooled_native_planning(residuals: dict, critical: float) -> dict:
    """Average the scaled within-family noise differences with equal model weights."""
    require(all(frame.index.is_unique for frame in residuals.values()), 'native planning families are duplicated')
    pooled = pd.concat(residuals.values()).groupby(level=0).mean()
    token_sd, accuracy_sd = pooled.std(ddof=1).tolist()
    if not pooled.correct.notna().all():
        accuracy_sd = np.nan
    return {
        'transport_status': 'SUSPECTED', 'model_weights': 'equal over models present for each family',
        'log_reasoning_sd': token_sd, 'accuracy_sd': accuracy_sd if np.isfinite(accuracy_sd) else None,
        'accuracy_observed': int(pooled.correct.notna().sum()),
        'mde_log_tokens': critical*token_sd/math.sqrt(len(pooled)),
        'mde_accuracy': critical*accuracy_sd/math.sqrt(len(pooled)) if np.isfinite(accuracy_sd) else None,
        'covariance_role': 'paired within-family differences scaled by sqrt(m); model covariance retained; transport SUSPECTED',
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
    table = table[table.get('nomination_status', pd.Series('nominated', index=table.index)) == 'nominated']
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
                if not {'target_promote', 'random_promote'} <= arms.keys():
                    continue  # a dropped promote cell has no executed pair for this family
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
            'inference': {'draws': 10000, 'seed': 20261015, 'shared_with_primary': True,
                          'preregistered_draws': 2000, 'preregistered_seed': 20261017,
                          'status': 'exploratory inference differs from the preregistration'},
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
    table = table[table.get('nomination_status', pd.Series('nominated', index=table.index)) == 'nominated']
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
    extraction_failures = []
    for model in MODELS:
        path = ROOT/model/'extraction/EXTRACTION.json'
        if path.exists():
            extraction_failures.extend({'model': model, **failure} for failure in verified(path)['failures'])
    design_drops = table[table.get('nomination_status', pd.Series('nominated', index=table.index)) != 'nominated']
    result = {'models': list(MODELS), 'bootstrap_draws': 10000, 'seed': 20261015,
              'release_sha256': verified(ROOT/'RELEASE.json')['sha256'],
              'manifest_sha256': {model: manifest(model)['sha256'] for model in MODELS},
              'scope': {model: manifest(model)['scope'] for model in MODELS},
              'extraction_failures': {'count': len(extraction_failures), 'units': extraction_failures},
              'unit_failures': table.loc[table.get('unit_failure_reason', pd.Series(index=table.index, dtype=object)).notna(),
                                         ['uid', 'model', 'family', 'unit_failure_reason']].to_dict('records')
                               if 'unit_failure_reason' in table else [],
              'not_in_design': design_drops[['model', 'fold', 'class', 'operation']].drop_duplicates().to_dict('records'),
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
                if operation == 'suppress' and class_name not in directions(model)['suppress']:
                    mechanism['|'.join((model, operation, class_name))] = {
                        'status': 'not_in_design', 'observed_classes': list(CLASSES),
                        'reason': 'This suppress class was not assigned; no zero effect is imputed.'}
                    continue
                sub = cells[(cells.model == model) & (cells.operation == operation) & (cells['class'] == class_name)]
                if sub.empty:
                    mechanism['|'.join((model, operation, class_name))] = {
                        'status': 'not_in_design', 'reason': 'No nominated cell; excluded from composites.'}
                    continue
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
    receipt = publish(folder/'EFFECTS.json', result)
    write_report(result, folder/'REPORT.md')
    return receipt


def write_report(result: dict, path: Path) -> None:
    """Write the required primary, manipulation and failure disclosures beside effects.

    Args:
        result: The bounded analysis result written to EFFECTS.json.
        path: Fresh report destination in the same analysis folder.

    Returns:
        None; detailed unit failures and bootstrap components remain in EFFECTS.json.
    """
    lines = ['# Whole-reasoning results', '',
             f"Extraction failures: {result['extraction_failures']['count']}; "
             f"sentence-coverage unit failures: {len(result['unit_failures'])}. "
             'Their units and reasons are listed in EFFECTS.json and retain unknown bounds.', '',
             '| Contrast | Estimate | Simultaneous bounds |', '|---|---:|---|']
    for name, value in result['primary'].items():
        lines.append(f"| {name} | {value.get('estimate')} | {value.get('bound_simultaneous')} |")
    lines += ['', '| Model / arm | Native frequency | Changed membership | L1 | Deactivated native weight |',
              '|---|---:|---:|---:|---:|']
    for level in result['levels']:
        check = level['manipulation_check']
        values = [check[key] for key in ('mean_native_frequency_steered_experts', 'changed_membership_fraction',
                                        'l1', 'deactivated_native_weight')]
        lines.append('| '+level['model']+' / '+level['arm']+' | '+' | '.join(map(str, values))+' |')
    lines += ['', LINK_WORDING, '',
              'The exploratory link uses the shared 10,000-draw family bootstrap with seed 20261015, '
              'rather than the preregistered 2,000 draws with seed 20261017.', '',
              'Non-nominated cells are not_in_design and do not enter composites. '
              'Targets retain the larger native-frequency perturbation. Historical generic disruption '
              'lengthened reasoning and reduced accuracy, so this residual mismatch works against an efficiency claim.']
    path.write_text('\n'.join(lines)+'\n')

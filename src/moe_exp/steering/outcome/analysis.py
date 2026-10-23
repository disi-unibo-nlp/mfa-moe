"""Effects of the outcome-selected nudge and of the stop-thinking-earlier baseline.

Every number is a within-family (same question) contrast with the bounds of whole_effects: a run that
hits its allowance or never finishes keeps an interval, never a guessed value.

- ``target_random``: nudged experts against matched random experts at the same delta (primary).
- ``target_native`` / ``random_native``: each nudge against unedited routing (the base release's native
  run of the same family); ``noise`` is native_repeat - native, the rerun noise of an unedited run.
- ``budget_native``: the length-limit baseline at each fraction against the full native run.
- ``target_budget``: target accuracy minus the accuracy the family's length-limit curve reaches at the
  target's own reasoning length. Positive means the nudge buys more accuracy than simply stopping
  the reasoning at the same length.

Token contrasts are log(1+T_control) - log(1+T_treated): positive means shorter reasoning.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from moe_exp.io import file_sha, publish, require, verified

from moe_exp.steering.outcome import REPORTS

NATIVES = REPORTS/'whole-reasoning-20261021-r17/analysis-combined'
METRICS = (('log_reasoning', 'tokens'), ('accuracy', 'accuracy'))
COUNTERS = ('active_positions', 'native_shift_members', 'executed_shift_members',
            'native_shift_weight', 'executed_shift_weight')


def release_rows(model: str) -> pd.DataFrame:
    """This release's nudge and budget rows, joined to their extracted outcomes, with bounds."""
    from moe_exp.steering.whole import ROOT, manifest
    from moe_exp.steering.whole_effects import unit_bounds
    value = manifest(model)
    folder = ROOT/model/'extraction'
    observed = pd.read_parquet(folder/'outcomes.parquet') if (folder/'outcomes.parquet').exists() else pd.DataFrame()
    by_uid = {r['uid']: r for r in observed.to_dict('records')} if len(observed) else {}
    missing = {'outcome_status': 'missing', 'finish_reason': 'unknown', 'reasoning_tokens': np.nan,
               'reasoning_end': None, 'strict_correct': None, 'repetition_16gram': np.nan, 'dose_json': '{}'}
    rows = []
    for row in value['records']:
        if row['nomination_status'] != 'nominated':
            continue
        merged = {**{k: v for k, v in row.items() if k not in ('actions', 'source', 'budget_prefix')},
                  **by_uid.get(row['uid'], missing)}
        merged.update(unit_bounds(merged))
        if row['arm'] == 'budget':
            # The reasoning length of a length-limit row is its cut, by construction.
            cut = math.log1p(row['budget_cut'])
            merged.update({'log_reasoning': cut, 'log_reasoning:lower': cut, 'log_reasoning:upper': cut})
        rows.append(merged)
    return pd.DataFrame(rows)


def native_rows(model: str, dose: str) -> pd.DataFrame:
    """Native and native-repeat rows of the base releases, from the sealed final whole-reasoning table."""
    columns = ['model', 'dose', 'arm', 'family', 'benchmark', 'reasoning_tokens', 'finish_reason',
               *[m+s for m, _ in METRICS for s in ('', ':lower', ':upper')]]
    table = pd.read_parquet(NATIVES/'continuations.parquet', columns=columns)
    return table[(table.model == model) & (table.dose == dose) & table.arm.isin(['native', 'native_repeat'])]


def contrast(treated: dict, control: dict, metric: str) -> tuple[float, float, float]:
    """Point, lower and upper of treated - control (reversed for tokens, so positive = shorter)."""
    from moe_exp.steering.whole_effects import contrast as bounded
    return bounded(treated, control, metric, metric == 'log_reasoning')


def curve_range(points: list[tuple[float, float, float]], low: float, high: float) -> tuple[float, float]:
    """Lowest and highest accuracy a family's length-limit curve allows for a length in [low, high].

    Args:
        points: (log length, accuracy lower, accuracy upper) of the budget fractions and the native run.
        low, high: Bounds of the target's log reasoning length.

    The curve is linear between points and flat beyond the ends, so its extremes over an interval lie at
    the interval ends or at curve points inside it.
    """
    points = sorted(points)
    xs = [p[0] for p in points]
    grid = [low, high, *[x for x in xs if low < x < high]]
    lowest = min(float(np.interp(x, xs, [p[1] for p in points])) for x in grid)
    highest = max(float(np.interp(x, xs, [p[2] for p in points])) for x in grid)
    return lowest, highest


def family_frame(model: str, ours: pd.DataFrame, natives: pd.DataFrame) -> pd.DataFrame:
    """One row per family with every contrast of this model as point/lower/upper columns."""
    native = {f: g.set_index('arm') for f, g in natives.groupby('family')}
    records = []
    for family, members in ours.groupby('family'):
        if family not in native or set(native[family].index) != {'native', 'native_repeat'}:
            continue  # a family whose native pair is unknown stays out (reported as a count)
        base, repeat = native[family].loc['native'].to_dict(), native[family].loc['native_repeat'].to_dict()
        record = {'family': family}

        def put(name, treated, control, metrics=METRICS):
            for metric, label in metrics:
                for suffix, value in zip(('', ':lower', ':upper'), contrast(treated, control, metric)):
                    record[f'{model}:{name}:{label}{suffix}'] = value

        put('noise', repeat, base)
        rows = members.to_dict('records')
        curve = [(base['log_reasoning'], base['accuracy:lower'], base['accuracy:upper'])]
        for row in rows:
            if row['arm'] == 'budget':
                put(f"{row['class']}:budget_native", row, base)
                curve.append((row['log_reasoning'], row['accuracy:lower'], row['accuracy:upper']))
        for delta in sorted({r['class'] for r in rows if r['arm'] != 'budget'}):
            arms = {r['arm']: r for r in rows if r['class'] == delta}
            target, random = arms['target_nudge'], arms['random_nudge']
            put(f'{delta}:target_random', target, random)
            put(f'{delta}:target_native', target, base)
            put(f'{delta}:random_native', random, base)
            if len(curve) > 1:
                lowest, highest = curve_range(curve, target['log_reasoning:lower'], target['log_reasoning:upper'])
                lower, upper = target['accuracy:lower']-highest, target['accuracy:upper']-lowest
                name = f'{model}:{delta}:target_budget:accuracy'
                record.update({name: (lower+upper)/2, name+':lower': lower, name+':upper': upper})
        records.append(record)
    return pd.DataFrame(records)


def manipulation(ours: pd.DataFrame) -> list[dict]:
    """How much routing the nudged experts held before and after the nudge, per arm and delta."""
    result = []
    for (arm, delta), group in ours[ours.arm != 'budget'].groupby(['arm', 'class']):
        totals = dict.fromkeys(COUNTERS, 0.)
        for text in group.dose_json.dropna():
            for counters in json.loads(text).values():
                for key in COUNTERS:
                    totals[key] += float(counters.get(key, 0.))
        kept = totals['executed_shift_members']/totals['native_shift_members'] if totals['native_shift_members'] else None
        result.append({'arm': arm, 'class': delta, 'rows': len(group), **totals, 'kept_slot_share': kept})
    return result


def levels(ours: pd.DataFrame, natives: pd.DataFrame) -> list[dict]:
    """Plain per-arm levels: completed rows, mean reasoning tokens, accuracy and length-limit rate."""
    result = []
    frames = [natives.assign(**{'class': None}), ours]
    for frame in frames:
        for (arm, name), group in frame.groupby(['arm', frame['class'].fillna('-')]):
            done = group[group.finish_reason.isin(['stop', 'length'])]
            result.append({'arm': arm, 'class': name, 'rows': len(group), 'completed': len(done),
                           'mean_reasoning_tokens': float(np.expm1(done.log_reasoning).mean()) if len(done) else None,
                           'accuracy': float(done.accuracy.mean()) if len(done) else None,
                           'length_limit_rate': float((done.finish_reason == 'length').mean()) if len(done) else None})
    return result


def analyze() -> dict:
    """Write EFFECTS_OUTCOME.json for every model of this release."""
    from moe_exp.steering.whole import MODELS, ROOT, manifest
    from moe_exp.steering.whole_effects import bootstrap, summarize_present
    effects = verified(NATIVES/'EFFECTS_WHOLE.json')
    frames, models = [], {}
    for model in MODELS:
        ours = release_rows(model)
        natives = native_rows(model, manifest(model)['native_dose'])
        frame = family_frame(model, ours, natives)
        frames.append(frame.set_index('family'))
        models[model] = {'levels': levels(ours, natives), 'manipulation': manipulation(ours),
                         'families': len(frame), 'families_without_native_pair':
                         int(ours.family.nunique()-len(frame))}
    frame = pd.concat(frames, axis=1).reset_index()
    boot = bootstrap()
    columns = [c for c in frame.columns if c != 'family']

    def family(kind):
        # Each family of contrasts gets its own simultaneous 95% envelope across models and deltas.
        names = [c for c in columns if f':{kind}:' in c]
        return summarize_present(boot, frame[['family', *names]], 'outcome-'+kind)

    result = {kind: family(kind) for kind in ('target_random', 'target_native', 'random_native', 'noise',
                                              'budget_native', 'target_budget')}
    folder = ROOT/'analysis'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    frame.to_parquet(folder/'families.parquet', index=False)
    return publish(folder/'EFFECTS_OUTCOME.json', {
        **result, 'models': models, 'native_effects_sha256': effects['sha256'],
        'native_table_sha256': file_sha(NATIVES/'continuations.parquet'),
        'families_sha256': file_sha(folder/'families.parquet')})

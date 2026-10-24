"""Choosing one of a question's finished attempts from its routing (ATTEMPT_SELECTION.json, CPU).

Each question of a capture release has six native attempts. A selector looks only at what a finished
attempt shows (its routing, its reasoning classes, its answer, its length) and picks one; it scores
the accuracy of the picked attempt. A capped or unscored pick counts as wrong.

Routing selectors:
- ``consensus``: the attempt whose routing (selections per reasoning token, all layers) is most similar
  to the question's other attempts (summed weighted Jaccard), after "Does the Same Token Mean the
  Same State?" (RAD).
- ``outcome_<set>``: the attempt that uses the set's wrong-linked experts least (within-question z of
  its usage on that set; the set was selected without this question), for the class-blind set and
  every class set.
- ``class_mix``: the attempt whose class occupancy (share of reasoning tokens per class, z within the
  question) lies furthest in the direction separating right from wrong attempts of the other questions.

Baselines: ``majority`` (most common answer; equivalence by math_verify, else normalised text; ties
broken by a fixed digest), ``random`` (expected accuracy of a uniform pick), ``shortest``, ``longest``.
Intervals: 95% family bootstrap, stratified by benchmark, on all questions and on mixed-outcome ones.
"""
from __future__ import annotations

import re

import numpy as np

from store import digest

from steering.outcome.select import BLIND, CLASS_NAMES, class_rates

DRAWS, SEED = 2000, 20261023


def same_answer(first: str, second: str) -> bool:
    """Two final answers are equal if math_verify says so, else if their normalised text matches."""
    def norm(text):
        text = re.sub(r'\\boxed\{(.*)\}', r'\1', text or '')
        return re.sub(r'[\s$]|\\[,!;:]|\.$', '', text).lower()
    if not first or not second:
        return False
    if norm(first) == norm(second):
        return True
    try:
        from generation.scoring import _math_verify_direct
        verdict, _ = _math_verify_direct(f'${first}$', second)
        return bool(verdict)
    except Exception:
        return False


def majority(answers: list[str | None], key: str) -> int | None:
    """Index of an attempt holding the most common answer (None when no attempt has an answer)."""
    groups = []
    for i, answer in enumerate(answers):
        if not answer:
            continue
        for group in groups:
            if same_answer(answers[group[0]], answer):
                group.append(i)
                break
        else:
            groups.append([i])
    if not groups:
        return None
    size = max(map(len, groups))
    tied = [g for g in groups if len(g) == size]
    group = min(tied, key=lambda g: digest([key, g[0]]))
    return group[0]


def jaccard_consensus(rates: np.ndarray) -> int:
    """Index of the attempt with the largest summed weighted-Jaccard similarity to the others."""
    flat = rates.reshape(len(rates), -1)
    total = np.zeros(len(flat))
    for i in range(len(flat)):
        lo, hi = np.minimum(flat[i], flat).sum(1), np.maximum(flat[i], flat).sum(1)
        similarity = np.divide(lo, hi, out=np.zeros_like(lo), where=hi > 0)
        total[i] = similarity.sum()-similarity[i]
    return int(np.argmax(total))


def zscore(block: np.ndarray) -> np.ndarray:
    """z-score across a question's attempts (zero where the attempts do not differ)."""
    spread = block.std(0, ddof=1) if len(block) > 1 else np.zeros(block.shape[1:])
    return np.divide(block-block.mean(0), spread, out=np.zeros_like(block, dtype=float), where=spread > 0)


def picks(usage: dict, arrays: dict, selection: dict) -> tuple[list[dict], list[str]]:
    """Every selector's pick for every question with at least two usable attempts."""
    records = usage['attempts']
    usable = np.array([r['usable'] for r in records])
    questions = np.array([r['question'] for r in records])
    correct = np.array([r['strict'] is True for r in records])
    known = np.array([r['strict'] is not None for r in records])
    all_rates, _ = class_rates(arrays['counts'], arrays['tokens'], 7)
    occupancy = np.divide(arrays['tokens'][:, :7], arrays['tokens'][:, [8]],
                          out=np.zeros(arrays['tokens'][:, :7].shape), where=arrays['tokens'][:, [8]] > 0)
    layers = usage['layers']
    rates_by_set = {BLIND: all_rates, **{c: class_rates(arrays['counts'], arrays['tokens'], i)[0]
                                         for i, c in enumerate(CLASS_NAMES)}}
    # Class-mix direction per question: mean over the other mixed questions of z(right) - z(wrong).
    directions = {}
    for question in sorted(set(questions[usable])):
        rows = np.flatnonzero((questions == question) & usable & known)
        if correct[rows].all() or not correct[rows].any():
            continue
        z = zscore(occupancy[rows])
        directions[question] = z[correct[rows]].mean(0)-z[~correct[rows]].mean(0)
    total_direction = sum(directions.values())
    selectors = ['consensus', 'class_mix', *[f'outcome_{s}' for s in (BLIND, *CLASS_NAMES)]]
    rows_out = []
    for question in sorted(set(questions[usable])):
        rows = np.flatnonzero((questions == question) & usable)
        if len(rows) < 2:
            continue
        record = {'question': question, 'benchmark': question.split('|')[0], 'attempts': len(rows),
                  'mixed': bool(correct[rows].any() and not correct[rows].all()),
                  'random': float(correct[rows].mean())}
        lengths = np.array([records[i]['reasoning_tokens'] for i in rows])
        record['shortest'] = bool(correct[rows[int(np.argmin(lengths))]])
        record['longest'] = bool(correct[rows[int(np.argmax(lengths))]])
        pick = majority([records[i]['model_answer'] for i in rows], question)
        record['majority'] = bool(correct[rows[pick]]) if pick is not None else False
        record['consensus'] = bool(correct[rows[jaccard_consensus(all_rates[rows])]])
        own = directions.get(question)
        direction = (total_direction-(own if own is not None else 0))/max(1, len(directions)-(own is not None))
        record['class_mix'] = bool(correct[rows[int(np.argmax(zscore(occupancy[rows]) @ direction))]])
        for name in (BLIND, *CLASS_NAMES):
            chosen = selection['sets'].get(name, {}).get('sets', {})
            targets = (chosen.get(question) or chosen.get('all') or {}).get('targets')
            if not targets:
                record[f'outcome_{name}'] = None
                continue
            index = [(layers.index(t['layer']), t['expert']) for t in targets]
            z = zscore(rates_by_set[name][rows])
            exposure = np.mean([z[:, l, e] for l, e in index], axis=0)
            record[f'outcome_{name}'] = bool(correct[rows[int(np.argmin(exposure))]])
        rows_out.append(record)
    return rows_out, selectors


def finite_mean(values: np.ndarray, index: np.ndarray) -> float:
    """Mean of the known entries at `index` (nan when none is known)."""
    chosen = values[index]
    return float(np.nanmean(chosen)) if np.isfinite(chosen).any() else float('nan')


def summarize(rows: list[dict], methods: list[str], rng: np.random.Generator) -> dict:
    """Accuracy of each method and its difference from majority and random, with bootstrap intervals."""
    if not rows:
        return {'questions': 0}
    benchmarks = sorted({r['benchmark'] for r in rows})
    strata = {b: [i for i, r in enumerate(rows) if r['benchmark'] == b] for b in benchmarks}
    table = {m: np.array([np.nan if r.get(m) is None else float(r[m]) for r in rows]) for m in methods}

    draws = [np.concatenate([rng.choice(ix, len(ix)) for ix in strata.values()]) for _ in range(DRAWS)]
    full = np.arange(len(rows))
    result = {'questions': len(rows), 'benchmarks': {b: len(ix) for b, ix in strata.items()}, 'methods': {}}
    for method in methods:
        entry = {'accuracy': finite_mean(table[method], full),
                 'interval': np.nanquantile([finite_mean(table[method], d) for d in draws], [.025, .975]).tolist()}
        for base in ('majority', 'random'):
            if method == base:
                continue
            difference = table[method]-table[base]
            entry[f'minus_{base}'] = finite_mean(difference, full)
            entry[f'minus_{base}_interval'] = np.nanquantile([finite_mean(difference, d) for d in draws],
                                                             [.025, .975]).tolist()
        result['methods'][method] = entry
    return result


def attempt_selection(usage: dict, arrays: dict, selection: dict, model: str) -> dict:
    """ATTEMPT_SELECTION.json body for one model."""
    rows, selectors = picks(usage, arrays, selection)
    methods = ['majority', 'random', 'shortest', 'longest', *selectors]
    rng = np.random.default_rng(int(digest([SEED, model, 'attempts'])[:8], 16))
    return {'model': model, 'usage_sha256': usage['sha256'], 'selection_sha256': selection['sha256'],
            'draws': DRAWS, 'seed': SEED, 'all_questions': summarize(rows, methods, rng),
            'mixed_questions': summarize([r for r in rows if r['mixed']], methods, rng), 'per_question': rows}

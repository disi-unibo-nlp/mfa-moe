"""Pick the experts to nudge by outcome, not by reasoning-step class.

The question asked of the data is simple: when a model answers the same question several
times, which experts does it use more in its wrong attempts than in its right ones?
Cohort B of the saved seven-model captures holds 6-8 attempts per AIME/AMC question with their per-token
routing.

Two stages:

- ``usage(model)`` reads every cohort-B routing capture (torch; run it as a CPU job) and
  stores, for each attempt, how often each (layer, expert) is selected per reasoning token.
- ``selection(...)`` turns those rates into one small expert set per model (pure numpy):
  1. within each question, z-score every expert's usage across that question's attempts;
  2. score = mean over mixed-outcome questions of (mean z of wrong) - (mean z of right);
  3. keep pairs whose sign survives leaving out any single question;
  4. never touch the shared core (most used experts) or the most execution-specific experts;
  5. take the best pairs until they carry MASS_TARGET of native routing (MIN..MAX pairs);
  6. draw a matched random control set (same layers and native-frequency quintiles).
  An evaluation family whose question is in cohort B gets a set selected without it.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gzip
import math
from pathlib import Path

import numpy as np

from store import FAST, digest, file_sha, load, publish, require

# The saved routing captures of the seven-model study, and the re-scored outcomes of those attempts.
CAPTURES = Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3')
OUTCOMES = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/'
                'campaign-v3/outcomes/outcomes_v3-99bf8d59506e2893.parquet')
BULK = FAST/'outcome-steering'

COMMITTEE_SHARE = .05            # top share of experts per layer by native use = shared core
EXECUTION_SHARE = .10            # top share of positive Implement/Analyze candidates per fold
EXECUTION_CLASSES = ('Implement', 'Analyze')
MASS_TARGET = .012               # share of native routing slots carried by the nudged experts
MIN_PAIRS, MAX_PAIRS = 5, 40


# ---------------------------------------------------------------------------------------------
# Stage 1: per-attempt usage rates (needs torch, so it runs in a CPU allocation)
# ---------------------------------------------------------------------------------------------

def nominations_path(reports: Path, model: str) -> Path:
    """The class nominations of the base release r7; `reports` is the report/ folder."""
    return reports/'whole-reasoning-20261021-r7'/model/'WHOLE_NOMINATIONS.json'


def class_nominations(reports: Path, model: str) -> dict:
    """The class nominations of the base release r7, read without checking their receipt hash."""
    return load(nominations_path(reports, model))


def cohort_attempts(model: str) -> list[dict]:
    """Cohort-B attempts of one model, with their capture file and its recorded hash."""
    root = CAPTURES/'manifests'/model/'B'
    ready = load(root/'ready.json')
    rows = []
    for shard in ready['shards']:
        manifest = load(root/f"{shard['index']}.json")
        capture = Path(manifest['capture_dir'])
        for attempt in manifest['attempts']:
            identity = attempt['identity']
            tensor = capture/f"{attempt['id']}.pt.gz"
            marker = load(capture/f"{attempt['id']}.complete.json")
            sha = next(f['sha256'] for f in marker['files'] if Path(f['path']) == tensor)
            rows.append({'attempt': attempt['id'], 'dataset': identity['dataset'],
                         'problem_id': identity['problem_id'], 'sample_id': int(identity['sample_id']),
                         'question': f"{identity['dataset']}|{identity['source_problem_id']}",
                         'trace_sha256': attempt['trace_sha256'], 'tensor': str(tensor), 'tensor_sha256': sha})
    require(rows, f'no cohort-B attempts for {model}')
    return sorted(rows, key=lambda r: r['attempt'])


def strict_outcomes(model: str, attempts: list[dict]) -> list[bool | None]:
    """Re-scored strict correctness per attempt; capped or unscored attempts stay None."""
    import pandas as pd
    table = pd.read_parquet(OUTCOMES, columns=['corpus', 'model', 'dataset', 'problem_id',
                                               'sample_id', 'strict_correct'])
    table = table[(table.corpus == 'card-v3') & (table.model == model)]
    lookup = {(r.dataset, r.problem_id, int(r.sample_id)): r.strict_correct for r in table.itertuples()}
    result = []
    for row in attempts:
        value = lookup.get((row['dataset'], row['problem_id'], row['sample_id']))
        result.append(None if value is None or pd.isna(value) else bool(value))
    return result


def _attempt_rate(task: tuple[str, str, int]) -> np.ndarray:
    """(layers, experts) selection rate over one attempt's reasoning tokens."""
    import torch
    torch.set_num_threads(1)
    path, sha, experts = task
    require(file_sha(path) == sha, f'capture changed: {path}')
    with gzip.open(path, 'rb') as stream:
        saved = torch.load(stream, map_location='cpu', weights_only=True)
    ids = saved['ids'].numpy()                                   # (layers, tokens, k)
    reasoning = saved['reasoning_tokens'].numpy().astype(np.int64)
    require(len(reasoning) > 0, f'no reasoning tokens: {path}')
    picked = ids[:, reasoning, :]
    rate = np.stack([np.bincount(layer.ravel(), minlength=experts) for layer in picked])
    return (rate/len(reasoning)).astype(np.float32)


def usage(model: str, reports: Path, workers: int = 4, out: Path = BULK) -> dict:
    """Write out/MODEL/USAGE.npz (rates) and USAGE.json (attempts, outcomes, hashes) for one model.

    `out` is where selection() and the outcome release (steering.release outcome) read the usage from: BULK."""
    nominations = class_nominations(reports, model)
    attempts = cohort_attempts(model)
    strict = strict_outcomes(model, attempts)
    tasks = [(row['tensor'], row['tensor_sha256'], nominations['experts']) for row in attempts]
    with ProcessPoolExecutor(workers) as pool:
        rates = np.stack(list(pool.map(_attempt_rate, tasks)))
    require(rates.shape[1] == len(nominations['layers']), 'capture layers differ from nominations')
    folder = out/model
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    np.savez(folder/'USAGE.npz', rates=rates)
    return publish(folder/'USAGE.json', {
        'model': model, 'shape': list(rates.shape), 'rates_sha256': file_sha(folder/'USAGE.npz'),
        'code_sha256': file_sha(__file__),
        'outcomes': str(OUTCOMES), 'outcomes_sha256': file_sha(OUTCOMES),
        'attempts': [{**row, 'strict': value} for row, value in zip(attempts, strict)]})


# ---------------------------------------------------------------------------------------------
# Stage 2: scores, protected experts and the selected sets (pure numpy, unit-tested)
# ---------------------------------------------------------------------------------------------

def question_scores(rates: np.ndarray, questions: list[str], strict: list[bool | None],
                    drop: frozenset = frozenset()) -> tuple[np.ndarray, list[str]]:
    """Mean over mixed-outcome questions of z(wrong) - z(right), per (layer, expert).

    Usage is z-scored across the attempts of each question, so a question's overall difficulty
    or length cannot leak into the score. Attempts with unknown correctness are ignored.
    """
    questions, strict = np.asarray(questions), np.asarray(strict, dtype=object)
    differences, used = [], []
    for question in sorted(set(questions.tolist()) - set(drop)):
        rows = np.flatnonzero((questions == question) & (strict != None))  # noqa: E711
        right, wrong = rows[strict[rows] == True], rows[strict[rows] == False]  # noqa: E712
        if not len(right) or not len(wrong):
            continue
        block = rates[rows].astype(np.float64)
        spread = block.std(0, ddof=1)
        z = np.divide(block-block.mean(0), spread, out=np.zeros_like(block), where=spread > 0)
        position = {row: i for i, row in enumerate(rows)}
        differences.append(z[[position[r] for r in wrong]].mean(0)-z[[position[r] for r in right]].mean(0))
        used.append(question)
    require(used, 'no question has both a right and a wrong attempt')
    return np.mean(differences, axis=0), used


def stable_scores(rates, questions, strict, drop=frozenset()):
    """Scores plus a mask of pairs whose sign survives leaving out any one question."""
    score, used = question_scores(rates, questions, strict, drop)
    stable = score != 0
    for question in used:
        refit, _ = question_scores(rates, questions, strict, frozenset(drop) | {question})
        stable &= np.sign(refit) == np.sign(score)
    return score, stable, used


def protected_pairs(nominations: dict) -> tuple[np.ndarray, set, set]:
    """Native frequency plus the shared-core and execution pairs that are never nudged."""
    layers, experts = nominations['layers'], nominations['experts']
    by_fold = {r['fold']: np.asarray(r['frequency'], dtype=float) for r in nominations['records']}
    frequency = np.mean([by_fold[f] for f in sorted(by_fold)], axis=0)
    top = math.ceil(COMMITTEE_SHARE*experts)
    committee = {(layers[i], int(e)) for i in range(len(layers))
                 for e in np.argsort(-frequency[i], kind='stable')[:top]}
    execution = set()
    for record in nominations['records']:
        if record['class'] in EXECUTION_CLASSES:
            positive = record['candidates']['positive']          # already sorted by |RD|
            execution |= {(p['layer'], p['expert']) for p in positive[:math.ceil(EXECUTION_SHARE*len(positive))]}
    return frequency, committee, execution


def choose(score, stable, frequency, layers, k, protected, identity) -> dict:
    """Best stable pairs until MASS_TARGET of routing, each with a feasible matched control."""
    from steering.selection import layer_matched_controls

    def mass(pairs):
        return float(sum(frequency[layers.index(p['layer'])][p['expert']] for p in pairs))/(len(layers)*k)

    def controls(pairs):
        excluded = protected | {(p['layer'], p['expert']) for p in pairs}
        return layer_matched_controls([], pairs, frequency.tolist(), layers, excluded, identity)['deactivate']

    def ranked(mask):
        candidates = [(float(score[i, e]), layers[i], int(e)) for i, e in zip(*np.nonzero(mask))
                      if (layers[i], int(e)) not in protected]
        return [{'layer': l, 'expert': e, 'score': s} for s, l, e in sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))]

    chosen, unstable = [], 0
    for pool, flag in ((ranked(stable & (score > 0)), False), (ranked(~stable & (score > 0)), True)):
        for pair in pool:
            if len(chosen) >= MAX_PAIRS or (len(chosen) >= MIN_PAIRS and (flag or mass(chosen) >= MASS_TARGET)):
                break
            try:
                controls(chosen+[pair])                  # skip pairs whose control pool is exhausted
            except ValueError:
                continue
            chosen.append({**pair, 'unstable_fill': flag})
            unstable += flag
    require(len(chosen) >= MIN_PAIRS, f'fewer than {MIN_PAIRS} nudgeable pairs: {identity}')
    random = controls(chosen)
    return {'targets': chosen, 'controls': random, 'mass': mass(chosen), 'control_mass': mass(random),
            'unstable_fill': unstable, 'stable_positive': int((stable & (score > 0)).sum())}


def selection(model: str, holdouts: list[str], reports: Path) -> dict:
    """All expert sets of one model: 'all' plus one per held-out evaluation question."""
    receipt = load(BULK/model/'USAGE.json')
    require(file_sha(BULK/model/'USAGE.npz') == receipt['rates_sha256'], 'USAGE.npz changed')
    rates = np.load(BULK/model/'USAGE.npz')['rates']
    nominations = class_nominations(reports, model)
    layers, k = nominations['layers'], nominations['records'][0]['native_k']
    questions = [row['question'] for row in receipt['attempts']]
    strict = [row['strict'] for row in receipt['attempts']]
    frequency, committee, execution = protected_pairs(nominations)
    protected = committee | execution
    sets = {}
    for holdout in ['all', *sorted(set(holdouts) & set(questions))]:
        drop = frozenset() if holdout == 'all' else frozenset({holdout})
        score, stable, used = stable_scores(rates, questions, strict, drop)
        sets[holdout] = {**choose(score, stable, frequency, layers, k, protected,
                                  [model, 'outcome-v1', holdout]), 'questions': len(used)}
    return {'model': model, 'selector': 'outcome-within-question-z-v1', 'layers': layers, 'native_k': k,
            'experts': nominations['experts'], 'usage_sha256': receipt['sha256'],
            'nominations_sha256': nominations['sha256'],
            'rules': {'committee_share': COMMITTEE_SHARE, 'execution_share': EXECUTION_SHARE,
                      'execution_classes': list(EXECUTION_CLASSES), 'mass_target': MASS_TARGET,
                      'pairs': [MIN_PAIRS, MAX_PAIRS]},
            'protected': {'committee': len(committee), 'execution': len(execution), 'union': len(protected)},
            'attempts': len(questions), 'known_outcomes': sum(s is not None for s in strict), 'sets': sets,
            'set_id': digest(sets)}


# ---------------------------------------------------------------------------------------------
# r2: class-conditioned selection on the captured attempts
# ---------------------------------------------------------------------------------------------
#
# Every class c gets its own score: the same within-question wrong-minus-right z-difference as r1,
# computed on the usage of the tokens labelled c. Candidates come only from the three routed layers at
# relative depth 1/2, 2/3 and 5/6 (layers 12/16/20 of GPT-OSS, where Beyond the Trace chose its five
# experts). Pairs are added until they carry MASS_SHARE of those layers' native gate weight per
# reasoning token, the paper's definition (its five carry "1.2% of the gated mass across the three
# recorded layers"), at most R2_MAX_PAIRS. Two disjoint matched random sets are drawn per expert set.
# No leave-one-question-out stability filter is applied here: it passed as many pairs under shuffled
# labels as under real ones.

CLASS_NAMES = ('Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor')
BLIND = 'blind'                   # the class-blind set: all reasoning tokens (index ALL_REASONING)
MASS_SHARE = .012                 # share of the window layers' native gate weight (Beyond the Trace, App. A.6)
R2_MIN_PAIRS, R2_MAX_PAIRS = 5, 10
DRAWS = 2                         # matched random sets per expert set
PAPER_FIVE = ((12, 23), (12, 12), (12, 24), (16, 11), (20, 0))    # Beyond the Trace, GPT-OSS-20B


def window_layers(layers: list[int]) -> list[int]:
    """The three routed layers at relative depth 1/2, 2/3 and 5/6 (12, 16, 20 for 24 layers)."""
    n = len(layers)
    return [layers[n//2], layers[2*n//3], layers[5*n//6]]


def question_differences(rates: np.ndarray, questions: list[str], strict: list[bool | None],
                         valid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Per mixed-outcome question, z(wrong) - z(right) for every (layer, expert).

    Args:
        rates: (attempts, layers, experts) usage per token.
        questions, strict: Question and known correctness (None = unknown) of every attempt.
        valid: Attempts that may enter (e.g. the class occurs in them).

    Returns:
        (questions used, layers, experts) differences and the questions used, sorted.
    """
    questions, strict = np.asarray(questions), np.asarray(strict, dtype=object)
    known = valid & (strict != None)  # noqa: E711
    differences, used = [], []
    for question in sorted(set(questions[known].tolist())):
        rows = np.flatnonzero((questions == question) & known)
        right, wrong = strict[rows] == True, strict[rows] == False  # noqa: E712
        if not right.any() or not wrong.any():
            continue
        block = rates[rows].astype(np.float64)
        spread = block.std(0, ddof=1)
        z = np.divide(block-block.mean(0), spread, out=np.zeros_like(block), where=spread > 0)
        differences.append(z[wrong].mean(0)-z[right].mean(0))
        used.append(question)
    shape = (0,)+rates.shape[1:]
    return (np.stack(differences) if differences else np.zeros(shape)), used


def holdout_scores(differences: np.ndarray, used: list[str], holdouts: list[str]) -> dict[str, np.ndarray]:
    """'all' plus, for every held-out question, the mean difference without it (exact, no refit)."""
    total, n = differences.sum(0), len(used)
    require(n >= 2, 'fewer than two mixed-outcome questions')
    position = {q: i for i, q in enumerate(used)}
    scores = {'all': total/n}
    for question in holdouts:
        scores[question] = (total-differences[position[question]])/(n-1) if question in position else total/n
    return scores


class WindowSets:
    """What choosing a window set shares: the gate-weight share of a pair list and its matched random draws.

    Args:
        frequency: (layers, experts) native top-k selections per reasoning token; controls are matched on it.
        layers, window, k: Routed layers, the three candidate layers and native top-k.
        protected: (layer, expert) pairs never nudged and never drawn as controls.
        identity: Deterministic identity of the set (model, set name, holdout).
        gate: (layers, experts) native gate weight per reasoning token; the set size is its window share
            (defaults to `frequency`, i.e. the slot share, for callers without weights).
    """
    def __init__(self, frequency: np.ndarray, layers: list[int], window: list[int], k: int, protected: set,
                 identity: list, gate: np.ndarray | None = None):
        self.frequency, self.layers, self.window, self.k = frequency, layers, window, k
        self.protected, self.identity = protected, identity
        self.gate = frequency if gate is None else gate
        self.window_gate = float(sum(self.gate[layers.index(layer)].sum() for layer in window))
        self.window_frequency = [frequency[layers.index(layer)].tolist() for layer in window]   # rows in window order
        self.matching = []                            # draws that fell back to nearest-usage matching

    def mass(self, pairs: list[dict]) -> float:
        """Share of the window layers' gate weight that the pairs carry."""
        return float(sum(self.gate[self.layers.index(p['layer'])][p['expert']] for p in pairs))/self.window_gate

    def slots(self, pairs: list[dict]) -> float:
        """Share of the window layers' top-k slots that the pairs fill."""
        return float(sum(self.frequency[self.layers.index(p['layer'])][p['expert']] for p in pairs)
                     )/(len(self.window)*self.k)

    def nearest(self, pairs: list[dict], taken: set, draw: int) -> list[dict]:
        """Controls from the same layer with the closest native usage (ties by a fixed digest).

        Used only when a fixed set leaves too few experts in its layer/quintile pool (GPT-OSS has 32 experts
        per layer and two disjoint draws).
        """
        chosen = []
        for p in pairs:
            row = self.window_frequency[self.window.index(p['layer'])]
            pool = [e for e in range(len(row)) if (p['layer'], e) not in taken]
            require(pool, f'no control expert left in layer {p["layer"]}: {self.identity}')
            expert = min(pool, key=lambda e: (abs(row[e]-row[p['expert']]), digest([self.identity, draw, p['layer'], e])))
            taken = taken | {(p['layer'], expert)}
            chosen.append({'layer': int(p['layer']), 'expert': int(expert)})
        return chosen

    def draws(self, pairs: list[dict], exact: bool = True) -> list[list[dict]]:
        """DRAWS disjoint matched random sets for the pairs; with `exact` False, nearest-usage matching may fill in."""
        from steering.selection import layer_matched_controls
        result, taken = [], set(self.protected) | {(p['layer'], p['expert']) for p in pairs}
        for draw in range(DRAWS):
            try:
                controls = layer_matched_controls([], pairs, self.window_frequency, self.window, taken,
                                                  self.identity+['draw', draw])['deactivate']
            except ValueError:
                if exact:
                    raise
                controls = self.nearest(pairs, taken, draw)
                self.matching.append(draw)
            result.append(controls)
            taken |= {(p['layer'], p['expert']) for p in controls}
        return result


def ranked_window_pairs(score: np.ndarray, layers: list[int], window: list[int], protected: set,
                        sets: WindowSets) -> list[dict]:
    """The best positive unprotected window pairs, added until MASS_SHARE or R2_MAX_PAIRS."""
    rows = [layers.index(l) for l in window]
    candidates = sorted(((float(score[i, e]), layers[i], int(e)) for i in rows for e in range(score.shape[1])
                         if score[i, e] > 0 and (layers[i], int(e)) not in protected),
                        key=lambda c: (-c[0], c[1], c[2]))
    chosen = []
    for value, layer, expert in candidates:
        if len(chosen) >= R2_MAX_PAIRS or (len(chosen) >= R2_MIN_PAIRS and sets.mass(chosen) >= MASS_SHARE):
            break
        pair = {'layer': layer, 'expert': expert, 'score': value}
        try:
            sets.draws(chosen+[pair])                 # skip pairs whose control pools would run out
        except ValueError:
            continue
        chosen.append(pair)
    return chosen


def choose_window(score: np.ndarray, frequency: np.ndarray, layers: list[int], window: list[int], k: int,
                  protected: set, identity: list, fixed: list[tuple[int, int]] | None = None,
                  gate: np.ndarray | None = None) -> dict:
    """Best positive window pairs until MASS_SHARE of the window's gate weight, plus DRAWS matched random sets.

    Args:
        score: (layers, experts) score of this set (ignored for a fixed set).
        frequency: (layers, experts) native top-k selections per reasoning token (sums to k per layer);
            controls are matched on it.
        layers, window, k: Routed layers, the three candidate layers and native top-k.
        protected: (layer, expert) pairs never nudged and never drawn as controls.
        identity: Deterministic identity of the set (model, set name, holdout).
        fixed: A given set (Beyond the Trace's five) instead of ranked candidates.
        gate: (layers, experts) native gate weight per reasoning token; the set size is its window share
            (defaults to `frequency`, i.e. the slot share, for callers without weights).
    """
    sets = WindowSets(frequency, layers, window, k, protected, identity, gate)
    if fixed is not None:
        chosen = [{'layer': l, 'expert': e, 'score': float(score[layers.index(l), e]), 'fixed': True} for l, e in fixed]
    else:
        chosen = ranked_window_pairs(score, layers, window, protected, sets)
    require(chosen, f'no nudgeable pair: {identity}')
    controls = sets.draws(chosen, exact=fixed is None)
    return {'targets': chosen, 'controls': controls, 'mass': sets.mass(chosen), 'slot_share': sets.slots(chosen),
            'control_matching': ['nearest_usage' if d in sets.matching else 'layer_quintile' for d in range(DRAWS)],
            'control_mass': [sets.mass(c) for c in controls], 'short': len(chosen) < R2_MIN_PAIRS}


def class_rates(counts: np.ndarray, tokens: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Usage per token of class `index` and the attempts where the class occurs."""
    n = tokens[:, index].astype(np.float64)
    rates = np.divide(counts[:, index], n[:, None, None], out=np.zeros(counts[:, index].shape), where=n[:, None, None] > 0)
    return rates, n > 0


def gate_rate(usage: dict, arrays: dict) -> np.ndarray:
    """(layers, experts) native gate weight per reasoning token, averaged over usable attempts."""
    usable = np.array([r['usable'] for r in usage['attempts']])
    tokens = arrays['tokens'][usable, 7].astype(np.float64)
    weights = arrays['weights'][usable].astype(np.float64)
    return np.divide(weights, tokens[:, None, None], out=np.zeros(weights.shape), where=tokens[:, None, None] > 0).mean(0)


def r2_protected(usage: dict, arrays: dict, nominations: dict) -> tuple[np.ndarray, int, set, set]:
    """Native frequency (selections per reasoning token), native k, committee and execution pairs.

    The committee (top COMMITTEE_SHARE experts per layer by native use) comes from the captures
    themselves; the execution-specific experts from the r7 class nominations.
    """
    layers = usage['layers']
    usable = np.array([r['usable'] for r in usage['attempts']])
    rates, _ = class_rates(arrays['counts'][usable], arrays['tokens'][usable], 7)
    frequency = rates.mean(0)
    k = int(round(float(frequency[0].sum())))
    top = math.ceil(COMMITTEE_SHARE*frequency.shape[1])
    committee = {(layers[i], int(e)) for i in range(len(layers)) for e in np.argsort(-frequency[i], kind='stable')[:top]}
    require(nominations['layers'] == layers, 'nomination layers differ from the capture layers')
    _, _, execution = protected_pairs(nominations)
    return frequency, k, committee, execution


def r2_selection(usage: dict, arrays: dict, nominations: dict, holdouts: list[str], model: str) -> dict:
    """Every r2 expert set of one model: 7 class sets and the class-blind set, each for 'all' and per holdout.

    Args:
        usage: USAGE_CLASS.json (attempt records, layers, experts).
        arrays: USAGE_CLASS.npz arrays (counts, tokens, weights).
        nominations: r7 WHOLE_NOMINATIONS.json of the model (execution-specific experts).
        holdouts: Evaluation questions; each gets sets selected without it.
        model: Study model (GPT-OSS uses Beyond the Trace's five as its class-blind set).
    """
    layers, records = usage['layers'], usage['attempts']
    questions = [r['question'] for r in records]
    strict = [r['strict'] for r in records]
    usable = np.array([r['usable'] for r in records])
    frequency, k, committee, execution = r2_protected(usage, arrays, nominations)
    gate = gate_rate(usage, arrays)
    protected = committee | execution
    window = window_layers(layers)
    sets = {}
    for name, index in [*((c, i) for i, c in enumerate(CLASS_NAMES)), (BLIND, 7)]:
        rates, present = class_rates(arrays['counts'], arrays['tokens'], index)
        differences, used = question_differences(rates, questions, strict, usable & present)
        if len(used) < 2:
            sets[name] = {'status': 'too_few_mixed_questions', 'questions': len(used)}
            continue
        scores = holdout_scores(differences, used, holdouts)
        fixed = list(PAPER_FIVE) if model == 'gpt' and name == BLIND else None
        sets[name] = {'questions': len(used), 'fixed': fixed is not None, 'sets': {
            holdout: choose_window(score, frequency, layers, window, k, protected,
                                   [model, 'outcome-r2', name, holdout], fixed, gate)
            for holdout, score in scores.items()}}
    return {'model': model, 'selector': 'class-outcome-window-v2', 'layers': layers, 'window': window,
            'native_k': k, 'experts': usage['experts'], 'usage_sha256': usage['sha256'],
            'nominations_sha256': nominations['sha256'],
            'rules': {'mass_share': MASS_SHARE, 'mass_definition': 'native gate weight per reasoning token, share of the '
                      'three window layers (Beyond the Trace App. A.6)', 'pairs': [R2_MIN_PAIRS, R2_MAX_PAIRS], 'draws': DRAWS,
                      'committee_share': COMMITTEE_SHARE, 'execution_share': EXECUTION_SHARE,
                      'committee_source': 'capture native frequency', 'execution_source': 'r7 nominations',
                      'paper_five': [list(p) for p in PAPER_FIVE] if model == 'gpt' else None},
            'protected': {'committee': len(committee), 'execution': len(execution), 'union': len(protected)},
            'attempts': len(records), 'usable': int(usable.sum()),
            'known_outcomes': sum(s is not None for s in strict), 'sets': sets, 'set_id': digest(sets)}

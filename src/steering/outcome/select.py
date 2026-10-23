"""Pick the experts to nudge by outcome, not by reasoning-step class.

The question asked of the data is simple: when a model answers the same question several
times, which experts does it use more in its wrong attempts than in its right ones?
Campaign-v3 cohort B holds 6-8 attempts per AIME/AMC question with their per-token routing.

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
import json
import math
from pathlib import Path

import numpy as np

from store import FAST, digest, file_sha, load, publish, require

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

def class_nominations(reports: Path, model: str) -> dict:
    """The class nominations of the base release r7 (the reports folder is report/ of the working repository)."""
    return load(reports/'whole-reasoning-20261021-r7'/model/'WHOLE_NOMINATIONS.json')


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


def usage(model: str, reports: Path, workers: int = 4) -> dict:
    """Write USAGE.npz (rates) and USAGE.json (attempts, outcomes, hashes) for one model."""
    nominations = class_nominations(reports, model)
    attempts = cohort_attempts(model)
    strict = strict_outcomes(model, attempts)
    tasks = [(row['tensor'], row['tensor_sha256'], nominations['experts']) for row in attempts]
    with ProcessPoolExecutor(workers) as pool:
        rates = np.stack(list(pool.map(_attempt_rate, tasks)))
    require(rates.shape[1] == len(nominations['layers']), 'capture layers differ from nominations')
    folder = BULK/model
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

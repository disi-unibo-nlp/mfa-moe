"""Is there a real wrong-versus-right routing signal in each reasoning class? (SIGNAL.json, CPU)

For every class (and the class-blind set) of one model, on the capture release's attempts:

- real: mean score of the top window pairs (the best R2_MAX_PAIRS positive, unprotected pairs in the
  three candidate layers), and the same for all layers (top 1% of pairs);
- null: the same statistics after shuffling right/wrong labels within each question (SHUFFLES times);
  p is the share of shuffles reaching the real value;
- replication: choose the top window pairs on a random half of the mixed questions and score them on
  the other half (SPLITS splits); mean held-out score and share of splits where it is positive;
- length: correlation, over window pairs, between the outcome score and the same within-question
  score computed against reasoning length (long minus short attempts). A high value means the score
  mostly tracks length (as Qwen3-30B's did in r1).

It is reported, never used as a gate (D-0006).
"""
from __future__ import annotations

import numpy as np

from moe_exp.final_routing.common import digest

from .select import (BLIND, CLASS_NAMES, R2_MAX_PAIRS, class_rates, question_differences, window_layers)

SHUFFLES, SPLITS, SEED = 200, 50, 20261023


def question_blocks(rates: np.ndarray, questions: list[str], strict: list, valid: np.ndarray):
    """Per mixed-outcome question: z-scored usage of its known attempts and their labels."""
    questions, strict = np.asarray(questions), np.asarray(strict, dtype=object)
    known = valid & (strict != None)  # noqa: E711
    blocks = []
    for question in sorted(set(questions[known].tolist())):
        rows = np.flatnonzero((questions == question) & known)
        labels = np.array([bool(v) for v in strict[rows]])
        if labels.all() or not labels.any():
            continue
        block = rates[rows].astype(np.float64)
        spread = block.std(0, ddof=1)
        blocks.append((np.divide(block-block.mean(0), spread, out=np.zeros_like(block), where=spread > 0),
                       labels, rows))
    return blocks


def score_from(blocks, labels_of=lambda labels: labels) -> np.ndarray:
    """Mean over questions of z(wrong) - z(right) with the labels given by `labels_of`."""
    total = 0.
    for z, labels, _ in blocks:
        right = labels_of(labels)
        total = total+z[~right].mean(0)-z[right].mean(0)
    return total/len(blocks)


def top_mean(score: np.ndarray, mask: np.ndarray, count: int) -> float:
    """Mean of the `count` largest scores among the allowed pairs."""
    values = np.sort(score[mask])[::-1][:count]
    return float(values.mean()) if len(values) else float('nan')


def length_score(blocks, lengths: np.ndarray) -> np.ndarray:
    """Per pair, mean over questions of the within-question correlation of usage with log length."""
    total = 0.
    for z, _, rows in blocks:
        x = np.log1p(lengths[rows].astype(np.float64))
        x = (x-x.mean())/(x.std(ddof=1) or 1.)
        total = total+np.tensordot(x, z, axes=(0, 0))/(len(rows)-1)
    return total/len(blocks)


def class_signal(rates, valid, questions, strict, lengths, layers, protected, rng) -> dict:
    """SIGNAL entry of one class: real, null, replication and length statistics."""
    blocks = question_blocks(rates, questions, strict, valid)
    if len(blocks) < 4:
        return {'questions': len(blocks), 'status': 'too_few_mixed_questions'}
    window = window_layers(layers)
    allowed = np.ones(rates.shape[1:], bool)
    for layer, expert in protected:
        allowed[layers.index(layer), expert] = False
    in_window = np.zeros_like(allowed)
    in_window[[layers.index(l) for l in window]] = True
    count_all = max(1, int(.01*allowed.sum()))

    def stats(score):
        return top_mean(score, allowed & in_window, R2_MAX_PAIRS), top_mean(score, allowed, count_all)

    real = score_from(blocks)
    real_window, real_all = stats(real)
    null = np.array([stats(score_from(blocks, lambda labels: rng.permutation(labels))) for _ in range(SHUFFLES)])
    held = []
    for _ in range(SPLITS):
        order = rng.permutation(len(blocks))
        first, second = [blocks[i] for i in order[:len(order)//2]], [blocks[i] for i in order[len(order)//2:]]
        pick = score_from(first)
        mask = allowed & in_window
        top = np.argsort(np.where(mask, pick, -np.inf), axis=None)[::-1][:R2_MAX_PAIRS]
        held.append(float(score_from(second).ravel()[top].mean()))
    length = length_score(blocks, lengths)
    pairs = allowed & in_window
    return {'questions': len(blocks), 'real_window': real_window, 'real_all_layers': real_all,
            'null_window_mean': float(null[:, 0].mean()), 'null_window_95': float(np.quantile(null[:, 0], .95)),
            'p_window': float((null[:, 0] >= real_window).mean()),
            'null_all_mean': float(null[:, 1].mean()), 'null_all_95': float(np.quantile(null[:, 1], .95)),
            'p_all_layers': float((null[:, 1] >= real_all).mean()),
            'heldout_window_mean': float(np.mean(held)), 'heldout_window_positive': float(np.mean(np.array(held) > 0)),
            'length_correlation_window': float(np.corrcoef(real[pairs], length[pairs])[0, 1]),
            'length_correlation_all': float(np.corrcoef(real[allowed], length[allowed])[0, 1])}


def signal(usage: dict, arrays: dict, protected: set, model: str) -> dict:
    """SIGNAL.json body for one model: one entry per class and for the class-blind set."""
    layers, records = usage['layers'], usage['attempts']
    questions = [r['question'] for r in records]
    strict = [r['strict'] for r in records]
    usable = np.array([r['usable'] for r in records])
    lengths = np.array([r['reasoning_tokens'] or 0 for r in records])
    rng = np.random.default_rng(int(digest([SEED, model])[:8], 16))
    result = {}
    for name, index in [*((c, i) for i, c in enumerate(CLASS_NAMES)), (BLIND, 7)]:
        rates, present = class_rates(arrays['counts'], arrays['tokens'], index)
        result[name] = class_signal(rates, usable & present, questions, strict, lengths, layers, protected, rng)
    return {'model': model, 'usage_sha256': usage['sha256'], 'shuffles': SHUFFLES, 'splits': SPLITS,
            'seed': SEED, 'classes': result}

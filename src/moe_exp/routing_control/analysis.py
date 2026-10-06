"""Dense semantic-window checks and paired family-clustered ITT summaries."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

CLASSES = ('Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor')


def trajectory_completion(rows, *, triggering_sentence, ordered_transitions, horizon=1024):
    """Evaluate frozen arm-blind behavioral ratings; class names alone never prove a check.

    `substantive` is an independent reader's frozen-rubric judgment, not a lexical cue.
    A transition succeeds only after the trigger, within the horizon and in the frozen order.
    """
    from .design import TRANSITIONS
    if not 1 <= len(ordered_transitions) <= 2 or any(t not in TRANSITIONS for t in ordered_transitions):
        raise ValueError('one or two frozen supported transitions required')
    window = dense_window(rows, triggering_sentence=triggering_sentence, horizon=horizon)
    cursor, completed = 0, []
    for row in window:
        event = row.get('behavioral_transition')
        if event is not None and event not in TRANSITIONS:
            raise ValueError('unregistered behavioral event')
        if event == ordered_transitions[cursor] and row.get('substantive') is True:
            completed.append({'transition': event, 'sentence_index': row['sentence_index'],
                              'token_start': row['token_start']})
            cursor += 1
            if cursor == len(ordered_transitions):
                break
    return {'success': int(cursor == len(ordered_transitions)), 'completed': completed,
            'trigger_excluded': True, 'horizon': horizon}


def dense_window(rows, *, triggering_sentence: int, horizon=1024):
    """Require all contiguous sentence indices and agreeing branch-relative token offsets.

    The triggering sentence is excluded. A sentence counts only if it is fully
    contained within the frozen token horizon.
    """
    if type(triggering_sentence) is not int or triggering_sentence < 0 or type(horizon) is not int or horizon < 1:
        raise ValueError('invalid trigger index or token horizon')
    ordered = sorted(rows, key=lambda r: r['sentence_index'])
    indices = [r['sentence_index'] for r in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError('duplicate sentence labels')
    for before, after in zip(ordered, ordered[1:]):
        if after['sentence_index'] != before['sentence_index'] + 1 or after['segment'] != before['segment']:
            raise ValueError('sparse gaps cannot be joined in a new dense window')
        if after['token_start'] < before['token_end']:
            raise ValueError('label token offsets overlap or disagree')
    if any(r['label'] not in CLASSES or not 0 <= r['token_start'] < r['token_end'] for r in ordered):
        raise ValueError('invalid dense label or token offsets')
    following = [r for r in ordered if r['sentence_index'] > triggering_sentence]
    if following and following[0]['sentence_index'] != triggering_sentence + 1:
        raise ValueError('missing first post-trigger sentence in dense window')
    return [r for r in following if r['token_end'] <= horizon]


def class_summary(rows):
    """Seven-class transitions, dwell, re-entry and loops without bridging gaps."""
    transitions, dwell, reentries, loops = Counter(), defaultdict(list), Counter(), Counter()
    sequence, last, seen, run = [], None, set(), 0
    for row in sorted(rows, key=lambda r: r['sentence_index']):
        if row['label'] not in CLASSES:
            raise ValueError('unknown class')
        contiguous = last is not None and row['segment'] == last['segment'] and row['sentence_index'] == last['sentence_index'] + 1
        if not contiguous:
            if last is not None:
                dwell[last['label']].append(run)
            sequence, seen, run = [], set(), 0
        label = row['label']
        if contiguous:
            transitions[last['label'], label] += 1
        if not contiguous or label != last['label']:
            if contiguous:
                dwell[last['label']].append(run)
            if label in seen:
                reentries[label] += 1
            seen.add(label)
            run = 0
        run += 1
        sequence.append(label)
        if len(sequence) >= 3 and sequence[-3] == label != sequence[-2]:
            loops[label, sequence[-2], label] += 1
        last = row
    if last is not None:
        dwell[last['label']].append(run)
    counts = np.array([[transitions[a, b] for b in CLASSES] for a in CLASSES])
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts, float), where=totals > 0)
    return {'classes': list(CLASSES), 'transition_counts': counts.tolist(),
            'transition_probabilities': probabilities.tolist(),
            'dwell_sentences_observed_including_censoring': {c: dwell[c] for c in CLASSES},
            'reentries': {c: reentries[c] for c in CLASSES},
            'loop_counts': {'|'.join(k): n for k, n in loops.items()},
            'note': 'observed-window dwell includes boundaries; no uncensored-duration claim'}


def routing_kinematics(gates, *, window: int):
    """Fixed contiguous-window TV velocities and signed differences of consecutive speeds."""
    a = np.asarray(gates, float)
    if a.ndim != 3 or type(window) is not int or window < 1 or not np.isfinite(a).all():
        raise ValueError('expected contiguous [tokens,layers,experts] gate distributions')
    if (a < 0).any() or not np.allclose(a.sum(axis=2), 1., atol=1e-6):
        raise ValueError('gate distributions must be normalized')
    blocks = [a[start:start + window].mean(axis=0) for start in range(0, len(a) - window + 1, window)]
    if len(blocks) < 2:
        return {'window': window, 'velocity': [], 'acceleration': []}
    velocity = np.abs(np.diff(np.asarray(blocks), axis=0)).sum(axis=2).mean(axis=1) / (2 * window)
    acceleration = np.diff(velocity) / window
    return {'window': window, 'velocity': velocity.tolist(), 'acceleration': acceleration.tolist(),
            'units': 'TV per emitted token; change in speed per emitted token'}


def paired_itt(assigned, observed, *, reference_arm='native', n_boot=5000, seed=20261001,
               alpha=.05, comparison_pairs=None,
               metrics=('semantic_success', 'correct', 'tokens')):
    """All assignments must have outcome receipts, including failures and nonfires.

    Average seeds within question then questions equally. Shared families are resampled jointly
    across all contrasts. Bonferroni percentile intervals approximate simultaneous coverage.
    The intervals are bootstrap approximations, not exact randomization inference.
    """
    if n_boot < 100 or not 0 < alpha < 1:
        raise ValueError('invalid interval configuration')
    allowed_metrics = ('semantic_success', 'correct', 'tokens')
    if (not metrics or len(set(metrics)) != len(metrics) or
            any(metric not in allowed_metrics for metric in metrics)):
        raise ValueError('invalid ITT metric selection')
    if len({r['uid'] for r in assigned}) != len(assigned):
        raise ValueError('duplicate assigned UID')
    by_uid = {r['uid']: r for r in observed}
    if len(by_uid) != len(observed) or set(by_uid) != {r['uid'] for r in assigned}:
        raise ValueError('ITT requires an explicit receipt for every assignment; missing cells are incomplete')
    grouped = defaultdict(list)
    family_of = {}
    for request in assigned:
        row = by_uid[request['uid']]
        for field in ('question', 'family', 'arm', 'seed'):
            if row[field] != request[field]:
                raise ValueError('assignment/result identity mismatch')
        if row['status'] not in ('complete', 'nonfire', 'early_finish', 'cap', 'failure', 'unscored'):
            raise ValueError('unknown operational status')
        if row['status'] in ('failure', 'unscored') and row['correct'] != 0:
            raise ValueError('operational missing/unscored failures must count as wrong')
        if row['semantic_success'] not in (0, 1) or row['correct'] not in (0, 1):
            raise ValueError('binary endpoints required')
        if type(row['tokens']) is not int or row['tokens'] < 0:
            raise ValueError('record actual emitted/injection tokens even for failures')
        q = request['question']
        if q in family_of and family_of[q] != request['family']:
            raise ValueError('question mapped to inconsistent families')
        family_of[q] = request['family']
        grouped[q, request['arm']].append(row)
    arms = sorted({r['arm'] for r in assigned})
    if reference_arm not in arms or len(arms) < 2:
        raise ValueError('paired design needs a reference and intervention arm')
    questions = sorted(family_of)
    paired = []
    comparisons = comparison_pairs or [(arm, reference_arm) for arm in arms if arm != reference_arm]
    if len(set(comparisons)) != len(comparisons) or any(a == b or a not in arms or b not in arms
                                                     for a, b in comparisons):
        raise ValueError('invalid or duplicate frozen comparison pairs')
    for arm, reference in comparisons:
        for metric in metrics:
            diffs = []
            for q in questions:
                left, right = grouped[q, arm], grouped[q, reference]
                if not left or not right or sorted(r['seed'] for r in left) != sorted(r['seed'] for r in right):
                    raise ValueError('incomplete paired arms or mismatched seeds')
                if len({r['seed'] for r in left}) != len(left):
                    raise ValueError('duplicate question/arm/seed')
                diffs.append(np.mean([r[metric] for r in left]) - np.mean([r[metric] for r in right]))
            paired.append((arm, reference, metric, np.asarray(diffs)))
    families = sorted(set(family_of.values()))
    if len(families) < 2:
        raise ValueError('at least two independent families are needed for uncertainty')
    index = np.array([families.index(family_of[q]) for q in questions])
    cluster_n = np.bincount(index, minlength=len(families))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=n_boot)
    denominator = weights @ cluster_n
    result = []
    multiplicity = len(paired)
    for arm, reference, metric, diff in paired:
        sums = np.bincount(index, weights=diff, minlength=len(families))
        boot = (weights @ sums) / denominator
        lo, hi = np.quantile(boot, [alpha / (2 * multiplicity), 1 - alpha / (2 * multiplicity)])
        result.append({'arm': arm, 'reference': reference, 'metric': metric,
                       'estimate': float(diff.mean()), 'interval': [float(lo), float(hi)],
                       'precision_status': 'DEGENERATE_BOOTSTRAP_NO_RETENTION_CLAIM' if np.std(boot) == 0 else 'ESTIMATED',
                       'n_questions': len(questions), 'n_families': len(families),
                       'multiplicity_family_size': multiplicity})
    return {'contrasts': result, 'resampling': 'paired family clusters, question-weighted ratio',
            'simultaneous_method': 'Bonferroni percentile bootstrap (approximate)',
            'replicates': n_boot, 'seed': seed,
            'statuses': dict(Counter(r['status'] for r in observed)),
            'retention': 'not established; no noninferiority margin chosen'}

"""Prospective numerical comparison of two APIs at identical native prefixes."""
from __future__ import annotations

import math


MODES = ('designated_a', 'full_vocab_a', 'designated_b', 'full_vocab_b')


def percentile(values, quantile):
    values = sorted(values)
    index = (len(values) - 1) * quantile
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (index - low) * (values[high] - values[low])


def calibration(pairs, ratio=1.25, epsilon=1e-6):
    """Compare cross-API errors against independent repeats of both APIs.

    Mean and p99 mirror the pre-existing Q3 relative numerical parity rule.
    A pass qualifies score extraction on these fixtures, never engine equality.
    """
    if not pairs or ratio < 1 or epsilon < 0:
        raise ValueError('nonempty score fixtures and valid frozen tolerances required')
    columns = {mode: [float(row[mode]) for row in pairs] for mode in MODES}
    if any(not math.isfinite(x) or x > 1e-6 for col in columns.values() for x in col):
        raise ValueError('finite raw log probabilities required')
    errors = {}
    for name, left, right in (
        ('designated_repeat', 'designated_a', 'designated_b'),
        ('full_vocab_repeat', 'full_vocab_a', 'full_vocab_b'),
        ('cross_api_a', 'designated_a', 'full_vocab_a'),
        ('cross_api_b', 'designated_b', 'full_vocab_b'),
    ):
        values = [abs(a - b) for a, b in zip(columns[left], columns[right])]
        errors[name] = {'mean_abs': sum(values) / len(values),
            'p99_abs': percentile(values, .99), 'max_abs': max(values), 'by_case': values}
    checks = {}
    for metric in ('mean_abs', 'p99_abs'):
        baseline = max(errors[k][metric] for k in ('designated_repeat', 'full_vocab_repeat'))
        actual = max(errors[k][metric] for k in ('cross_api_a', 'cross_api_b'))
        limit = ratio * baseline + epsilon
        checks[metric] = {'test': actual, 'repeat_baseline': baseline,
            'limit': limit, 'pass': actual <= limit}
    return {'pass': all(row['pass'] for row in checks.values()), 'cases': len(pairs),
        'ratio': ratio, 'epsilon': epsilon, 'checks': checks, 'errors': errors,
        'interpretation': 'score-API numerical calibration, conditional on these prefixes and execution layout; no equivalence or semantic claim'}

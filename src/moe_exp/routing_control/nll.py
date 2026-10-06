"""Native teacher-forced surprisal extraction and Q3 repeat-baseline validation.

These CPU helpers do not import vLLM and cannot launch inference.
"""
from __future__ import annotations

import math
import numpy as np


def actual_logprobs(prompt, prompt_logprobs) -> np.ndarray:
    if not prompt or prompt_logprobs is None or len(prompt_logprobs) != len(prompt):
        raise ValueError('teacher-forced logprobs must cover the identical prompt positions')
    values = []
    for position in range(1, len(prompt)):
        entries = prompt_logprobs[position]
        if not entries or prompt[position] not in entries:
            raise ValueError(f'missing actual-token logprob at position {position}')
        entry = entries[prompt[position]]
        value = float(entry.logprob if hasattr(entry, 'logprob') else entry)
        if not math.isfinite(value) or value > 1e-6:
            raise ValueError('invalid native log probability')
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def pulse_surprisal(prompt, prompt_logprobs, *, continuation_start: int, n_pulse_tokens: int) -> dict:
    if not 1 <= continuation_start <= len(prompt) or not 1 <= n_pulse_tokens <= 256:
        raise ValueError('invalid pulse-window bounds')
    if continuation_start + n_pulse_tokens != len(prompt):
        raise ValueError('measurement prompt must end at its own pulse-window suffix')
    lp = actual_logprobs(prompt, prompt_logprobs)
    window = lp[continuation_start - 1:continuation_start - 1 + n_pulse_tokens]
    if len(window) != n_pulse_tokens:
        raise ValueError('pulse logprob alignment mismatch')
    return {'n_tokens': n_pulse_tokens, 'mean_nll': float(-window.mean()),
            'sum_nll': float(-window.sum()), 'native_logprobs': window.tolist()}


def validate_parity(measured, reference, repeat_a, repeat_b) -> dict:
    """Validate every identical fixture using mean/p99 Q3 logprob tolerances.

    No assertion about routing equivalence follows from this measurement-only validation.
    Routing metrics and the historical Q3 adjudication retain their separate status.
    """
    if not (len(measured) == len(reference) == len(repeat_a) == len(repeat_b)) or not measured:
        raise ValueError('parity fixture counts differ or are empty')
    deltas, baseline = [], []
    for arrays in zip(measured, reference, repeat_a, repeat_b):
        arrays = [np.asarray(a, np.float64) for a in arrays]
        if any(a.ndim != 1 or not np.isfinite(a).all() for a in arrays):
            raise ValueError('parity arrays must be finite one-dimensional logprobs')
        if len({a.shape for a in arrays}) != 1 or arrays[0].size == 0:
            raise ValueError('identical parity positions are required')
        deltas.append(np.abs(arrays[0] - arrays[1]))
        baseline.append(np.abs(arrays[2] - arrays[3]))
    d, b = np.concatenate(deltas), np.concatenate(baseline)
    rows = {}
    for name, f in [('mean_abs', np.mean), ('p99_abs', lambda a: np.percentile(a, 99))]:
        observed, repeat = float(f(d)), float(f(b))
        rows[name] = {'test': observed, 'baseline': repeat, 'limit': repeat * 1.25 + 1e-6,
                      'pass': observed <= repeat * 1.25 + 1e-6}
    exact_baseline = bool(np.array_equal(b, np.zeros_like(b)))
    passed = bool(np.array_equal(d, np.zeros_like(d))) if exact_baseline else all(r['pass'] for r in rows.values())
    return {'pass': passed, 'metrics': rows, 'n_positions': int(d.size),
            'rule': 'Q3 native logprob repeat rule (mean and p99), no fallback',
            'baseline_logprobs_bitwise': exact_baseline}

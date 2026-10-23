"""Declared rule for extending the nudge experiment to more models, applied automatically.

The rule was fixed before any nudge outcome existed (ledger D-0003, 2026-10-22): extend when, for at
least one first-wave model and one delta, either

- the nudge is shorter than matched random experts (target-random token interval above zero) and
  its accuracy against unedited routing is not lower than -1 point (point estimate), or
- the nudge is more accurate than matched random experts (target-random accuracy interval above zero).

Intervals are the bounded simultaneous ones (bound_simultaneous when unknown rows exist). The rule
reads only the sealed EFFECTS_OUTCOME.json; DECISION.json records the cells that met it.
"""
from __future__ import annotations

from pathlib import Path

from store import publish, verified


def interval(summary: dict) -> list[float]:
    return summary.get('bound_simultaneous') or summary['simultaneous']


def qualifying(effects: dict) -> list[dict]:
    """(model, delta) cells that meet the declared extension rule."""
    primary, native = effects['target_random'], effects['target_native']
    cells = []
    for name, tokens in primary.items():
        if not name.endswith(':target_random:tokens') or tokens.get('status') not in ('qualified', 'qualified_bounds'):
            continue
        model, delta = name.split(':')[:2]
        accuracy = primary.get(f'{model}:{delta}:target_random:accuracy', {})
        versus_native = native.get(f'{model}:{delta}:target_native:accuracy', {})
        shorter = interval(tokens)[0] > 0 and versus_native.get('estimate', -1.) >= -.01
        more_accurate = 'estimate' in accuracy and interval(accuracy)[0] > 0
        if shorter or more_accurate:
            cells.append({'model': model, 'delta': delta, 'shorter_without_loss': shorter,
                          'more_accurate': more_accurate})
    return cells


def decide(effects_path: str) -> dict:
    """Write DECISION.json next to the receipt; `extend` says whether the next release should be launched."""
    effects = verified(effects_path)
    cells = qualifying(effects)
    return publish(Path(effects_path).parent/'DECISION.json', {
        'rule': __doc__.strip().splitlines()[0], 'effects_sha256': effects['sha256'],
        'qualifying_cells': cells, 'extend': bool(cells)})

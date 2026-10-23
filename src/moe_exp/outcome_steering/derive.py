"""The CPU chain that turns one model's captures into selections and evidence (run after extraction).

    python -m moe_exp.outcome_steering derive --release report/<capture release> --models gpt

1. classes.usage: lexical labels and per-class expert usage (USAGE_CLASS.npz/.json, FAST);
2. select.r2_selection: the 7 class sets and the class-blind set, each for 'all' and with every roster
   question held out (SELECTION_R2.json);
3. evidence.signal: the per-class signal table (SIGNAL.json);
4. attempts.attempt_selection: routing-based choice among finished attempts (ATTEMPT_SELECTION.json).

Receipts 2-4 go to report/<capture release>/<model>/; a step whose receipt exists is not redone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from moe_exp.final_routing.common import file_sha, publish, require, verified

from . import REPORTS


def derive(release: Path, model: str, workers: int = 8) -> dict:
    """Run (or finish) the chain for one model and return the receipt hashes."""
    from .attempts import attempt_selection
    from .classes import derived_folder, usage
    from .evidence import signal
    from .select import r2_protected, r2_selection
    folder = derived_folder(release, model)
    receipt = folder/'USAGE_CLASS.json'
    if not receipt.exists():
        usage(release, model, workers)
    value = verified(receipt)
    require(file_sha(folder/'USAGE_CLASS.npz') == value['arrays_sha256'], 'USAGE_CLASS.npz changed')
    with np.load(folder/'USAGE_CLASS.npz') as saved:
        arrays = {key: saved[key] for key in ('counts', 'tokens', 'weights')}
    nominations = verified(REPORTS/f'whole-reasoning-20261021-r7/{model}/WHOLE_NOMINATIONS.json')
    questions = verified(release/'CAPTURE_NATIVE.json')['models'][model]['questions']
    out = release/model
    if not (out/'SELECTION_R2.json').exists():
        publish(out/'SELECTION_R2.json', r2_selection(value, arrays, nominations, questions, model))
    selection = verified(out/'SELECTION_R2.json')
    if not (out/'SIGNAL.json').exists():
        _, _, committee, execution = r2_protected(value, arrays, nominations)
        publish(out/'SIGNAL.json', signal(value, arrays, committee | execution, model))
    if not (out/'ATTEMPT_SELECTION.json').exists():
        publish(out/'ATTEMPT_SELECTION.json', attempt_selection(value, arrays, selection, model))
    return {name: verified(out/name)['sha256'] for name in ('SELECTION_R2.json', 'SIGNAL.json',
                                                             'ATTEMPT_SELECTION.json')}

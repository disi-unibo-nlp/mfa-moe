"""The frozen pick rule (D-0006): which half-A cell is confirmed on half B, applied automatically.

Among half-A cells (expert set, delta) whose nudge keeps accuracy against unedited routing no worse
than -5 points (point estimate), take the one with the largest token benefit against the mean of its
two random draws (point estimate; receipts store tokens as log(1+T_control) - log(1+T_treated), so
larger = shorter). If no cell keeps accuracy, take the cell with the best accuracy against random.
Half B confirms that cell and the class-blind set at its own best delta by the same rule.

    python -m moe_exp.outcome_steering pick --effects <half-A>/analysis/EFFECTS_OUTCOME.json --name <B> ...
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

from moe_exp.final_routing.common import publish, require, verified

from .select import BLIND

NONINFERIORITY = -.05
MIN_COMPLETED = .8          # share of half-A nudge rows that must have finished before a pick is made


def cells(effects: dict, model: str) -> list[dict]:
    """Every half-A cell of a model with its three point estimates."""
    result = {}
    for kind, metric in (('target_random', 'tokens'), ('target_random', 'accuracy'), ('target_native', 'accuracy')):
        for name, summary in effects.get(kind, {}).items():
            parts = name.split(':')
            if parts[0] != model or parts[-1] != metric or 'estimate' not in summary:
                continue
            key = (parts[1], parts[2])
            result.setdefault(key, {'set': parts[1], 'delta': float(parts[2].removeprefix('delta-'))})
            result[key][f'{kind}_{metric}'] = summary['estimate']
    complete = ('target_random_tokens', 'target_random_accuracy', 'target_native_accuracy')
    return [c for c in result.values() if all(k in c for k in complete)]


def best(candidates: list[dict]) -> dict | None:
    """The rule on a list of cells; None if the list is empty."""
    if not candidates:
        return None
    keeping = [c for c in candidates if c['target_native_accuracy'] >= NONINFERIORITY]
    if keeping:
        return max(keeping, key=lambda c: (c['target_random_tokens'], c['target_random_accuracy']))
    return max(candidates, key=lambda c: (c['target_random_accuracy'], c['target_random_tokens']))


def pick(effects_path: str, model: str) -> dict:
    """PICK.json body: the picked cell, the class-blind cell, and every candidate."""
    effects = verified(effects_path)
    levels = [r for r in effects['models'][model]['levels'] if r['arm'] != 'budget']
    completed = sum(r['completed'] for r in levels)/max(1, sum(r['rows'] for r in levels))
    require(completed >= MIN_COMPLETED, f'only {completed:.0%} of half-A nudge rows finished; no pick')
    candidates = cells(effects, model)
    picked = best(candidates)
    blind = best([c for c in candidates if c['set'] == BLIND])
    confirm = [c for c in (picked, blind) if c is not None]
    confirm = [c for i, c in enumerate(confirm) if c not in confirm[:i]]
    return {'rule': __doc__.strip().splitlines()[0], 'effects_sha256': effects['sha256'], 'model': model,
            'noninferiority': NONINFERIORITY, 'picked': picked, 'class_blind': blind,
            'confirm': [{'set': c['set'], 'delta': c['delta']} for c in confirm], 'candidates': candidates}


def pick_and_launch(effects_path: str, model: str, name: str, shards: str, local_tests: str) -> dict:
    """Publish PICK.json next to the half-A receipt and launch half B with the confirmed cells."""
    receipt = publish(Path(effects_path).parent/'PICK.json', pick(effects_path, model))
    if receipt['confirm']:
        code = Path(__file__).resolve().parents[3]
        env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'SBATCH_', 'SRUN_'))}
        env['SWEEP_PICK'] = str(Path(effects_path).parent/'PICK.json')
        subprocess.run([str(code/'scripts/outcome_steering/launch.sh'), name, shards, local_tests, model, 'sweep-B'],
                       env=env, check=True, cwd=code)
    return receipt

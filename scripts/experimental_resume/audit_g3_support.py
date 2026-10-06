"""Amend saved X2 G3 dose support against sealed v0.3 without rewriting prior results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

R = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
S = R / 'steering-v1'
OUT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/G3_DOSE_SUPPORT_AMENDMENT.json')
PROTOCOL_SHA = '4483782220d3e556e6a33c4596679f8292fee3a9beb4e6d6a6309f0e899a8932'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(cell):
    return (('L1', 'BAND', 'ALL').index(cell['scope']),
            ('reweight', 'bias', 'force').index(cell['operator']), cell['magnitude'])


def correct(cell):
    matches = cell['matches']
    if len(matches) != 2:
        raise ValueError('each target cell needs both frozen random sets')
    eligible = all(m.get('D', 0) > 0 and .9 <= m['ratio'] <= 1.1 for m in matches)
    checks = dict(cell['checks'])
    checks['random_dose_support'] = eligible
    if set(checks) != {'positive_dose', 'random_dose_support', 'degeneration_increase_le_3pp',
                       'marker_change_ge_15percent', 'target_random_same_direction_half_effect',
                       'native_NLL_increase_le_point10'}:
        raise ValueError('G3 checks differ from sealed result schema')
    return {'policy': cell['policy'], 'prior_G3_pass': cell['G3_pass'],
            'registered_G3_pass': all(v is True for v in checks.values()),
            'registered_random_dose_support': eligible,
            'random_dose_ratios': [m['ratio'] for m in matches],
            'random_magnitudes': [m['magnitude'] for m in matches],
            'checks': checks}


def build():
    protocol = S / 'PREREG_steering_v1.v0.3.md'
    if sha(protocol) != PROTOCOL_SHA:
        raise ValueError('sealed protocol changed')
    source = S / 'runs/x2-resume-v1/analysis/estimates.json'
    original = S / 'runs/x2-resume-v1/analysis/G3_SELECTION.json'
    value = json.loads(source.read_text())
    selection = json.loads(original.read_text())
    if value['status'] != 'COMPLETE' or value['requests'] != 6630 or len(value['cells']) != 42:
        raise ValueError('complete 42-cell X2 result required')
    if not all(v['pass'] for v in value['native_NLL_validation'].values()):
        raise ValueError('native fixture parity required')
    if selection['source_results_sha256'] != sha(source):
        raise ValueError('prior selection has different results source')
    cells = [correct(cell) for cell in value['cells']]
    indexed = {cell['policy']: cell for cell in cells}
    selected = {}
    for sign in (-1, 1):
        candidates = sorted((c for c in value['cells'] if c['sign'] == sign), key=rank)
        chosen = next((c for c in candidates if indexed[c['policy']]['registered_G3_pass']), None)
        selected[str(sign)] = chosen['policy'] if chosen else None
    previous = value['selected_policy_per_sign']
    body = {'schema': 'G3-dose-support-amendment-v1',
            'protocol_path': str(protocol), 'protocol_file_sha256': sha(protocol),
            'source_path': str(source), 'source_file_sha256': sha(source),
            'prior_selection_path': str(original), 'prior_selection_file_sha256': sha(original),
            'population': '39 eligible of 48 dev-cal questions; two seeds; 42 target cells',
            'rule': 'each random set must have a same-operator dose ratio in [0.90, 1.10]; otherwise that target cell is ineligible',
            'prior_passing_cells': sum(c['prior_G3_pass'] for c in cells),
            'registered_passing_cells': sum(c['registered_G3_pass'] for c in cells),
            'prior_selected': previous, 'registered_selected': selected,
            'selection_changed': previous != selected,
            'cells': cells,
            'interpretation': 'point-estimate dose selection only; off-band E-minus-M comparisons are unmatched diagnostics, not registered controls; no semantic steering claim'}
    body['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return body


def run():
    body = build()
    if OUT.exists() and json.loads(OUT.read_text()) != body:
        raise ValueError('amendment is immutable; a changed source needs a new version')
    if not OUT.exists():
        OUT.write_text(json.dumps(body, indent=1) + '\n')
    print(json.dumps({'path': str(OUT), 'prior_pass': body['prior_passing_cells'],
                      'registered_pass': body['registered_passing_cells'],
                      'selected': body['registered_selected'], 'seal': body['sha256']}))


if __name__ == '__main__':
    run()

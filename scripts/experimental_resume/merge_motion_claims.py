"""Append only source-verified descriptive native-motion claims to the ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
LEDGER = REPORT / 'CLAIM_LEDGER.json'
AMENDMENT = REPORT / 'MOTION_CLAIM_AMENDMENT_v0.1.json'
SOURCE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery/fixed-window-routes-84a85c92-363dbe1b/MOTION_UNCERTAINTY.json')
KEYS = {
    'native-discovery:gate-velocity-per-64-token-step': 'gate_velocity_TV_per_64_token_step',
    'native-discovery:gate-acceleration-second-difference': 'gate_acceleration_second_difference_L1_over_2',
    'native-discovery:frequent-expert-turnover': 'top8_frequent_expert_turnover',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    if source['schema'] != 'native-fixed-window-motion-uncertainty-v1' or source['sha256'] != digest({k: v for k, v in source.items() if k != 'sha256'}):
        raise ValueError('native motion interval source seal differs')
    claims = json.loads(AMENDMENT.read_text())
    if {row['id'] for row in claims} != set(KEYS) or len(claims) != len(KEYS):
        raise ValueError('motion amendment does not have exactly the registered three claims')
    for row in claims:
        estimate = source['estimates'][KEYS[row['id']]]
        if (row['source'] != str(SOURCE) or row['estimate'] != estimate['estimate'] or
            row['uncertainty'] != estimate['simultaneous_interval']):
            raise ValueError('amendment claim differs from saved estimate or interval')
    ledger = json.loads(LEDGER.read_text())
    if not isinstance(ledger, list) or len({row['id'] for row in ledger}) != len(ledger):
        raise ValueError('claim ledger has duplicate IDs or wrong format')
    prior = {row['id']: row for row in ledger}
    existing = [row['id'] in prior for row in claims]
    if any(existing):
        if not all(existing) or any(prior[row['id']] != row for row in claims):
            raise ValueError('partial or changed motion claim amendment already present')
    else:
        ledger += claims
        LEDGER.write_text(json.dumps(ledger, indent=1) + '\n')
    print(json.dumps({'ledger': str(LEDGER), 'claims': len(ledger),
                      'source_sha256': source['sha256']}))


if __name__ == '__main__':
    main()

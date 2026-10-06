"""Seal CPU resource price for B4 precision receipt aggregation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
SOURCE = REPO / 'scripts/experimental_resume/aggregate_r3e_b4_precision_v2.py'
OUT = REPORT / 'R3E_B4_PRECISION_AGGREGATE_PRICE_v1.json'


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def main():
    parent = json.loads((REPORT / 'R3E_B4_PRECISION_PRICE_v2.json').read_text())
    if parent['sha256'] != digest({k: v for k,v in parent.items() if k != 'sha256'}):
        raise ValueError('changed parent simulation price')
    body = {'schema': 'r3e-b4-precision-aggregation-price-v1', 'status': 'READY',
            'parent_price_sha256': parent['sha256'],
            'driver_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'inputs': '21 sealed result receipts with exactly one copy of replicate IDs 0..999',
            'output': 'steering-v1/runs/resume-v1/r3e-b4-precision-v2/summary.json',
            'cpus': 2, 'memory': '16G', 'wall_seconds': 1200,
            'max_core_hours': 2/3, 'gpus': 0,
            'dependency': 'afterok entire 20-task array, smoke PASS',
            'slurm': {'account': 'iscrc_miosr', 'partition': 'boost_usr_prod', 'qos': 'normal'}}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing aggregation price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'price': str(OUT), 'seal': value['sha256']}))


if __name__ == '__main__':
    main()

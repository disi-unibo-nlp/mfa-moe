"""Seal one bounded clean B1 cache-path recovery after the failed v1 preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
OUT = REPORT / 'R3E_B1_CLEAN_PRICE_v2.json'


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    v1 = json.loads((REPORT / 'R3E_B1_CLEAN_PRICE_v1.json').read_text())
    if v1['sha256'] != digest({k: v for k, v in v1.items() if k != 'sha256'}):
        raise ValueError('prior clean B1 price seal differs')
    body = {'schema': 'r3e-clean-b1-price-v2', 'status': 'READY_CPU_ONE_RECOVERY',
            'source_price_sha256': v1['sha256'], 'preflight_seal': v1['preflight_seal'],
            'failed_job': {'job_id': '59187838', 'state': 'FAILED', 'exit_code': '1:0',
                           'elapsed_seconds': 22, 'cpus': 8,
                           'charged_core_hours_upper_bound': 8 * 22 / 3600,
                           'cause': 'copied estimator package default OUT_GPT resolves to absent frozen-code/results path; no fit began'},
            'corrected_driver_sha256': sha(REPO / 'scripts/experimental_resume/run_r3e_b1_clean_v2.py'),
            'corrected_sbatch_sha256': sha(REPO / 'scripts/experimental_resume/run_r3e_b1_clean_v2.sbatch'),
            'corrected_token_cache': '/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/dynamics-routing/results/B1/tokens',
            'new_job': {'cpus': 8, 'memory': '64G', 'gpus': 0, 'wall_seconds': 1800,
                        'max_core_hours': 4.0},
            'amended_total_ceiling_core_hours': 4.1,
            'amended_total_ceiling_GPU_hours': 0.0,
            'scientific_specification': 'unchanged clean population, frozen family folds, B1 estimator and diagnostics',
            'resubmission_rule': 'one recovery; a later attempt needs a new receipt'}
    if body['failed_job']['charged_core_hours_upper_bound'] + body['new_job']['max_core_hours'] > 4.1:
        raise ValueError('one recovery exceeds amended B1 stage ceiling')
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing B1 recovery price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'out': str(OUT), 'sha256': value['sha256'],
                      'ceiling_core_hours': 4.1}))


if __name__ == '__main__':
    main()

"""Price complete dense discovery labels from the finished 200-item GPU audit."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
PARITY = ROOT / ('steering-v1/runs/routing-control-v1/dense-judge-parity/'
                 'results-8e939bee-2341bac1/PARITY.json')
UNITS = ROOT / 'steering-v1/runs/routing-control-v1/dense-discovery/UNITS.json'
LABEL_DRIVER = REPO / 'scripts/experimental_resume/label_dense_discovery.py'
OUT = REPO / 'report/experimental-resume-v1/DENSE_DISCOVERY_LABEL_PRICE.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'changed input seal: {path}')
    return value


def main():
    parity, units = sealed(PARITY), sealed(UNITS)
    if parity['schema'] != 'dense-judge-parity-v1' or units['schema'] != 'dense-discovery-units-v1':
        raise ValueError('unexpected input schema')
    job_id = parity['job_id']
    output = subprocess.run(['sacct', '-j', job_id, '-n', '-P', '-o', 'JobID,State,ExitCode'],
                            check=True, text=True, capture_output=True).stdout
    rows = [row.split('|') for row in output.splitlines() if row.startswith(job_id + '|')]
    if len(rows) != 1 or rows[0][1:3] != ['COMPLETED', '0:0']:
        raise ValueError('parity Slurm parent has not completed cleanly')
    records = parity['records']
    if len(records) != 200 or units['attempts'] != 48 or units['sentences'] != 5659:
        raise ValueError('prospective audit or discovery counts changed')
    covered = [r for r in records if r['new_label'] is not None and r['finish_reason'] == 'stop']
    agreed = [r for r in records if r['new_label'] == r['historical_label'] and r['finish_reason'] == 'stop']
    by_class = Counter(r['historical_label'] for r in records)
    agreement = Counter(r['historical_label'] for r in agreed)
    weak = [name for name, n in by_class.items() if agreement[name] / n < .5]
    qualified = len(covered) >= 190 and len(agreed) >= 140
    if abs(parity['coverage']-len(covered)/200)>1e-12:
        raise ValueError('parity coverage differs from raw records')
    if not qualified:
        status = 'HOLD_JUDGE_QUALIFICATION_FAILED'
        price = None
    else:
        # The audit is class-balanced; the full input is shorter on average.
        # Twofold generation multiplier covers class mix, latency variance and
        # prefill; a second full cold load and two teardown allowances cover
        # one checkpointed continuation. Both GPUs count for every second.
        generation = parity['generation_seconds'] * units['sentences'] / 200 * 2.
        seconds = generation + 2 * parity['load_seconds'] + 2 * 196
        gpu_hours = 2 * seconds / 3600
        ceiling = math.ceil(gpu_hours * 4) / 4
        first_job_seconds = generation + parity['load_seconds'] + 196
        first_job_hours = math.ceil(first_job_seconds / 1800) / 2
        price = {'audit_generation_seconds': parity['generation_seconds'],
                 'audit_load_seconds': parity['load_seconds'],
                 'full_to_audit_count_ratio': units['sentences']/200,
                 'generation_safety_factor': 2.,
                 'extra_cold_loads_for_resume': 1,
                 'shutdown_allowances_seconds': 392,
                 'complete_stage_seconds': seconds,
                 'complete_stage_GPU_hours': gpu_hours,
                 'revised_stage_ceiling_GPU_hours': ceiling,
                 'first_job_time_limit_hours': first_job_hours,
                 'unit': 'two A100 GPU allocation hours',
                 'coverage_scope': 'all 5659 prompts, prefill, decode, one retry load, shutdown',
                 'qualification_limits': 'direct LLM class consistency only; no semantic or online detector qualification'}
        status = 'PASS_COMPLETE_LABEL_STAGE_PRICE'
    value = {'schema': 'dense-discovery-label-price-v1',
             'parity_sha256': parity['sha256'], 'units_sha256': units['sha256'],
             'label_driver_sha256': hashlib.sha256(LABEL_DRIVER.read_bytes()).hexdigest(),
             'audit_job_id': job_id, 'audit_final_state': rows[0][1],
             'audit_exit_code': rows[0][2],
             'operational_coverage': len(covered)/200,
             'all_assigned_agreement': len(agreed)/200,
             'per_historical_class_agreement': {k:agreement[k]/v for k,v in sorted(by_class.items())},
             'weak_historical_classes': weak,
             'status': status, 'complete_price': price,
             'user_authorization': 'RESOURCE_AUTHORIZATION_EXPANDED.json; measured revised stage ceiling',
             'scientific_limit': 'No human ground truth; class agreement is operational consistency.'}
    value['sha256'] = digest(value)
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'status': status, 'price': price, 'weak_classes': weak}, default=str))


if __name__ == '__main__':
    main()

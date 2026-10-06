"""Seal a bounded clean B1 refit price from the completed historical full-cohort job."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
PREFLIGHT = REPORT / 'R3E_CLEAN_PREFLIGHT_v1.json'
OUT = REPORT / 'R3E_B1_CLEAN_PRICE_v1.json'


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = json.loads(PREFLIGHT.read_text())
    if p['sha256'] != digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('R3-E clean preflight seal differs')
    n = p['populations']['exact_id_clean']['questions']
    if n != 509 or p['design']['outcomes_read']:
        raise ValueError('unexpected clean B1 workload')
    historical_wall = 270
    projected = historical_wall * n / 1510 * 5 / 4
    stress = projected * 4 + 300
    if stress >= 1800:
        raise ValueError('clean B1 stress scenario exceeds 30-minute allocation')
    body = {'schema': 'r3e-clean-b1-price-v1', 'status': 'READY_CPU',
            'preflight_seal': p['sha256'],
            'driver_sha256': sha(REPO / 'scripts/experimental_resume/run_r3e_b1_clean.py'),
            'sbatch_sha256': sha(REPO / 'scripts/experimental_resume/run_r3e_b1_clean.sbatch'),
            'workload': {'clean_questions': n, 'clean_families': 496, 'layer_pairs': 43,
                         'outer_folds': 5, 'repeats': 5,
                         'whole_topk_permutations': 20, 'within_token_class_shuffles': 20,
                         'bootstraps': 5000, 'sign_flips': 10000},
            'historical_anchor': {'job_id': '59024489', 'state': 'COMPLETED',
                                  'elapsed_seconds': historical_wall, 'cpus': 8,
                                  'questions': 1510, 'outer_folds': 4, 'repeats': 5},
            'linear_projected_wall_seconds': projected,
            'fourfold_stress_plus_setup_seconds': stress,
            'slurm': {'account': 'iscrc_miosr', 'partition': 'boost_usr_prod',
                      'qos': 'normal', 'cpus': 8, 'memory': '64G',
                      'gpus': 0, 'wall_seconds': 1800},
            'max_core_hours': 4.0, 'max_gpu_hours': 0.0,
            'dependency': 'sealed R3E_CLEAN_PREFLIGHT_v1.json; existing frozen GPT A token cache',
            'scope': 'outcome-blind clean B1 observational refit; B4 feature freeze and correctness remain separate'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing B1 price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'out': str(OUT), 'sha256': value['sha256'],
                      'max_core_hours': 4.0, 'stress_seconds': stress}))


if __name__ == '__main__':
    main()

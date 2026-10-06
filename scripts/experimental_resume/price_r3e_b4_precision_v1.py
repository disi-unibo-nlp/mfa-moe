"""Seal the complete 1,000-replicate R3-E B4 CPU precision stage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
REPORT = REPO / 'report/experimental-resume-v1'
PKG = REPO / 'scripts/experimental_resume/r3e_clean_v2/dynrt'
FEATURES = S / 'runs/resume-v1/r3e-b4-features-v2'
PRIMARY = S / 'runs/resume-v1/r3e-cv-clean-v1'
OUT = REPORT / 'R3E_B4_PRECISION_PRICE_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    frozen = json.loads((FEATURES / 'FEATURES_FROZEN_B4.json').read_text())
    if (frozen.get('sha256') != digest({k: v for k, v in frozen.items()
                                       if k not in ('sha256', 'time', 'seconds', 'slurm_job')}) or
        frozen['outcomes_read'] is not False or frozen['universe']['attempts'] != 509 or
        frozen['design']['training_sets'] != 100):
        raise ValueError('B4 feature freeze changed')
    source = REPO / 'scripts/experimental_resume/run_r3e_b4_precision_v1.py'
    paths = [FEATURES / 'FEATURES_FROZEN_B4.json', FEATURES / 'features_A.npz',
             PRIMARY / 'full.json', PRIMARY / 'full-predictions.npz',
             REPORT / 'R3E_CLEAN_PREFLIGHT_v1.json',
             REPORT / 'R3E_RESOURCE_AND_METHODS_AMENDMENT_v0.2.md',
             REPORT / 'family-freeze.json', S / 'manifests/split-v1.json',
             Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/v3_analysis/results-r2/gpt/A/attempts.parquet'),
             S.parent / 'dynamics-routing/results/gpt/A/sentences.parquet',
             S.parent / 'dynamics-routing/results/gpt/A/hist.npz']
    paths += [PKG / n for n in ('cv.py', 'd3c_cv.py', 'd3c_features.py',
                                'd3c_freeze.py', 'controls.py', 'common.py',
                                'data.py', 'features.py', 'extract.py')]
    inputs = {str(p): sha(p) for p in paths}
    body = {'schema': 'r3e-b4-calibrated-precision-price-v1', 'status': 'READY_SMOKE',
            'population': {'assigned_questions': 509, 'assigned_families': 496,
                           'scored_questions': 508, 'scored_families': 495},
            'freeze_sha256': frozen['sha256'], 'driver_sha256': sha(source),
            'input_sha256': inputs,
            'design': {'targets_oracle_KL_nats_per_question': [0, .001, .003, .005, .010],
                       'replicates_per_target': 200, 'total_replicates': 1000,
                       'bootstrap_per_replicate': 1000,
                       'models': ['C+28', 'C+28+PA+PAlex+PA_sd', 'C+28+three iid noise columns'],
                       'estimator': 'frozen five-repeat five-outer three-inner duplicate-family CV',
                       'generator': 'five-fold original-outcome cross-fitted Bernoulli baseline; in-fold PA residualized signal and target KL calibration; conditionally independent question outcomes',
                       'no_heldout_original_outcomes_in_generator': True},
            'stages': {'smoke': {'ids': [0,1,2,3,4], 'cpus': 8, 'memory': '64G',
                                 'wall_seconds': 1800, 'max_core_hours': 4},
                       'shards': {'count': 20, 'ids_after_smoke': 995, 'cpus_each': 8,
                                  'memory_each': '64G', 'wall_seconds_each': 1800,
                                  'max_core_hours': 80, 'requires_smoke_pass': True}},
            'total_max_core_hours': 84,
            'reprice_rule': 'After exact smoke, if 50 replicates plus 4x overhead cannot fit 30 minutes, stop and amend before shards.',
            'slurm': {'account': 'iscrc_miosr', 'partition': 'boost_usr_prod',
                      'qos': 'normal', 'gpus': 0},
            'output': str(S / 'runs/resume-v1/r3e-b4-precision-v1'),
            'interpretation': 'Power and interval calibration of this tested estimator only; no causal steering or equivalence from non-rejection.'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing precision price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'price': str(OUT), 'seal': value['sha256'],
                      'max_core_hours': value['total_max_core_hours']}))


if __name__ == '__main__':
    main()

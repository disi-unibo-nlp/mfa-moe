"""Seal clean B4 nested correctness CV CPU stage after outcome-blind feature freeze."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
REPORT = REPO / 'report/experimental-resume-v1'
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
FEATURES = S / 'runs/resume-v1/r3e-b4-features-v2'
PKG = REPO / 'scripts/experimental_resume/r3e_clean_v2/dynrt'
OUT = REPORT / 'R3E_CV_CLEAN_PRICE_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    v = json.loads(path.read_text())
    if v.get('sha256') != digest({k: x for k, x in v.items() if k != 'sha256'}):
        raise ValueError(f'changed seal: {path}')
    return v


def main():
    frozen = json.loads((FEATURES / 'FEATURES_FROZEN_B4.json').read_text())
    pre = sealed(REPORT / 'R3E_CLEAN_PREFLIGHT_v1.json')
    if (frozen.get('sha256') != digest({k: x for k, x in frozen.items()
                                       if k not in ('sha256','time','seconds','slurm_job')}) or
        frozen['outcomes_read'] is not False or frozen['universe']['attempts'] != 509 or
        frozen['design']['training_sets'] != 100 or frozen['design']['pairs'] != 43 or
        frozen['design']['n_shuffles'] != 10 or
        frozen['design']['sets_digest'] != pre['populations']['exact_id_clean']['nested_training_set_keys_sha256']):
        raise ValueError('clean B4 feature freeze differs')
    input_paths = [FEATURES / 'FEATURES_FROZEN_B4.json', FEATURES / 'features_A.npz',
                   REPO / 'report/experimental-resume-v1/family-freeze.json',
                   S / 'manifests/split-v1.json', REPORT / 'R3E_CLEAN_PREFLIGHT_v1.json',
                   REPORT / 'R3E_RESOURCE_AND_METHODS_AMENDMENT_v0.2.md',
                   Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/v3_analysis/results-r2/gpt/A/attempts.parquet')]
    input_paths += [PKG / x for x in ('cv.py','controls.py','common.py','data.py','d3c_cv.py',
                                     'd3c_features.py','d3c_freeze.py','features.py','extract.py')]
    inp = {str(p): sha(p) for p in input_paths}
    body = {'schema': 'r3e-clean-b4-correctness-cv-price-v1',
            'status': 'READY_CV_SMOKE_AND_FULL',
            'population': {'assigned_questions': 509, 'families': 496,
                           'sensitivity_excluding_confirm_connected': 483,
                           'sensitivity_status': 'requires separately frozen 483-row training-fold features'},
            'freeze_seal': frozen['sha256'],
            'fold_keys_sha256': frozen['design']['sets_digest'],
            'driver_sha256': sha(REPO / 'scripts/experimental_resume/run_r3e_cv_clean_v1.py'),
            'input_sha256': inp,
            'output_path': str(S / 'runs/resume-v1/r3e-cv-clean-v1'),
            'design': {'outer': 5, 'inner': 3, 'full_repeats': 5, 'models': 6,
                       'family_clustered_bootstrap_full': 5000,
                       'simultaneous_comparisons': 5,
                       'outcome': 'observed correctness; retain all assigned and record any missing outcomes',
                       'no_GPT_B_outcomes_in_primary': True},
            'stages': {'smoke': {'cpus': 2, 'wall_seconds': 1200, 'repeats': 1,
                                 'bootstrap': 200, 'memory': '32G', 'max_core_hours': 2/3},
                       'full': {'cpus': 8, 'wall_seconds': 14400, 'repeats': 5,
                                'bootstrap': 5000, 'memory': '64G', 'max_core_hours': 32.0,
                                'requires_smoke_pass': True}},
            'total_max_core_hours': 32 + 2/3,
            'compute_rationale': 'Smoke runs one repeat of five outer folds and estimates full runtime; full stage covers 25 outer folds, six nested ridge models, family-clustered bootstrap and saved predictions. CPU remains appropriate for small tabular matrices; GPU load overhead is not justified.',
            'slurm': {'account': 'iscrc_miosr', 'partition': 'boost_usr_prod', 'qos': 'normal',
                      'gpus': 0},
            'reprice_rule': 'If the full stage cannot fit its four-hour allocation based on smoke, stop before submission and amend.'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing clean CV price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'price': str(OUT), 'seal': value['sha256'],
                      'max_core_hours': value['total_max_core_hours']}))


if __name__ == '__main__':
    main()

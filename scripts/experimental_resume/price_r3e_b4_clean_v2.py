"""Seal complete source, data, method and CPU price for clean B4 feature freeze."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = REPO / 'scripts/experimental_resume/r3e_clean_v2/dynrt'
REPORT = REPO / 'report/experimental-resume-v1'
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
CACHE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/dynamics-routing/results')
OUT = REPORT / 'R3E_B4_CLEAN_PRICE_v2.json'


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
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid seal: {path}')
    return value


def main():
    pre = sealed(REPORT / 'R3E_CLEAN_PREFLIGHT_v1.json')
    b1 = sealed(S / 'runs/resume-v1/r3e-b1-clean-v2/result.json')
    if (pre['populations']['exact_id_clean']['attempts'] != 509 or
        pre['populations']['exact_id_clean']['nested_training_sets'] != 100 or
        b1['population']['questions'] != 509):
        raise ValueError('clean cohort or folds changed')
    code = {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}
    files = [REPORT / 'R3E_CLEAN_PREFLIGHT_v1.json',
             S / 'runs/resume-v1/r3e-b1-clean-v2/result.json',
             REPORT / 'R3E_RESOURCE_AND_METHODS_AMENDMENT_v0.2.md']
    for sub in ('B1/tokens', 'B4/tokens_B'):
        files.extend(CACHE / sub / n for n in ('tokens.npz', 'sentences.parquet',
                                               'attempts.parquet', 'provenance.json'))
    inp = {str(p): sha(p) for p in files}
    result = {
        'schema': 'r3e-b4-clean-feature-price-v2',
        'status': 'READY_OUTCOME_BLIND_FEATURES',
        'population': {'source_preflight_sha256': pre['sha256'], 'attempts': 509,
                       'families': 496, 'training_sets': 100},
        'method': 'prospective in-fold PAlex with ten training-token conditional pseudo-class draws; reading-B scoring',
        'method_amendment_sha256': inp[str(REPORT / 'R3E_RESOURCE_AND_METHODS_AMENDMENT_v0.2.md')],
        'preflight_seal': pre['sha256'],
        'b1_seal': b1['sha256'],
        'wrapper_sha256': sha(REPO / 'scripts/experimental_resume/run_r3e_b4_clean_v2.py'),
        'test_sha256': sha(REPO / 'scripts/experimental_resume/test_r3e_fold_shuffle_v2.py'),
        'code_sha256': code,
        'inputs_sha256': inp,
        'output_path': str(S / 'runs/resume-v1/r3e-b4-features-v2'),
        'stages': {
            'smoke': {'pairs': 1, 'pseudo_draws': 1, 'cpus': 2, 'wall_seconds': 900,
                      'memory': '48G', 'max_core_hours': 0.5},
            'full': {'pairs': 43, 'pseudo_draws': 10, 'cpus': 16, 'wall_seconds': 7200,
                     'memory': '128G', 'max_core_hours': 32.0, 'requires_smoke_pass': True},
        },
        'total_max_core_hours': 32.5,
        'slurm': {'account': 'iscrc_miosr', 'partition': 'boost_usr_prod', 'qos': 'normal',
                  'gpus': 0, 'one_job_per_stage': True},
        'actual_cost_rule': 'Slurm allocated CPU-hours including failed attempts; do not report ceilings as expenditure',
        'uncertainty': 'Historical 80-set 43-pair ten-shuffle freeze used about 5.04 core-hours. The clean 100-set fold-only reference changes setup cost; short smoke validates but does not fully predict the total.',
    }
    value = {**result, 'sha256': digest(result)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing clean B4 price differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'price': str(OUT), 'seal': value['sha256'], 'max_core_hours': 32.5}))


if __name__ == '__main__':
    main()

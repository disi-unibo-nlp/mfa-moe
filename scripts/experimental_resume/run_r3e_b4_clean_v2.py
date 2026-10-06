"""Source-bound clean B4 feature freeze; no outcome data are read."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
ROOT = REPO / 'scripts/experimental_resume/r3e_clean_v2'
PRICE = REPO / 'report/experimental-resume-v1/R3E_B4_CLEAN_PRICE_v2.json'
OUT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/resume-v1/r3e-b4-features-v2')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('smoke', 'full'))
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.getuid() != os.stat(REPO).st_uid:
        raise ValueError('B4 feature fit must run under owner Slurm allocation')
    price = json.loads(PRICE.read_text())
    if price.get('sha256') != digest({k: v for k, v in price.items() if k != 'sha256'}):
        raise ValueError('B4 price seal invalid')
    if sha(__file__) != price['wrapper_sha256']:
        raise ValueError('B4 wrapper changed')
    for rel, expected in price['code_sha256'].items():
        if sha(ROOT / 'dynrt' / rel) != expected:
            raise ValueError(f'B4 source changed: {rel}')
    for path, expected in price['inputs_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'B4 input changed: {path}')
    stage = price['stages'][args.stage]
    if os.environ.get('SLURM_CPUS_PER_TASK') != str(stage['cpus']):
        raise ValueError('Slurm CPU shape differs from price')
    if OUT != Path(price['output_path']):
        raise ValueError('B4 output path differs from price')
    output = OUT / ('features_A_smoke.npz' if args.stage == 'smoke' else 'features_A.npz')
    if output.exists():
        raise ValueError('B4 stage output already exists; inspect before a retry')
    OUT.mkdir(parents=True, exist_ok=True)
    binding = {'schema': 'r3e-b4-clean-run-binding-v2', 'stage': args.stage,
               'job_id': os.environ['SLURM_JOB_ID'], 'price_sha256': price['sha256'],
               'code_sha256': price['code_sha256'], 'preflight_sha256': price['preflight_seal']}
    marker = OUT / f'run-binding-{args.stage}.json'
    if marker.exists():
        raise ValueError('B4 stage binding already exists; do not silently rerun')
    marker.write_text(json.dumps({**binding, 'sha256': digest(binding)}, indent=1) + '\n')
    sys.path.insert(0, str(ROOT))
    from dynrt import d3c_freeze
    if args.stage == 'smoke':
        return d3c_freeze.main(['--workers', '2', '--max-pairs', '1', '--n-shuf', '1', '--tag', 'smoke'])
    return d3c_freeze.main(['--workers', '16', '--n-shuf', '10'])


if __name__ == '__main__':
    raise SystemExit(main())

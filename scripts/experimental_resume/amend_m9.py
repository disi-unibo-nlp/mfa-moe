"""Seal the measured M9 resource amendment without changing its frozen design.

Run only after the replay gate and complete-stage source price have passed their
own verification. This command prepares an authorization-bound resource artifact;
it does not submit an experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
AUTH = REPO / 'report/experimental-resume-v1/RESOURCE_AUTHORIZATION_EXPANDED.json'
DRIVER = REPO / 'scripts/experimental_resume/closure_driver_amended.py'
USER_MESSAGE = ('use all the needed GPU hours. Sbatch everything you need to do '
                'all the naalisis and runs to finish the paper succesfully')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('source M9 price seal differs')
    return value


def ceiling(hours):
    if not math.isfinite(hours) or hours < 0:
        raise ValueError('invalid M9 hours')
    return math.ceil((hours + .1 - 1e-9) * 10) / 10


def amend(price):
    auth = json.loads(AUTH.read_text())
    if auth.get('schema') != 'experimental-resume-expanded-authorization-v1' or auth.get('user_message') != USER_MESSAGE:
        raise ValueError('M9 resource amendment lacks the current user authorization')
    if price['status'] != 'AMENDMENT_REQUIRED_BEFORE_PRODUCTION':
        raise ValueError('measured price has no amendment requirement')
    generation = ceiling(price['generation_replay_loads_GPU_h'])
    grading = ceiling(price['grading_GPU_h'])
    body = {'schema': 'M9-complete-stage-amendment-v1',
            'status': 'PASS_COMPLETE_STAGE_AMENDED',
            'manifest_sha256': price['manifest_sha256'],
            'replay_sha256': price['replay_sha256'],
            'source_price_sha256': price['sha256'], 'source_price': price,
            'driver_sha256': hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
            'authority_sha256': hashlib.sha256(AUTH.read_bytes()).hexdigest(),
            'generation_replay_ceiling_GPU_h': generation,
            'grading_ceiling_GPU_h': grading,
            'combined_ceiling_GPU_h': generation + grading,
            'additional_above_historical_lines_GPU_h':
                max(0., generation - 2.) + max(0., grading - 1.1377777777777778),
            'per_stage_basis': 'Complete pessimistic measured source price plus 0.1 GPU-hour, rounded up to 0.1',
            'registered_generation_GPU_h': 2.,
            'registered_shared_grading_remaining_GPU_h': 1.1377777777777778,
            'scope': 'Frozen M9 single-cut 400 assignments, unchanged sampler and grading; no transfer from X2/X3 or routing-action study',
            'execution': 'Each Slurm allocation and all retries count toward this combined ceiling; checkpointed output is digest-bound',
            'interpretation': 'Resource approval only; no semantic, accuracy or utility finding'}
    return {**body, 'sha256': digest(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--price', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = amend(sealed(args.price))
    if args.out.exists():
        if sealed(args.out) != value:
            raise ValueError('M9 resource amendment already exists with different contents')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'path': str(args.out), 'sha256': value['sha256'],
                      'generation_replay_ceiling_GPU_h': value['generation_replay_ceiling_GPU_h'],
                      'grading_ceiling_GPU_h': value['grading_ceiling_GPU_h']}))


if __name__ == '__main__':
    main()

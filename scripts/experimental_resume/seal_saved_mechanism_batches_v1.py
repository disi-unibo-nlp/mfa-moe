"""Seal completed mechanism batches after the native engine exits its process.

The frozen GPU runner writes and seals every batch and SUMMARY.json, then its
native shutdown helper calls os._exit(0).  This separate CPU step validates
those immutable outputs and publishes the two shard seals and stage seal.  It
never generates a token or changes a batch receipt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_boundary_micro_screen as base
import run_mechanism_validation_v1 as mechanism

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
            'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not args.preflight and (not os.environ.get('SLURM_JOB_ID') or
                               not os.environ.get('SLURM_STEP_ID')):
        raise RuntimeError('completion seal requires a CPU Slurm step')

    manifest = base.sealed(mechanism.MANIFEST)
    price = base.sealed(mechanism.PRICE)
    mechanism.validate(manifest, base.__file__)
    if (price['manifest_sha256'] != manifest['sha256'] or
            price['status'] != 'PASS_COMPLETE_STAGE' or
            len(price['shards']) != 2):
        raise ValueError('complete-stage price or shard count differs')
    out = ROOT / ('mechanism-validation-v1-' + manifest['sha256'][:16])
    for index, shard in enumerate(price['shards']):
        directory = out / f'shard-{index:03d}'
        summary = base.sealed(directory / 'SUMMARY.json')
        expected = (shard['end_row'] - shard['start_row']) * 8
        if (summary['manifest_sha256'] != manifest['sha256'] or
                summary['requests'] != expected or
                summary['batches'] != expected // 4 or
                summary['status'] != 'COMPLETE_UNGRADED_EXPLORATORY_GENERATION'):
            raise ValueError(f'shard {index} generation summary incomplete')
    if args.preflight:
        print(json.dumps({'status': 'PASS_CPU_SEAL_PREFLIGHT',
                          'manifest_sha256': manifest['sha256'],
                          'requests': manifest['expected_requests'],
                          'batches': sum((s['end_row'] - s['start_row']) * 2
                                         for s in price['shards'])}))
        return

    for index, shard in enumerate(price['shards']):
        mechanism.completion(out / f'shard-{index:03d}', manifest,
                             (shard['end_row'] - shard['start_row']) * 8)
    mechanism.seal_stage(out, manifest, price)
    stage = base.sealed(out / 'STAGE_COMPLETION.json')
    if (stage['manifest_sha256'] != manifest['sha256'] or
            stage['counts']['assigned'] != manifest['expected_requests']):
        raise ValueError('sealed stage does not retain every assigned request')
    print(json.dumps({'status': 'COMPLETE_UNGRADED_GENERATION',
                      'stage_sha256': stage['sha256'],
                      'requests': stage['counts']['assigned'],
                      'families': manifest['accepted_family_count'],
                      'output': str(out / 'STAGE_COMPLETION.json')}), flush=True)


if __name__ == '__main__':
    main()

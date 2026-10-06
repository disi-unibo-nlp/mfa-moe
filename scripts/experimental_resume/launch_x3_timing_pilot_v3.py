"""Guarded, cost-only X3 timing-pilot plan/test/one-shot submission.

The stage ceiling covers one two-hour, two-A100 Slurm allocation. Its runner
checkpoint remains resumable, but a second allocation needs a new price receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
S = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1')
PRICE = REPO / 'report/experimental-resume-v1/X3_TIMING_PILOT_PRICE_v3.json'
PLANNER = S / 'code/s1-9a61e32f48c04c24/scripts/steer_submit.py'
SNAPSHOT = PLANNER.parent.parent
LABEL = 'x3-timing-pilot-v3'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid seal: {path}')
    return value


def verify():
    if socket.gethostname().split('.')[1:] != ['leonardo', 'local'] or os.getuid() != os.stat(REPO).st_uid:
        raise ValueError('launcher must run as owner on a LEONARDO host')
    v3 = sealed(PRICE)
    if v3['schema'] != 'x3-32k-timing-pilot-price-v3' or v3['status'] != 'READY_TIMING_ONLY':
        raise ValueError('v3 timing-only amendment is not ready')
    if sha(__file__) != v3['launcher_sha256']:
        raise ValueError('launcher differs from v3 amendment')
    v2 = sealed(REPO / 'report/experimental-resume-v1/X3_TIMING_PILOT_PRICE_v2.json')
    manifest = sealed(Path(v3['manifest_path']))
    if (v2['sha256'] != v3['source_price_sha256'] or
        manifest['sha256'] != v3['manifest_sha256'] or
        len(manifest['requests']) != 40 or manifest['n_shards'] != 1 or
        manifest['code_tree'] != v3['snapshot_tree_sha256']):
        raise ValueError('price, manifest, or frozen code-tree binding differs')
    for key, relpath in [('builder_sha256', 'scripts/experimental_resume/x3_build_v2.py'),
                         ('preparation_sha256', 'scripts/experimental_resume/prepare_x3_timing_pilot.py')]:
        if sha(REPO / relpath) != v3[key]:
            raise ValueError(f'{relpath} differs from amended pilot')
    if (str(SNAPSHOT) != v3['snapshot_path'] or sha(SNAPSHOT / 'MANIFEST.json') != v3['snapshot_manifest_file_sha256'] or
        sha(PLANNER) != v3['planner_sha256'] or
        v3['proposed_job_walltime_seconds'] != 7200 or v3['proposed_GPUs'] != 2 or
        v3['proposed_stage_ceiling_GPUh'] != 4.0):
        raise ValueError('snapshot, planner, or 4-GPU-hour stage ceiling differs')
    out = Path(v3['output_path'])
    if out.parent != Path(v3['manifest_path']).parent or out.name != 'results-' + manifest['sha256'][:16]:
        raise ValueError('output is not the fresh digest-bound timing-pilot path')
    return v3, out


def plan(v3):
    argv = [sys.executable, '-B', str(PLANNER), 'gpu', '--snapshot', str(SNAPSHOT),
            '--time=02:00:00', '--label', LABEL, '--plan-only', 'run',
            '--manifest', v3['manifest_path'], '--shard', '0', '--out', v3['output_path']]
    proc = subprocess.run(argv, capture_output=True, text=True, check=True)
    value = json.loads(proc.stdout)
    sbatch = value['argv']
    if (sbatch[0] != 'sbatch' or '--gres=gpu:2' not in sbatch or
        '--time=02:00:00' not in sbatch or
        f'--comment=steer:run:{LABEL}' not in sbatch or
        sbatch[-6:] != ['--manifest', v3['manifest_path'], '--shard', '0', '--out', v3['output_path']]):
        raise ValueError('frozen planner returned a different Slurm/workload shape')
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('plan', 'test-only', 'submit'))
    args = ap.parse_args()
    v3, out = verify()
    planned = plan(v3)
    if args.action == 'plan':
        print(json.dumps(planned, indent=1))
        return
    if args.action == 'test-only':
        test_argv = [a for a in planned['argv'] if a != '--parsable']
        test_argv.insert(1, '--test-only')
        proc = subprocess.run(test_argv, capture_output=True, text=True)
        print(json.dumps({'test_only_exit_code': proc.returncode, 'stdout': proc.stdout,
                          'stderr': proc.stderr, 'sbatch_argv': test_argv}, indent=1))
        if proc.returncode:
            raise SystemExit(proc.returncode)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'pilot output already exists; a further allocation needs repricing: {out}')
    ledger = S / 'ledger.jsonl'
    if ledger.exists() and any(LABEL == json.loads(line).get('label') for line in ledger.read_text().splitlines()):
        raise ValueError('this one-shot pilot was already submitted; further GPU time needs repricing')
    out.mkdir(parents=True, exist_ok=True)
    binding = {'schema': 'x3-timing-pilot-output-binding-v1',
               'manifest_sha256': v3['manifest_sha256'], 'price_sha256': v3['sha256'],
               'snapshot_tree_sha256': v3['snapshot_tree_sha256']}
    (out / 'timing-pilot-binding.json').write_text(json.dumps({**binding, 'sha256': digest(binding)}, indent=1) + '\n')
    submit_argv = [sys.executable, '-B', str(PLANNER), 'gpu', '--snapshot', str(SNAPSHOT),
                   '--time=02:00:00', '--label', LABEL, 'run',
                   '--manifest', v3['manifest_path'], '--shard', '0', '--out', v3['output_path']]
    subprocess.run(submit_argv, check=True)


if __name__ == '__main__':
    main()

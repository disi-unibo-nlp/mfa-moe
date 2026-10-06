"""Guarded X3 timing-only recovery after v3's three-hour admission margin.

The stage ceiling covers one three-hour, two-A100 Slurm allocation. Its runner
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
PRICE = REPO / 'report/experimental-resume-v1/X3_TIMING_PILOT_PRICE_v4.json'
PLANNER = S / 'code/s1-9a61e32f48c04c24/scripts/steer_submit.py'
SNAPSHOT = PLANNER.parent.parent
LABEL = 'x3-timing-pilot-v4'
RUNNER_FLAGS = ['--margin-seconds', '900', '--abort-margin-seconds', '600']


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
    v4 = sealed(PRICE)
    if v4['schema'] != 'x3-32k-timing-pilot-price-v4' or v4['status'] != 'READY_TIMING_ONLY_RECOVERY':
        raise ValueError('v4 timing-only recovery amendment is not ready')
    if sha(__file__) != v4['launcher_sha256']:
        raise ValueError('launcher differs from v4 amendment')
    v3 = sealed(REPO / 'report/experimental-resume-v1/X3_TIMING_PILOT_PRICE_v3.json')
    manifest = sealed(Path(v4['manifest_path']))
    if (v3['sha256'] != v4['source_price_sha256'] or
        manifest['sha256'] != v4['manifest_sha256'] or
        len(manifest['requests']) != 40 or manifest['n_shards'] != 1 or
        manifest['code_tree'] != v4['snapshot_tree_sha256']):
        raise ValueError('price, manifest, or frozen code-tree binding differs')
    previous = Path(v3['output_path'])
    status = json.loads((previous / 'shard-0.status.json').read_text())
    if (status.get('manifest_sha256') != manifest['sha256'] or status.get('n_done') != 0 or
        status.get('tokens_this_run') != 0 or status.get('status') != 'deadline'):
        raise ValueError('v3 failure metadata differs; recovery requires a fresh amendment')
    if sha(previous / 'shard-0.status.json') != v4['previous_status_sha256']:
        raise ValueError('v3 failure receipt changed')
    for key, relpath in [('builder_sha256', 'scripts/experimental_resume/x3_build_v2.py'),
                         ('preparation_sha256', 'scripts/experimental_resume/prepare_x3_timing_pilot.py')]:
        if sha(REPO / relpath) != v4[key]:
            raise ValueError(f'{relpath} differs from amended pilot')
    if (str(SNAPSHOT) != v4['snapshot_path'] or sha(SNAPSHOT / 'MANIFEST.json') != v4['snapshot_manifest_file_sha256'] or
        sha(PLANNER) != v4['planner_sha256'] or
        v4['proposed_job_walltime_seconds'] != 10800 or v4['proposed_GPUs'] != 2 or
        v4['proposed_stage_ceiling_GPUh'] != 6.0 or v4['runner_flags'] != RUNNER_FLAGS):
        raise ValueError('snapshot, planner, or six-GPU-hour stage ceiling differs')
    out = Path(v4['output_path'])
    if out.parent != Path(v4['manifest_path']).parent or out.name != 'results-' + manifest['sha256'][:16] + '-v4':
        raise ValueError('output is not the fresh digest-bound timing-pilot path')
    return v4, out


def plan(v4):
    argv = [sys.executable, '-B', str(PLANNER), 'gpu', '--snapshot', str(SNAPSHOT),
            '--time=03:00:00', '--label', LABEL, '--plan-only', 'run',
            '--manifest', v4['manifest_path'], '--shard', '0', '--out', v4['output_path'], *RUNNER_FLAGS]
    proc = subprocess.run(argv, capture_output=True, text=True, check=True)
    value = json.loads(proc.stdout)
    sbatch = value['argv']
    if (sbatch[0] != 'sbatch' or '--gres=gpu:2' not in sbatch or
        '--time=03:00:00' not in sbatch or
        f'--comment=steer:run:{LABEL}' not in sbatch or
        sbatch[-10:] != ['--manifest', v4['manifest_path'], '--shard', '0', '--out', v4['output_path'], *RUNNER_FLAGS]):
        raise ValueError('frozen planner returned a different Slurm/workload shape')
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('plan', 'test-only', 'submit'))
    args = ap.parse_args()
    v4, out = verify()
    planned = plan(v4)
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
               'manifest_sha256': v4['manifest_sha256'], 'price_sha256': v4['sha256'],
               'snapshot_tree_sha256': v4['snapshot_tree_sha256']}
    (out / 'timing-pilot-binding.json').write_text(json.dumps({**binding, 'sha256': digest(binding)}, indent=1) + '\n')
    submit_argv = [sys.executable, '-B', str(PLANNER), 'gpu', '--snapshot', str(SNAPSHOT),
                   '--time=03:00:00', '--label', LABEL, 'run',
                   '--manifest', v4['manifest_path'], '--shard', '0', '--out', v4['output_path'], *RUNNER_FLAGS]
    subprocess.run(submit_argv, check=True)


if __name__ == '__main__':
    main()

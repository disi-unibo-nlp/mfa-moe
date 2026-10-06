"""Bounded, checkpoint-aware continuation of the frozen R3-D anticipation fit.

Run only in Slurm. A prior failed slice without valid progress stops the chain.
No model/statistical code is changed; this wrapper bounds one CPU slice and
records its state before the next dependent slice is eligible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
CONTEXT = ROOT / 'forum/tests/r3_context'
DRIVER = CONTEXT / 'code/r3d_anticipation.py'
FROZEN = CONTEXT / 'FROZEN.anticipation.json'
RESULT = CONTEXT / 'anticipation-results.json'
CHECKPOINTS = CONTEXT / 'anticipation'
RECEIPTS = CONTEXT / 'anticipation-guard-v1'
PY = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python')
EXPECTED_DRIVER_SHA = '4f1e0e1aa57126eb3757a554560992314ffabbaf65310e007be40cb638f717ca'
EXPECTED_FROZEN_SHA = '62f1038003b18b09816635ada31d68136856477df335cbd54665942d159baa92'
TASKS = 1650
MAX_SLICES = 6
SLICE_COREH_MAX = 96
STAGE_COREH_CEILING = 1500
CURRENT_JOB = '59111360'
CHECKPOINT_NAME = re.compile(r'^[0-4]-[0-4]-[0-9a-f]{12}\.npz$')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    if path.exists():
        raise FileExistsError(path)
    value = {**value, 'sha256': digest(value)}
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)
    return value


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'changed receipt: {path}')
    return value


def checkpoint_count():
    return sum(CHECKPOINT_NAME.fullmatch(path.name) is not None for path in CHECKPOINTS.rglob('*.npz'))


def previous_state(job):
    for _ in range(12):
        output = subprocess.run(['sacct', '-j', job, '-n', '-P', '-o', 'JobID,State,ExitCode'],
                                check=True, text=True, capture_output=True).stdout
        rows = [r.split('|') for r in output.splitlines() if r.startswith(job + '|')]
        if len(rows) == 1 and rows[0][1] in ('TIMEOUT', 'COMPLETED', 'FAILED', 'CANCELLED'):
            return rows[0][1], rows[0][2]
        time.sleep(5)
    raise ValueError('previous Slurm parent final state unavailable after 60 seconds')


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-job', required=True)
    parser.add_argument('--slice-index', type=int, required=True)
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    if not job or socket.gethostname().startswith('login'):
        raise RuntimeError('anticipation continuation requires CPU Slurm')
    expected_guard = os.environ.get('R3D_GUARD_SHA')
    if not expected_guard or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected_guard:
        raise ValueError('submitted guard source digest changed')
    if not 1 <= args.slice_index <= MAX_SLICES:
        raise ValueError('unpriced continuation slice')
    if args.slice_index == 1 and args.previous_job != CURRENT_JOB:
        raise ValueError('first continuation must follow the verified current job')
    if args.slice_index > 1:
        prior = sealed(RECEIPTS / f'job-{args.previous_job}-start.json')
        if prior['slice_index'] != args.slice_index - 1:
            raise ValueError('continuation chain index differs')
    if hashlib.sha256(DRIVER.read_bytes()).hexdigest() != EXPECTED_DRIVER_SHA:
        raise ValueError('frozen anticipation driver changed')
    frozen = sealed(FROZEN)
    if frozen['sha256'] != EXPECTED_FROZEN_SHA:
        raise ValueError('frozen anticipation specification changed')
    # Even if every slice consumed its full allocation, total continuation
    # spending plus a 32-core-hour allowance for historical work stays below
    # the amended 1,500-core-hour stage ceiling.
    if 32 + (args.slice_index + 1) * SLICE_COREH_MAX > STAGE_COREH_CEILING:
        raise ValueError('prospective stage ceiling exceeded')
    state, exit_code = previous_state(args.previous_job)
    if args.slice_index == 1:
        if state not in ('TIMEOUT', 'COMPLETED'):
            raise ValueError(f'prior unqualified final state: {state} {exit_code}')
        if state == 'COMPLETED' and exit_code != '0:0':
            raise ValueError('prior completed with a nonzero exit code')
        before_previous = 18
    else:
        if state != 'COMPLETED' or exit_code != '0:0':
            raise ValueError(f'prior continuation did not complete cleanly: {state} {exit_code}')
        before_previous = prior['checkpoint_count_before']
    count = checkpoint_count()
    if not 18 <= count <= TASKS or count <= before_previous and not RESULT.exists():
        raise ValueError(f'prior slice made no valid checkpoint progress: {before_previous} -> {count}')
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    start = atomic(RECEIPTS / f'job-{job}-start.json',
                   {'schema': 'r3d-anticipation-guard-start-v1', 'job_id': job,
                    'previous_job': args.previous_job, 'previous_state': state,
                    'previous_exit_code': exit_code, 'slice_index': args.slice_index,
                    'checkpoint_count_before': count, 'driver_sha256': EXPECTED_DRIVER_SHA,
                    'frozen_sha256': EXPECTED_FROZEN_SHA})
    if RESULT.exists():
        sealed(RESULT)
        atomic(RECEIPTS / f'job-{job}-end.json',
               {'schema': 'r3d-anticipation-guard-end-v1', 'start_sha256': start['sha256'],
                'status': 'SKIPPED_ALREADY_COMPLETE', 'checkpoint_count_after': count})
        print(json.dumps({'status': 'SKIPPED_ALREADY_COMPLETE', 'job_id': job}), flush=True)
        return
    started = time.monotonic()
    process = subprocess.Popen([str(PY), '-B', '-u', str(DRIVER), '--workers', '4'],
                               start_new_session=True)
    try:
        code = process.wait(timeout=23*3600+40*60)
        timed = False
    except subprocess.TimeoutExpired:
        stop_group(process)
        code, timed = process.returncode, True
    after = checkpoint_count()
    if RESULT.exists():
        sealed(RESULT)
        status = 'COMPLETE_ANALYSIS'
    elif timed and after > count:
        status = 'CHECKPOINTED_TIME_SLICE'
    else:
        raise RuntimeError(f'anticipation fit failed or made no progress: code={code}, checkpoints={count}->{after}')
    atomic(RECEIPTS / f'job-{job}-end.json',
           {'schema': 'r3d-anticipation-guard-end-v1', 'start_sha256': start['sha256'],
            'status': status, 'child_exit_code': code,
            'elapsed_seconds': time.monotonic()-started,
            'checkpoint_count_after': after})
    print(json.dumps({'status': status, 'job_id': job, 'checkpoints': after}), flush=True)


if __name__ == '__main__':
    main()

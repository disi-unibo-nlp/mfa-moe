"""Run the frozen R3-D readout after verified sidecar checkpoint merges.

The original continuation guard only accepts a timeout or clean completion of
the canonical writer. That writer was intentionally cancelled at a verified
checkpoint boundary; this versioned wrapper handles that specific handoff.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
CONTEXT = ROOT / 'forum/tests/r3_context'
CHECKPOINTS = CONTEXT / 'anticipation'
REPORT = REPO / 'report/experimental-resume-v1'
DRIVER = CONTEXT / 'code/r3d_anticipation.py'
FROZEN = CONTEXT / 'FROZEN.anticipation.json'
PYTHON = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python')
EXPECTED_DRIVER_SHA = '4f1e0e1aa57126eb3757a554560992314ffabbaf65310e007be40cb638f717ca'
EXPECTED_FROZEN_SHA = '62f1038003b18b09816635ada31d68136856477df335cbd54665942d159baa92'
EXPECTED_CELLS = {
    model: {tag: (75 if model == 'qwen36' and tag.startswith('next') else 150)
            for tag in ('next', 'next-contained', 'switch', 'switch-contained',
                        'dest', 'dest-contained')}
    for model in ('gpt', 'qwen36')
}
FIT_NAME = re.compile(r'[0-4]-[0-4]-[0-9a-f]{12}\.npz\Z')


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'broken seal: {path}')
    return value


def job_state(job):
    out = subprocess.run(['sacct', '-j', str(job), '-X', '-P', '-n',
                          '-o', 'State,ExitCode'], check=True, capture_output=True,
                         text=True).stdout.strip().splitlines()
    if len(out) != 1:
        raise ValueError(f'cannot verify job {job}: {out}')
    return out[0].split('|')[:2]


def held_continuation():
    out = subprocess.run(['scontrol', 'show', 'job', '59115320'], check=True,
                         capture_output=True, text=True).stdout
    if not re.search(r'\bJobState=PENDING\b', out) or not re.search(r'\bReason=JobHeldUser\b', out):
        raise ValueError('original continuation 59115320 is not safely held')


def inventory():
    observed = {}
    for model, cells in EXPECTED_CELLS.items():
        observed[model] = {}
        for tag, expected in cells.items():
            folder = CHECKPOINTS / model / tag
            names = sorted(p.name for p in folder.glob('*.npz') if FIT_NAME.fullmatch(p.name))
            if len(names) != expected or len(names) != len(set(names)):
                raise ValueError(f'{model}/{tag}: {len(names)} fit files, expected {expected}')
            if list(folder.glob('*.pending')):
                raise ValueError(f'{model}/{tag}: incomplete pending checkpoint')
            observed[model][tag] = len(names)
    if sum(sum(cells.values()) for cells in observed.values()) != 1650:
        raise ValueError('R3-D total fit inventory differs from frozen 1650')
    return observed


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('R3-D finalization requires CPU Slurm')
    if sha(DRIVER) != EXPECTED_DRIVER_SHA or sealed(FROZEN)['sha256'] != EXPECTED_FROZEN_SHA:
        raise ValueError('frozen driver or specification differs')
    state = job_state('59111360')
    if not state[0].startswith('CANCELLED by 133943') or state[1] != '0:0':
        raise ValueError(f'canonical writer stop differs: {state}')
    held_continuation()
    merges = {}
    for kind, name, expected in (
        ('qwen', 'R3D_SIDECAR_MERGE_QWEN_v2.json', 750),
        ('gpt-dest', 'R3D_SIDECAR_MERGE_GPT_DEST_v2.json', 300),
    ):
        receipt = sealed(REPORT / name)
        if receipt['kind'] != kind or receipt['frozen_specification_sha256'] != EXPECTED_FROZEN_SHA:
            raise ValueError(f'{kind}: merge binding differs')
        if receipt['expected_fit_checkpoints'] != expected or len(receipt['copied']) + len(receipt['preserved_existing_valid']) != expected:
            raise ValueError(f'{kind}: incomplete merge receipt')
        if job_state(receipt['merge_job_id']) != ['COMPLETED', '0:0']:
            raise ValueError(f'{kind}: merge Slurm job failed')
        merges[kind] = receipt['sha256']
    before = inventory()
    final_path = CONTEXT / 'anticipation-results.json'
    if final_path.exists():
        raise FileExistsError(f'R3-D top-level result already exists: {final_path}')
    subprocess.run([str(PYTHON), '-B', '-u', str(DRIVER), '--workers', '4'], check=True)
    result = sealed(final_path)
    if result['frozen'] != EXPECTED_FROZEN_SHA:
        raise ValueError('R3-D final result has wrong specification')
    if set(result['models']) != set(EXPECTED_CELLS):
        raise ValueError('R3-D result model roster differs')
    for model, cells in EXPECTED_CELLS.items():
        if set(result['models'][model]) != set(cells):
            raise ValueError(f'{model}: final result cell roster differs')
        for tag in cells:
            sealed(CHECKPOINTS / model / tag / 'result.json')
    after = inventory()
    if after != before:
        raise ValueError('fit inventory changed during finalization')
    body = {
        'schema': 'r3d-after-verified-merge-finalization-v1',
        'job_id': os.environ['SLURM_JOB_ID'],
        'stopped_canonical_job_id': '59111360',
        'original_continuation_held_job_id': '59115320',
        'driver_sha256': EXPECTED_DRIVER_SHA,
        'frozen_specification_sha256': EXPECTED_FROZEN_SHA,
        'merge_receipts': merges,
        'fit_inventory': after,
        'result_sha256': result['sha256'],
    }
    receipt_path = REPORT / 'R3D_FINALIZATION_AFTER_MERGE_v1.json'
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    temp = receipt_path.with_name(receipt_path.name + '.part-' + os.environ['SLURM_JOB_ID'])
    temp.write_text(json.dumps({**body, 'sha256': digest(body)}, indent=1) + '\n')
    temp.replace(receipt_path)
    print(json.dumps({'status': 'COMPLETE', 'result': str(final_path),
                      'result_sha256': result['sha256'], 'receipt': str(receipt_path)}), flush=True)


if __name__ == '__main__':
    main()

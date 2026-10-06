"""Copy only independently verified missing R3-D fit checkpoints at stopped boundary.

The canonical driver reconstructs predictions from checkpoints. Existing
canonical fits are retained after validating their binding, fold, and shapes.
No continuation is released here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess

import numpy as np

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
CANON = ROOT / 'forum/tests/r3_context/anticipation'
SIDES = ROOT / 'steering-v1/runs/resume-v1'
REPORT = REPO / 'report/experimental-resume-v1'
FROZEN = '62f1038003b18b09816635ada31d68136856477df335cbd54665942d159baa92'
EXPECTED = {
    'qwen': ('qwen36', 'r3d-anticipation-qwen-shard-v1',
             REPORT / 'R3D_SIDECAR_VERIFY_QWEN_v2.json',
             REPORT / 'R3D_SIDECAR_MERGE_QWEN_v2.json'),
    'gpt-dest': ('gpt', 'r3d-anticipation-gpt-dest-shard-v1',
                 REPORT / 'R3D_SIDECAR_VERIFY_GPT_DEST_v2.json',
                 REPORT / 'R3D_SIDECAR_MERGE_GPT_DEST_v2.json'),
}


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    x = json.loads(path.read_text())
    if x.get('sha256') != digest({k: v for k, v in x.items() if k != 'sha256'}):
        raise ValueError(f'invalid sealed verification receipt: {path}')
    return x


def state(job):
    cmd = ['sacct', '-j', str(job), '-X', '-P', '-n', '-o', 'State,ExitCode']
    text = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    if len(text) != 1:
        raise ValueError(f'could not obtain one Slurm state for {job}: {text}')
    return text[0].split('|')[:2]


def check_hold():
    text = subprocess.run(['scontrol', 'show', 'job', '59115320'], capture_output=True,
                          text=True, check=True).stdout
    if not re.search(r'\bJobState=PENDING\b', text) or not re.search(r'\bReason=JobHeldUser\b', text):
        raise ValueError('first canonical continuation 59115320 is not user-held')
    # Slurm removes a fulfilled afterany dependency from scontrol output.
    # The canonical writer's terminal state is verified immediately before
    # this call; the exact continuation must still be user-held. Both live
    # afterany and fulfilled (null) states are therefore safe here.
    if not (re.search(r'\bDependency=afterany:59111360\b', text) or
            re.search(r'\bDependency=\(null\)(?:\s|$)', text)):
        raise ValueError('first continuation dependency differs')


def validate_existing(path, source):
    with np.load(source, allow_pickle=False) as s, np.load(path, allow_pickle=False) as d:
        if set(s.files) != set(d.files) or str(d['binding']) != FROZEN or not np.array_equal(d['test'], s['test']):
            raise ValueError(f'existing canonical checkpoint binding/fold differs: {path}')
        for key in d.files:
            if key.startswith('lp:') or key == 'stats':
                if d[key].shape != s[key].shape or not np.isfinite(d[key]).all():
                    raise ValueError(f'existing canonical checkpoint is invalid: {path}/{key}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=sorted(EXPECTED), required=True)
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('R3-D checkpoint merge requires CPU Slurm')
    model, side, verify_path, receipt_path = EXPECTED[args.kind]
    verify = sealed(verify_path)
    if verify['kind'] != args.kind or verify['model'] != model or verify['frozen_specification_sha256'] != FROZEN:
        raise ValueError('sidecar verification scope differs')
    canonical_state = state('59111360')
    if canonical_state[0] in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'):
        raise ValueError(f'canonical writer 59111360 has not stopped: {canonical_state}')
    for job in (verify['source_job_id'], verify['validation_job_id']):
        if state(job) != ['COMPLETED', '0:0']:
            raise ValueError(f'sidecar or verifier job {job} is not successful')
    check_hold()
    source_root = SIDES / side / model
    copies, retained = [], []
    for cell in verify['cells']:
        tag = cell['tag']
        src = source_root / tag
        dst = CANON / model / tag
        if len(cell['fit_file_sha256']) != cell['n_fit_checkpoints']:
            raise ValueError(f'{tag}: verification receipt count differs')
        dst.mkdir(parents=True, exist_ok=True)
        for name, expected_hash in sorted(cell['fit_file_sha256'].items()):
            if not re.fullmatch(r'[0-4]-[0-4]-[0-9a-f]{12}\.npz', name):
                raise ValueError(f'invalid checkpoint name in verifier: {name}')
            source = src / name
            dest = dst / name
            if sha(source) != expected_hash:
                raise ValueError(f'source checkpoint changed after verification: {source}')
            if dest.exists():
                validate_existing(dest, source)
                retained.append({'cell': tag, 'name': name, 'sha256': sha(dest)})
                continue
            stage = dst / ('.' + name + '.merge-' + os.environ['SLURM_JOB_ID'])
            if stage.exists():
                raise ValueError(f'stale staged file exists: {stage}')
            shutil.copyfile(source, stage)
            if sha(stage) != expected_hash:
                raise ValueError(f'staged checkpoint copy differs: {stage}')
            stage.replace(dest)
            copies.append({'cell': tag, 'name': name, 'sha256': expected_hash})
    body = {'schema': 'r3d-sidecar-checkpoint-merge-v2', 'kind': args.kind,
            'merge_job_id': os.environ['SLURM_JOB_ID'], 'source_verification_sha256': verify['sha256'],
            'canonical_stopped_job_id': '59111360', 'canonical_state': canonical_state,
            'held_continuation_job_id': '59115320', 'frozen_specification_sha256': FROZEN,
            'model': model, 'source_job_id': verify['source_job_id'],
            'expected_fit_checkpoints': verify['total_fit_checkpoints'],
            'copied': copies, 'preserved_existing_valid': retained,
            'scope': 'fit checkpoint files only; no sidecar result, predictions, or status copied; exact continuation remains user-held; fulfilled afterany dependency may appear as null'}
    if len(copies) + len(retained) != verify['total_fit_checkpoints']:
        raise ValueError('merge checkpoint count differs')
    value = {**body, 'sha256': digest(body)}
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != value:
            raise ValueError('existing merge receipt differs')
    else:
        temp = receipt_path.with_name(receipt_path.name + '.part-' + os.environ['SLURM_JOB_ID'])
        temp.write_text(json.dumps(value, indent=1) + '\n')
        temp.replace(receipt_path)
    print(json.dumps({'receipt': str(receipt_path), 'copied': len(copies),
                      'preserved': len(retained), 'expected': verify['total_fit_checkpoints']}), flush=True)


if __name__ == '__main__':
    main()

"""Independently verify the stopped R3-D writer's merged fit inventory.

This audits checkpoint bytes after the two merge jobs and leaves the held
continuation untouched. It does not run a fit or construct result tables.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
CANON = ROOT / 'forum/tests/r3_context/anticipation'
REPORT = REPO / 'report/experimental-resume-v1'
FROZEN = '62f1038003b18b09816635ada31d68136856477df335cbd54665942d159baa92'
FIT = re.compile(r'[0-4]-[0-4]-[0-9a-f]{12}\.npz\Z')
TAGS = ('next', 'next-contained', 'switch', 'switch-contained', 'dest', 'dest-contained')
SOURCE_CELLS = {
    'qwen': ('qwen36', 'R3D_SIDECAR_VERIFY_QWEN_v2.json',
             'R3D_SIDECAR_MERGE_QWEN_v2.json', '59206783', 750),
    'gpt-dest': ('gpt', 'R3D_SIDECAR_VERIFY_GPT_DEST_v2.json',
                 'R3D_SIDECAR_MERGE_GPT_DEST_v2.json', '59206978', 300),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(name):
    value = json.loads((REPORT / name).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'broken seal: {name}')
    return value


def state(job):
    out = subprocess.run(['sacct', '-j', str(job), '-X', '-P', '-n', '-o',
                          'State,ExitCode'], check=True, capture_output=True,
                         text=True).stdout.strip().splitlines()
    if len(out) != 1:
        raise ValueError(f'cannot verify job {job}: {out}')
    return out[0].split('|')[:2]


def check_hold():
    out = subprocess.run(['scontrol', 'show', 'job', '59115320'], check=True,
                         capture_output=True, text=True).stdout
    if not re.search(r'\bJobState=PENDING\b', out) or not re.search(r'\bReason=JobHeldUser\b', out):
        raise ValueError('original continuation is not held')
    if not (re.search(r'\bDependency=afterany:59111360\b', out) or
            re.search(r'\bDependency=\(null\)(?:\s|$)', out)):
        raise ValueError('unexpected continuation dependency')


def check_npz(path):
    with np.load(path, allow_pickle=False) as fit:
        if str(fit['binding']) != FROZEN or fit['test'].ndim != 1:
            raise ValueError(f'checkpoint binding or fold shape differs: {path}')
        if 'stats' not in fit.files or not any(key.startswith('lp:') for key in fit.files):
            raise ValueError(f'checkpoint fit arrays missing: {path}')
        for key in fit.files:
            if key == 'stats' or key.startswith('lp:'):
                if not np.isfinite(fit[key]).all():
                    raise ValueError(f'nonfinite checkpoint array: {path}/{key}')


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.getuid() != REPO.stat().st_uid:
        raise RuntimeError('merged R3-D inventory audit requires owner CPU Slurm')
    if not state('59111360')[0].startswith('CANCELLED by 133943'):
        raise ValueError('original canonical writer is not terminal')
    check_hold()
    if list(CANON.rglob('*.pending')) or list(CANON.rglob('*.merge-*')):
        raise ValueError('partial checkpoint artifact remains')

    copied_and_retained = {}
    receipts = {}
    for kind, (model, verify_name, merge_name, job, expected) in SOURCE_CELLS.items():
        verify, merge = sealed(verify_name), sealed(merge_name)
        if (verify['kind'] != kind or verify['model'] != model or
            verify['frozen_specification_sha256'] != FROZEN or
            verify['total_fit_checkpoints'] != expected or
            merge['kind'] != kind or merge['model'] != model or
            merge['frozen_specification_sha256'] != FROZEN or
            merge['source_verification_sha256'] != verify['sha256'] or
            merge['merge_job_id'] != job or
            merge['expected_fit_checkpoints'] != expected or
            state(verify['source_job_id']) != ['COMPLETED', '0:0'] or
            state(verify['validation_job_id']) != ['COMPLETED', '0:0'] or
            state(job) != ['COMPLETED', '0:0']):
            raise ValueError(f'{kind}: source or merge binding differs')
        reference = {(model, cell['tag'], name): value
                     for cell in verify['cells']
                     for name, value in cell['fit_file_sha256'].items()}
        if len(reference) != expected or any(not FIT.fullmatch(name) for _, _, name in reference):
            raise ValueError(f'{kind}: source fit name inventory differs')
        observed = {}
        for section in ('copied', 'preserved_existing_valid'):
            for entry in merge[section]:
                key = (model, entry['cell'], entry['name'])
                if key not in reference or key in observed:
                    raise ValueError(f'{kind}: merge contains missing/duplicate fit {key}')
                canonical_hash = sha(CANON / model / entry['cell'] / entry['name'])
                if canonical_hash != entry['sha256']:
                    raise ValueError(f'{kind}: canonical checkpoint changed: {key}')
                if section == 'copied' and canonical_hash != reference[key]:
                    raise ValueError(f'{kind}: copied checkpoint differs from verified source: {key}')
                observed[key] = canonical_hash
        if set(observed) != set(reference):
            raise ValueError(f'{kind}: incomplete merged fit inventory')
        copied_and_retained.update(observed)
        receipts[kind] = {'source': verify['sha256'], 'merge': merge['sha256'],
                          'copied': len(merge['copied']),
                          'retained': len(merge['preserved_existing_valid'])}

    all_hashes = {}
    counts = {}
    for model in ('gpt', 'qwen36'):
        counts[model] = {}
        for tag in TAGS:
            directory = CANON / model / tag
            fit_paths = sorted(path for path in directory.glob('*.npz') if FIT.fullmatch(path.name))
            expected = 75 if model == 'qwen36' and tag.startswith('next') else 150
            if len(fit_paths) != expected:
                raise ValueError(f'{model}/{tag}: {len(fit_paths)} fit files, expected {expected}')
            nonfit = {path.name for path in directory.glob('*.npz')
                      if not FIT.fullmatch(path.name)}
            expected_nonfit = ({'predictions.npz'} if model == 'gpt' and
                               tag in ('next', 'next-contained', 'switch',
                                       'switch-contained') else set())
            if nonfit != expected_nonfit:
                raise ValueError(f'{model}/{tag}: non-fit NPZ set differs: {nonfit}')
            fold_sets = {}
            for path in fit_paths:
                repeat, fold, feature = path.stem.split('-')
                fold_sets.setdefault(feature, set()).add((repeat, fold))
                check_npz(path)
                key = (model, tag, path.name)
                actual = sha(path)
                if key in copied_and_retained and actual != copied_and_retained[key]:
                    raise ValueError(f'fit hash changed during audit: {key}')
                all_hashes['/'.join(key)] = actual
            if len(fold_sets) != expected // 25 or any(len(pairs) != 25 for pairs in fold_sets.values()):
                raise ValueError(f'{model}/{tag}: incomplete repeated-fold grid')
            counts[model][tag] = len(fit_paths)
    if len(all_hashes) != 1650 or len(copied_and_retained) != 1050:
        raise ValueError('final exact fit total differs')
    check_hold()
    body = {'schema': 'r3d-merged-inventory-audit-v2', 'status': 'PASS',
            'job_id': os.environ['SLURM_JOB_ID'],
            'canonical_stopped_job_id': '59111360',
            'held_continuation_job_id': '59115320',
            'frozen_specification_sha256': FROZEN,
            'fit_counts': counts, 'fit_total': len(all_hashes),
            'sidecar_fit_total': len(copied_and_retained),
            'merge_receipts': receipts,
            'fit_hash_manifest_sha256': digest(all_hashes)}
    output = REPORT / 'R3D_MERGED_INVENTORY_AUDIT_v2.json'
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + '.part-' + os.environ['SLURM_JOB_ID'])
    temporary.write_text(json.dumps({**body, 'sha256': digest(body)}, indent=1) + '\n')
    temporary.replace(output)
    print(json.dumps({'status': body['status'], 'fit_total': body['fit_total'],
                      'receipt': str(output), 'sha256': digest(body)}), flush=True)


if __name__ == '__main__':
    main()

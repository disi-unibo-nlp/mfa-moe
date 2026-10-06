"""Verify every frozen R3-D sidecar fit before any canonical checkpoint copy.

Read-only except for a new sealed verification receipt. Run on CPU Slurm after
the named sidecar job completes. This script never modifies canonical files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import zipfile

import numpy as np

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
FROZEN = '62f1038003b18b09816635ada31d68136856477df335cbd54665942d159baa92'
DRIVER = '4f1e0e1aa57126eb3757a554560992314ffabbaf65310e007be40cb638f717ca'
CONFIG = {
    'gpt-dest': ('gpt', ('dest', 'dest-contained'),
                 'r3d-anticipation-gpt-dest-shard-v1',
                 'r3d-anticipation-gpt-dest-shard-cell-v1',
                 REPO / 'scripts/experimental_resume/r3d_anticipation_gpt_dest_shard.py'),
    'qwen': ('qwen36', ('next', 'next-contained', 'switch',
                       'switch-contained', 'dest', 'dest-contained'),
             'r3d-anticipation-qwen-shard-v1',
             'r3d-anticipation-qwen-shard-cell-v1',
             REPO / 'scripts/experimental_resume/r3d_anticipation_qwen_shard.py'),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if digest({k: v for k, v in value.items() if k != 'sha256'}) != value.get('sha256'):
        raise ValueError(f'changed sidecar receipt: {path}')
    return value


def safe_npy_shape(npz_path, member):
    """Read a NumPy member's header without unpickling object-valued question IDs."""
    with zipfile.ZipFile(npz_path) as archive:
        with archive.open(member + '.npy') as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in ((2, 0), (3, 0)):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f'unsupported NPY header version {version}')
    return shape, dtype


def expected_chunks(tag):
    routes = ['composition16'] if tag.startswith('next') else ['k16', 'k32']
    models = {'base', 'weak'}
    for route in routes:
        models.update((route, 'oracle_' + route))
        models.update(f'noise_{route}_{draw}' for draw in range(20))
    names = sorted(models)
    return [names[i:i + 8] for i in range(0, len(names), 8)]


def verify_cell(root, model, tag, schema, top, sidecar_sha):
    cell = root / model / tag
    receipt = sealed(cell / 'result.json')
    if (receipt['schema'] != schema or receipt['frozen'] != FROZEN or
        receipt['driver_sha256'] != DRIVER or
        receipt['scheduler_sha256'] != sidecar_sha or
        receipt['job_id'] != top['job_id'] or
        receipt['model'] != model or receipt['tag'] != tag or
        receipt['result'] != top['models'][model][tag] or
        receipt['result']['status'] != 'CLEAN'):
        raise ValueError(f'{tag}: sidecar cell receipt binding differs')
    chunks = expected_chunks(tag)
    expected = {f'{r}-{f}-{hashlib.sha256("|".join(chunk).encode()).hexdigest()[:12]}.npz':
                (r, f, chunk) for chunk in chunks for r in range(5) for f in range(5)}
    actual = {p.name: p for p in cell.glob('[0-4]-[0-4]-*.npz')}
    if set(actual) != set(expected):
        raise ValueError(f'{tag}: missing/extra fit checkpoint names: {len(actual)}/{len(expected)}')
    pred_path = cell / 'predictions.npz'
    with np.load(pred_path, allow_pickle=False) as saved:
        wanted = {name for chunk in chunks for name in chunk}
        if set(saved.files) != wanted | {'questions', 'y', 'folds'}:
            raise ValueError(f'{tag}: prediction keys differ')
        n = len(saved['y'])
        k = 2 if tag.startswith('switch') else 7
        folds = saved['folds']
        question_shape, question_dtype = safe_npy_shape(pred_path, 'questions')
        if (folds.shape != (5, n) or question_shape != (n,) or
            question_dtype.kind not in ('O', 'U', 'S')):
            raise ValueError(f'{tag}: prediction population/folds differ')
        for name in wanted:
            values = saved[name]
            if values.shape != (5, n, k) or not np.isfinite(values).all():
                raise ValueError(f'{tag}: nonfinite or wrong prediction shape {name}')
    hashes = {}
    for name, path in sorted(actual.items()):
        r, f, chunk = expected[name]
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['binding']) != FROZEN:
                raise ValueError(f'{tag}/{name}: frozen fit binding differs')
            if set(saved.files) != {'binding', 'test', 'stats'} | {'lp:' + key for key in chunk}:
                raise ValueError(f'{tag}/{name}: model chunk keys differ')
            test = saved['test']
            if (test.ndim != 1 or len(np.unique(test)) != len(test) or
                not np.array_equal(np.sort(test), np.flatnonzero(folds[r] == f))):
                raise ValueError(f'{tag}/{name}: held-out fold differs')
            if not np.isfinite(saved['stats']).all():
                raise ValueError(f'{tag}/{name}: nonfinite fit stats')
            for key in chunk:
                values = saved['lp:' + key]
                if values.shape != (len(test), k) or not np.isfinite(values).all():
                    raise ValueError(f'{tag}/{name}: invalid {key} predictions')
        hashes[name] = file_sha(path)
    return {'tag': tag, 'cell_receipt_sha256': receipt['sha256'],
            'predictions_file_sha256': file_sha(pred_path),
            'n_pairs': n, 'n_fit_checkpoints': len(expected),
            'fit_file_sha256': hashes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=sorted(CONFIG), required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('full fit verification requires CPU Slurm')
    model, tags, schema, cell_schema, script = CONFIG[args.kind]
    root = ROOT / 'steering-v1/runs/resume-v1' / schema
    top = sealed(root / 'result.json')
    sidecar_sha = file_sha(script)
    if (top['schema'] != schema or top['frozen'] != FROZEN or
        top['driver_sha256'] != DRIVER or top['scheduler_sha256'] != sidecar_sha or
        set(top['models']) != {model} or set(top['models'][model]) != set(tags)):
        raise ValueError('complete sidecar receipt binding differs')
    cells = [verify_cell(root, model, tag, cell_schema, top, sidecar_sha) for tag in tags]
    body = {'schema': 'r3d-sidecar-fit-verification-v1',
            'kind': args.kind, 'validation_job_id': os.environ['SLURM_JOB_ID'],
            'source_job_id': top['job_id'], 'source_receipt_sha256': top['sha256'],
            'source_code_sha256': sidecar_sha, 'frozen_specification_sha256': FROZEN,
            'canonical_driver_sha256': DRIVER, 'model': model, 'cells': cells,
            'total_fit_checkpoints': sum(c['n_fit_checkpoints'] for c in cells),
            'scope': 'read-only sidecar validation before any canonical checkpoint copy'}
    value = {**body, 'sha256': digest(body)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        if json.loads(args.out.read_text()) != value:
            raise ValueError('existing verifier receipt differs')
    else:
        tmp = args.out.with_name(args.out.name + '.part-' + os.environ['SLURM_JOB_ID'])
        tmp.write_text(json.dumps(value, indent=1) + '\n')
        tmp.replace(args.out)
    print(json.dumps({'receipt': str(args.out), 'kind': args.kind,
                      'cells': len(cells), 'fit_checkpoints': body['total_fit_checkpoints']}), flush=True)


if __name__ == '__main__':
    main()

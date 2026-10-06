"""Read-only numeric parity of canonical and verified GPT destination fits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
CANON = ROOT / 'forum/tests/r3_context/anticipation/gpt'
SIDE = ROOT / 'steering-v1/runs/resume-v1/r3d-anticipation-gpt-dest-shard-v1/gpt'
VERIFY = REPO / 'report/experimental-resume-v1/R3D_SIDECAR_VERIFY_GPT_DEST_v2.json'
PRICE = REPO / 'report/experimental-resume-v1/R3D_GPT_OVERLAP_AUDIT_PRICE_v1.json'
OUT = REPO / 'report/experimental-resume-v1/R3D_GPT_OVERLAP_AUDIT_v1.json'
FROZEN = '62f1038003b18b09816635ada31d68136856477df335cbd54665942d159baa92'


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    x = json.loads(path.read_text())
    if x.get('sha256') != digest({k: v for k,v in x.items() if k != 'sha256'}):
        raise ValueError(f'changed receipt: {path}')
    return x


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.getuid() != os.stat(REPO).st_uid:
        raise ValueError('R3-D overlap audit requires owner CPU Slurm')
    price = sealed(PRICE)
    verified = sealed(VERIFY)
    if (price['driver_sha256'] != sha(__file__) or
        price['verifier_sha256'] != verified['sha256'] or
        verified['kind'] != 'gpt-dest' or verified['model'] != 'gpt' or
        verified['total_fit_checkpoints'] != 300 or
        verified['frozen_specification_sha256'] != FROZEN):
        raise ValueError('GPT sidecar source/verification differs')
    cells, overlap = {}, []
    for cell in verified['cells']:
        tag = cell['tag']
        names = cell['fit_file_sha256']
        if tag not in ('dest','dest-contained') or len(names) != 150:
            raise ValueError('unexpected GPT destination cell inventory')
        dst = CANON / tag
        source = SIDE / tag
        existing = sorted(dst.glob('[0-4]-[0-4]-*.npz')) if dst.exists() else []
        if any(p.name not in names for p in existing):
            raise ValueError(f'canonical {tag} has unexpected fit name')
        cells[tag] = {'sidecar_fits': len(names), 'canonical_fits_at_scan': len(existing)}
        for cp in existing:
            src = source / cp.name
            if sha(src) != names[cp.name]:
                raise ValueError(f'verified source changed: {src}')
            with np.load(src, allow_pickle=False) as a, np.load(cp, allow_pickle=False) as b:
                if set(a.files) != set(b.files) or str(b['binding']) != FROZEN or not np.array_equal(a['test'],b['test']):
                    raise ValueError(f'canonical binding, keys or fold differs: {cp}')
                deltas = {}
                exact = True
                for key in a.files:
                    if key in ('binding','test'):
                        continue
                    av, bv = a[key], b[key]
                    if av.shape != bv.shape or not np.isfinite(av).all() or not np.isfinite(bv).all():
                        raise ValueError(f'nonfinite or shape mismatch: {cp}/{key}')
                    diff = float(np.max(np.abs(av.astype(float)-bv.astype(float)))) if av.size else 0.
                    deltas[key] = diff
                    exact &= bool(np.array_equal(av,bv))
            overlap.append({'cell': tag, 'name': cp.name, 'canonical_sha256': sha(cp),
                            'sidecar_sha256': names[cp.name], 'numeric_exact': exact,
                            'max_abs_delta': max(deltas.values(), default=0.)})
    body = {'schema': 'r3d-gpt-destination-overlap-audit-v1',
            'status': 'PASS_BINDING_AND_FINITE', 'job_id': os.environ['SLURM_JOB_ID'],
            'price_sha256': price['sha256'], 'verifier_sha256': verified['sha256'],
            'frozen_specification_sha256': FROZEN, 'cells': cells,
            'n_overlaps_at_scan': len(overlap),
            'n_numeric_exact': sum(x['numeric_exact'] for x in overlap),
            'max_abs_delta': max((x['max_abs_delta'] for x in overlap), default=0.),
            'overlaps': overlap,
            'scope': 'read-only canonical fit checkpoint snapshot; canonical writer may add fits after scan; no copy or stop performed'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing overlap audit differs')
    else:
        temp = OUT.with_name(OUT.name+'.part-'+os.environ['SLURM_JOB_ID'])
        temp.write_text(json.dumps(value,indent=1)+'\n')
        temp.replace(OUT)
    print(json.dumps({'receipt': str(OUT), 'overlaps': len(overlap),
                      'numeric_exact': body['n_numeric_exact'],
                      'max_abs_delta': body['max_abs_delta']}),flush=True)


if __name__ == '__main__':
    main()

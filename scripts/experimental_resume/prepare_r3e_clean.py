"""Outcome-blind clean GPT A and family-fold preflight for registered R3-E/B4."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import pandas as pd

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
FAMILY = REPO / 'report/experimental-resume-v1/family-freeze.json'
SPLIT = ROOT / 'steering-v1/manifests/split-v1.json'
A = ROOT / 'dynamics-routing/results/B1/tokens/attempts.parquet'
B = ROOT / 'v3_analysis/results-r2/gpt/B/attempts.parquet'
PROV = ROOT / 'dynamics-routing/results/B1/tokens/provenance.json'
OUT = REPO / 'report/experimental-resume-v1/R3E_CLEAN_PREFLIGHT_v1.json'


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
        raise ValueError(f'invalid input seal: {path}')
    return x


def folds(ids, family, n, seed):
    fam = [family[q] for q in ids]
    ordered = sorted(set(fam), key=lambda f: digest(f'forum-v1|{seed}|{f}'))
    by = {f: i % n for i, f in enumerate(ordered)}
    return np.asarray([by[f] for f in fam], dtype=np.int8)


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('R3-E clean preflight requires CPU Slurm')
    family = sealed(FAMILY)
    split = sealed(SPLIT)
    qf = {q: f for f, qs in family['new_parent_pools']['families'].items() for q in qs}
    sp = {f"{r['dataset']}|{r['source_problem_id']}": r['split'] for r in split['questions']}
    if len(qf) != len(set(qf)) or family['inputs']['split']['sha256'] != sha(SPLIT):
        raise ValueError('duplicate family map or changed split input')
    cols = ['attempt_id', 'question']
    a = pd.read_parquet(A, columns=cols)
    b = pd.read_parquet(B, columns=['question'])
    if len(a) != len(set(a['attempt_id'])):
        raise ValueError('duplicate GPT A attempt ID')
    ids = a['question'].astype(str).tolist()
    if not set(ids) <= set(sp) or not set(ids) <= set(qf):
        raise ValueError('GPT A question missing split or family assignment')
    reserved = set(b['question'].astype(str))
    exact = np.asarray([sp[q] in ('dev', 'tune') and q not in reserved for q in ids], bool)
    confirm_families = {qf[q] for q, label in sp.items() if label == 'confirm' and q in qf}
    sensitivity = np.asarray([keep and qf[q] not in confirm_families for keep, q in zip(exact, ids)], bool)
    rows = {}
    for name, keep in [('exact_id_clean', exact), ('exclude_confirm_connected_families', sensitivity)]:
        selected = [q for q, ok in zip(ids, keep) if ok]
        if not selected:
            raise ValueError(f'empty clean R3-E population: {name}')
        if any(sp[q] not in ('dev', 'tune') or q in reserved for q in selected):
            raise ValueError('confirm or reserved B question in clean R3-E population')
        outer = [folds(selected, qf, 5, 20260929 + r) for r in range(5)]
        for f in outer:
            for fam in set(qf[q] for q in selected):
                loc = [f[i] for i, q in enumerate(selected) if qf[q] == fam]
                if len(set(loc)) != 1:
                    raise ValueError('duplicate family straddles outer folds')
        training = []
        for r, f in enumerate(outer):
            for k in range(5):
                train = np.flatnonzero(f != k)
                training.append(hashlib.sha256(np.asarray(train, np.int64).tobytes()).hexdigest())
                inner = folds([selected[i] for i in train], qf, 3, 20260929 + r * 10 + k)
                for j in range(3):
                    sub = train[inner != j]
                    training.append(hashlib.sha256(np.asarray(sub, np.int64).tobytes()).hexdigest())
        if len(training) != 100 or len(set(training)) != 100:
            raise ValueError('clean R3-E must have 100 distinct nested training sets')
        rows[name] = {'attempts': len(selected), 'questions': len(set(selected)),
                      'families': len({qf[q] for q in selected}),
                      'attempt_ids_sha256': digest(a.loc[keep, 'attempt_id'].astype(str).tolist()),
                      'ordered_questions_sha256': digest(selected),
                      'outer_fold_sha256': digest([f.tolist() for f in outer]),
                      'nested_training_set_keys_sha256': digest(training),
                      'nested_training_sets': 100}
    body = {'schema': 'r3e-clean-outcome-blind-preflight-v1',
            'status': 'INPUTS_AND_FOLDS_READY_FEATURE_FIT_NOT_STARTED',
            'job_id': os.environ['SLURM_JOB_ID'],
            'source_sha256': sha(__file__),
            'inputs': {str(p): sha(p) for p in (FAMILY, SPLIT, A, B, PROV)},
            'family_seal': family['sha256'], 'split_seal': split['sha256'],
            'original_gpt_A_attempts': len(a), 'reserved_B_questions': len(reserved),
            'confirm_connected_families': len(confirm_families), 'populations': rows,
            'design': {'outer': 5, 'inner': 3, 'repeats': 5,
                       'seeds': list(range(20260929, 20260934)),
                       'group': 'frozen duplicate family', 'outcomes_read': False},
            'interpretation': 'Main is exact-ID clean; family-connected exclusion is sensitivity. Existing historical B1/B4 fits remain historical.'}
    value = {**body, 'sha256': digest(body)}
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing R3-E preflight differs')
    else:
        OUT.write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps({'out': str(OUT), 'sha256': value['sha256'], 'populations': rows}), flush=True)


if __name__ == '__main__':
    main()

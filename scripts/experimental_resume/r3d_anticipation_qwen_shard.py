"""Isolated Qwen scheduling shard for the frozen R3-D anticipation estimator.

The statistical functions come from the sealed R3-D driver. This file only
selects the Qwen cells and writes their checkpoints to a separate directory.
No file in the canonical anticipation checkpoint tree is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
OUT = ROOT / 'forum/tests/r3_context'
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DRIVER = OUT / 'code/r3d_anticipation.py'
FROZEN = OUT / 'FROZEN.anticipation.json'
SHARD = ROOT / 'steering-v1/runs/resume-v1/r3d-anticipation-qwen-shard-v1'
DRIVER_SHA = '4f1e0e1aa57126eb3757a554560992314ffabbaf65310e007be40cb638f717ca'
FROZEN_SHA = '62f1038003b18b09816635ada31d68136856477df335cbd54665942d159baa92'


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def preflight() -> dict:
    if file_sha(DRIVER) != DRIVER_SHA:
        raise ValueError('frozen driver changed')
    frozen = json.loads(FROZEN.read_text())
    if frozen.get('sha256') != FROZEN_SHA or digest({k: v for k, v in frozen.items() if k != 'sha256'}) != FROZEN_SHA:
        raise ValueError('frozen anticipation specification changed')
    # Verify all provenance inputs, including other-model files: the shard
    # remains bound to the same complete parent study, folds, and controls.
    return frozen


def load_driver():
    sys.path.insert(0, str(OUT / 'code'))
    import r3d_anticipation as module
    if Path(module.__file__).resolve() != DRIVER.resolve():
        raise RuntimeError('frozen driver import resolved to a different file')
    return module


def run(workers: int) -> None:
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('fit shard requires CPU Slurm')
    frozen = preflight()
    canonical_qwen = OUT / 'anticipation/qwen36'
    if any(canonical_qwen.rglob('[0-4]-[0-4]-*.npz')):
        print(json.dumps({'status': 'SKIP_CANONICAL_QWEN_ALREADY_STARTED',
                          'job_id': os.environ['SLURM_JOB_ID']}), flush=True)
        return
    shard_sha = file_sha(Path(__file__))
    for filename, expected in frozen['inputs'].items():
        if file_sha(Path(filename)) != expected:
            raise ValueError(f'parent input changed: {filename}')
    for filename, expected in frozen['code'].items():
        if file_sha(Path(filename)) != expected:
            raise ValueError(f'parent code changed: {filename}')
    A = load_driver()
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from r3d_readout import family_folds, verified
    from dynrt import common as D, cv as C
    D.RESULTS = ROOT / 'dynamics-routing/results'
    from dynrt import data as Data, d3a_pairs as P, d3a_text as T, features as F, d3a_softmax as Softmax
    from rkin import common as RK, k2 as K
    RK.DATA = ROOT / 'reasoning-kinematics/rk2/data'
    K.DATA = RK.DATA
    K.K2DATA = RK.DATA / 'k2'
    K.OUTER = 5
    family = verified(REPO / 'report/experimental-resume-v1/family-freeze.json')
    qf = {q: f for f, questions in family['new_parent_pools']['families'].items() for q in questions}
    split = D.load_split()
    folds = lambda qs, n, seed: family_folds(qs, n, seed if seed >= A.SEED else A.SEED * 1000 + seed, qf)
    C.question_folds = folds
    A.configure_blocks(K, F, C, Softmax, T, folds)
    model = 'qwen36'
    pairs_all = pd.read_parquet(K.K2DATA / model / 'pairs.parquet')
    keep = np.array([split[q] in ('dev', 'tune') for q in pairs_all['question']])
    reserved = set(pd.read_parquet(D.V3R2 / model / 'B/attempts.parquet', columns=['question'])['question'])
    keep &= ~pairs_all['question'].isin(reserved).to_numpy()
    rows = np.flatnonzero(keep)
    pairs = pairs_all.iloc[rows].reset_index(drop=True)
    cohort = Data.load_cohort(model, 'A')
    base, text = A.augment_baseline(model, pairs, cohort, P, T)
    lex = sparse.load_npz(K.K2DATA / model / 'lex_counts.npz').tocsr()[rows]
    with np.load(K.K2DATA / model / 'kin.npz') as z:
        kin = {'k16': z['k16'][rows], 'k32': z['k32'][rows]}
    prov = json.loads((K.K2DATA / model / 'provenance.json').read_text())
    weak_text = sparse.load_npz(prov['text_counts']).tocsr()[rows]
    a, b = pairs['a'].to_numpy(int), pairs['b'].to_numpy(int)
    all_targets = [('next', np.arange(len(pairs)), b, 7),
                   ('switch', np.arange(len(pairs)), (b != a).astype(int), 2),
                   ('dest', np.flatnonzero(b != a), b, 7)]
    results = {}
    SHARD.mkdir(parents=True, exist_ok=True)
    for target, response_rows, response_y, n_class in all_targets:
        for contained in (False, True):
            selected = response_rows[pairs['n_tokens'].to_numpy()[response_rows] >= 32] if contained else response_rows
            tag = target + ('-contained' if contained else '')
            groups = pairs['question'].to_numpy()[selected]
            _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
            d = K.KData(model, tag, n_class, response_y[selected], a[selected], groups, 1 / cnt[inv],
                        dense={'base': base[selected], 'weak_base': base[selected, :12],
                               **{name: value[selected] for name, value in kin.items()},
                               'oracle_class': np.eye(7)[a[selected]]},
                        text=text[selected], lex=lex[selected],
                        outer_folds=np.stack([folds(groups, 5, s) for s in range(A.SEED, A.SEED + 5)]),
                        s_text=10., s_lex=10.)
            d.text_weak = weak_text[selected]
            if target == 'next':
                index = {(int(cohort.table.attempt[i]), int(s.sentence_index)): i
                         for i, s in enumerate(cohort.sent.itertuples())}
                source_rows = [index[int(row.att), int(row.sentence_index)]
                               for row in pairs.iloc[selected].itertuples()]
                hist = cohort.hist
                pos = np.searchsorted(hist['last16_rows'], source_rows)
                if not np.array_equal(hist['last16_rows'][pos], source_rows):
                    raise ValueError('A1 last16 histogram alignment failed')
                h16 = hist['last16'][pos].reshape(len(selected), -1).astype(np.float32)
                ha = np.full_like(h16, np.nan)
                d.has_antic = np.zeros(len(selected), bool)
                d.composition = (cohort, pairs['att'].to_numpy()[selected], h16, ha)
                targets = ['composition16']
            else:
                targets = ['k16', 'k32']
            models = {'base': ['base', 'prefix_class'],
                      'weak': ['weak_sparse', 'weak_base', 'prefix_class']}
            for route in targets:
                models[route] = ['base', 'prefix_class', route]
                models['oracle_' + route] = ['base', 'oracle_class', route]
                for draw in range(20):
                    models[f'noise_{route}_{draw}'] = ['base', 'prefix_class', f'noise_{route}_{draw}']
            cell = SHARD / model / tag
            predictions, diagnostics = A.run_cv(K, d, models, workers, FROZEN_SHA, cell)
            result = {'status': 'CLEAN', 'questions': len(set(groups)), 'pairs': len(selected),
                      'short_sentence_exclusions': len(response_rows) - len(selected),
                      'routing': A.summarize(d, predictions, targets, qf),
                      'oracle_current_class': 'separate saved oracle_* predictions; never an online trigger',
                      'fit_converged_fraction': float(diagnostics[:, 1].mean()),
                      'A1_antic_replication': 'INCOMPLETE existing cache absent' if target == 'next' else None}
            receipt = {'schema': 'r3d-anticipation-qwen-shard-cell-v1', 'frozen': FROZEN_SHA,
                       'driver_sha256': DRIVER_SHA, 'scheduler_sha256': shard_sha,
                       'job_id': os.environ['SLURM_JOB_ID'], 'model': model, 'tag': tag,
                       'result': result}
            (cell / 'result.json').write_text(json.dumps(A.seal(receipt), indent=1) + '\n')
            results[tag] = result
            print(json.dumps({'status': 'CELL_COMPLETE', 'model': model, 'tag': tag,
                              'checkpoints': len(list(cell.glob('[0-4]-[0-4]-*.npz')))}), flush=True)
    output = {'schema': 'r3d-anticipation-qwen-shard-v1', 'frozen': FROZEN_SHA,
              'driver_sha256': DRIVER_SHA, 'scheduler_sha256': shard_sha,
              'job_id': os.environ['SLURM_JOB_ID'], 'models': {model: results}}
    (SHARD / 'result.json').write_text(json.dumps(A.seal(output), indent=1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps({'status': 'READY', 'frozen': preflight()['sha256'],
                          'driver': DRIVER_SHA, 'output': str(SHARD)}))
    else:
        run(args.workers)

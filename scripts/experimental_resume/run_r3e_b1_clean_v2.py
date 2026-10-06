"""Clean GPT A B1 refit, using frozen duplicate families and existing token caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24')
REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
CODE = ROOT / 'forum/tests/r3_context/code'
sys.path.insert(0, str(CODE))
from dynrt import d3b_b1 as B  # noqa: E402
from dynrt.d3b_tokens import load_tokens  # noqa: E402
from dynrt.common import load_split  # noqa: E402
from dynrt.d3b_util import clean  # noqa: E402

FAMILY = REPO / 'report/experimental-resume-v1/family-freeze.json'
PREFLIGHT = REPO / 'report/experimental-resume-v1/R3E_CLEAN_PREFLIGHT_v1.json'
TOKEN_CACHE = ROOT / 'dynamics-routing/results/B1/tokens'
OUT = ROOT / 'steering-v1/runs/resume-v1/r3e-b1-clean-v2/result.json'


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
        raise ValueError(f'invalid seal {path}')
    return x


def family_folds(groups, n, seed, qf):
    families = [qf[str(q)] for q in groups]
    order = sorted(set(families), key=lambda f: digest(f'forum-v1|{seed}|{f}'))
    fnum = {f: i % n for i, f in enumerate(order)}
    return np.asarray([fnum[f] for f in families], int)


def family_interval(gains, groups, qf):
    g = np.asarray(gains, float)
    if len(g) != len(groups) or not np.isfinite(g).all():
        raise ValueError('clean B1 question gains are missing or nonfinite')
    family = np.asarray([qf[q] for q in groups])
    names, inv = np.unique(family, return_inverse=True)
    sums = np.bincount(inv, weights=g)
    counts = np.bincount(inv)
    rng = np.random.default_rng(20260929)
    picks = rng.integers(0, len(names), size=(5000, len(names)))
    boots = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    signs = rng.choice((-1, 1), size=(10000, len(names)))
    flips = (signs * sums).sum(axis=1) / counts.sum()
    estimate = float(g.mean())
    return {'estimate_nats_per_token_pair': estimate,
            'family_cluster_ci95': np.quantile(boots, [.025, .975]).tolist(),
            'family_sign_flip_p_one_sided': float((1 + np.count_nonzero(flips >= estimate)) / (1 + len(flips))),
            'sign_flip_assumption': 'family-level exchangeability of observational held-out gain signs',
            'bootstrap_replicates': 5000, 'sign_flip_replicates': 10000,
            'families': int(len(names)), 'questions': len(groups)}


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('clean B1 fit requires CPU Slurm')
    family = sealed(FAMILY)
    preflight = sealed(PREFLIGHT)
    qf = {q: f for f, qs in family['new_parent_pools']['families'].items() for q in qs}
    split = load_split()
    if preflight['family_seal'] != family['sha256']:
        raise ValueError('R3-E family preflight differs')
    if sha(TOKEN_CACHE / 'provenance.json') != preflight['inputs'][str(TOKEN_CACHE / 'provenance.json')]:
        raise ValueError('historical GPT A token-cache provenance differs from clean preflight')
    tk = load_tokens(TOKEN_CACHE)
    att = tk['attempts']
    import pandas as pd
    b = set(pd.read_parquet(ROOT / 'v3_analysis/results-r2/gpt/B/attempts.parquet', columns=['question'])['question'])
    keep = np.asarray([split[q] in ('dev', 'tune') and q not in b for q in att['question']], bool)
    ids = att.loc[keep, 'attempt_id'].astype(str).tolist()
    if (len(ids) != preflight['populations']['exact_id_clean']['attempts'] or
        digest(ids) != preflight['populations']['exact_id_clean']['attempt_ids_sha256']):
        raise ValueError('clean B1 universe differs from outcome-blind preflight')
    ts, questions = B.make_tokenset(tk, keep, experts=32)
    if len(questions) != preflight['populations']['exact_id_clean']['questions']:
        raise ValueError('clean B1 question count differs')
    B.question_folds = lambda groups, n, seed: family_folds(groups, n, 20260929 + seed, qf)
    pairs = B.layer_pairs(24)
    result = B.run_b1(ts, questions, pairs, n_folds=5, repeats=5, n_perm=20,
                      n_classperm=20, workers=int(os.environ['SLURM_CPUS_PER_TASK']),
                      n_boot=5000, seed=20260929, tag='r3e-clean-gptA')
    interval = family_interval(result['question_gain'], questions, qf)
    body = {'schema': 'r3e-clean-gpt-b1-v1', 'status': 'COMPLETE',
            'job_id': os.environ['SLURM_JOB_ID'], 'source_sha256': sha(__file__),
            'family_seal': family['sha256'], 'preflight_seal': preflight['sha256'],
            'input_tokens_sha256': tk['provenance']['tokens_sha256'],
            'token_cache_path': str(TOKEN_CACHE),
            'frozen_estimator_code_sha256': {n: sha(CODE / 'dynrt' / n) for n in
                                             ('d3b_b1.py', 'd3b_tokens.py', 'cv.py', 'data.py', 'common.py')},
            'population': {'questions': len(questions), 'families': interval['families'],
                           'labelled_tokens': len(ts.cls), 'confirm_questions': 0,
                           'reserved_B_questions': 0},
            'design': {'layer_pairs': len(pairs), 'gaps': [1, 4], 'outer_folds': 5,
                       'repeats': 5, 'seeds': list(range(20260929, 20260934)),
                       'group': 'frozen duplicate family', 'permutation_nulls': 20,
                       'within_token_class_shuffles': 20, 'outcomes_read': False},
            'family_clustered': interval, 'result': clean(result),
            'interpretation': 'Clean observational B1 refit for registered R3-E, not correctness or causal steering.'}
    value = {**body, 'sha256': digest(body)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        if json.loads(OUT.read_text()) != value:
            raise ValueError('existing clean B1 result differs')
    else:
        temp = OUT.with_name('result.json.part-' + os.environ['SLURM_JOB_ID'])
        temp.write_text(json.dumps(value, indent=1) + '\n')
        temp.replace(OUT)
    print(json.dumps({'out': str(OUT), 'sha256': value['sha256'],
                      'questions': len(questions), 'families': interval['families'],
                      'gain': interval['estimate_nats_per_token_pair'],
                      'ci95': interval['family_cluster_ci95']}), flush=True)


if __name__ == '__main__':
    main()

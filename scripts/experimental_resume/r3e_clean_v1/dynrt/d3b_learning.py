"""D3b supplementary (not registered): B1 learning curve on the dense pilot.

B1 uses no adjacency, so the sparse-vs-dense difference of G_joint is a training-size effect: a random ~7% of the
sentences of the TRAINING questions gives noisier class-specific pair tables. Here the tables (and the IPF null) are
fitted from a random fraction of the training questions' sentences while the held-out questions are always scored on
all their tokens (same folds as the dense reference, repeat 0). Read next to the registered sparse simulation to
separate the training-size effect from anything else.

  python -m dynrt.d3b_learning run [--fractions 0.07 0.14 0.25 0.5 1.0 --draws 3] -> results/pilot/learning_curve.json
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from . import d3b_b1 as B
from .common import RESULTS
from .d3b_sparse import SEED, draw_mask
from .d3b_tokens import OUT_PILOT, PILOT_SHAPE, load_tokens
from .d3b_util import code_hashes, log, write_json

OUT = RESULTS / "pilot"
_STATE: dict = {}


def run_pair_train(index: int) -> np.ndarray:
    """Held-out per-question gains [n_q] of one layer pair; tables fitted on the masked training tokens."""
    st = _STATE
    i, j, _ = st["pairs"][index]
    ts: B.TokenSet = st["ts"]
    e = ts.experts
    t = B.pair_tables(ts.ids[:, i, :], ts.ids[:, j, :], ts.cls, ts.q, ts.n_q, e)
    m = st["train_mask"]
    if m is None:
        t_tr = t
    else:
        t_tr = B.pair_tables(ts.ids[m][:, i, :], ts.ids[m][:, j, :], ts.cls[m], ts.q[m], ts.n_q, e)
    total = t_tr.sum(0, dtype=np.float64)
    k2 = float(ts.k * ts.k)
    gains = np.full(ts.n_q, np.nan)
    for f in range(st["n_folds"]):
        te = st["folds"] == f
        n_train = ((total - t_tr[te].sum(0, dtype=np.float64)) / k2).reshape(B.NC, e, e)
        delta, ok, _, _ = B.fold_delta(n_train)
        gains[te] = B.score_questions(t[te], delta, ok)[0]
    return gains


def gains_for(ts: B.TokenSet, folds: np.ndarray, pairs: list, train_mask, workers: int) -> np.ndarray:
    _STATE.update(ts=ts, folds=folds, n_folds=8, pairs=pairs, train_mask=train_mask)
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(run_pair_train, range(len(pairs)), chunksize=1)
    return np.nanmean(np.stack(res), axis=0)


def run(a) -> int:
    t0 = time.time()
    tk = load_tokens(OUT_PILOT)
    ts, questions = B.make_tokenset(tk, np.ones(len(tk["attempts"]), bool), PILOT_SHAPE["num_experts"])
    pairs = B.layer_pairs(PILOT_SHAPE["num_layers"])
    if a.max_pairs:
        pairs = pairs[::max(1, len(pairs) // a.max_pairs)][:a.max_pairs]
    folds = B.make_folds(questions, 8, 1)[0]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    sent = tk["sentences"]
    rows = []
    for frac in a.fractions:
        draws = 1 if frac >= 1.0 else a.draws
        for d in range(draws):
            if frac >= 1.0:
                mask = None
            else:
                mask = draw_mask(sent, 1000 + d, fraction=frac, min_labels=10)[tk["sent"]]
            qg = gains_for(ts, folds, pairs, mask, workers)
            rows.append(dict(fraction=frac, draw=d, G=float(np.nanmean(qg)),
                             training_tokens=int(len(ts.cls) if mask is None else mask.sum())))
            log(f"fraction {frac}: draw {d}: G {rows[-1]['G']:.6f} ({time.time() - t0:.0f}s)")
    curve = {}
    for r in rows:
        curve.setdefault(str(r["fraction"]), []).append(r["G"])
    out = dict(design=dict(fractions=list(a.fractions), draws=a.draws, seed=SEED, layer_pairs=len(pairs),
                           note="supplementary, not registered: training tables from a random fraction of the "
                                "sentences of the training questions; held-out questions scored on all tokens"),
               curve={k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                              draws=v) for k, v in curve.items()},
               rows=rows, code_sha256=code_hashes(), tokens_sha256=tk["provenance"]["tokens_sha256"],
               slurm_job=os.environ.get("SLURM_JOB_ID"), seconds=time.time() - t0,
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(OUT / f"learning_curve{('_' + a.tag) if a.tag else ''}.json", out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--fractions", type=float, nargs="+", default=[0.07, 0.14, 0.25, 0.5, 1.0])
    s.add_argument("--draws", type=int, default=3)
    s.add_argument("--max-pairs", type=int)
    s.add_argument("--tag", default="")
    s.set_defaults(fn=run)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

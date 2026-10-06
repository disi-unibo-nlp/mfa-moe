"""D3b supplementary (not registered): A2 sparse-vs-dense sweep over the label fraction.

The registered 7% simulation leaves ~99 a->b events over ~4 contributing questions of the 64 pilot questions, so its
A2 intervals are uninformative about the sampling design. Here the same estimator and folds are run at larger label
fractions (uniform per-trace masks as in d3b_sparse) to show when bias and coverage stabilise.

  python -m dynrt.d3b_sparse_sweep run [--fractions 0.07 0.14 0.28 0.5 --draws 20] -> results/pilot/sparse_sweep.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from . import d3b_a2 as A
from . import d3b_b1 as B
from .common import RESULTS
from .d3b_sparse import compare, covers, draw_mask, pair_statistics
from .d3b_tokens import OUT_PILOT, PILOT_SHAPE, load_tokens
from .d3b_util import code_hashes, log, write_json

OUT = RESULTS / "pilot"


def run(a) -> int:
    t0 = time.time()
    tk = load_tokens(OUT_PILOT)
    p = A.make_pilot(tk, PILOT_SHAPE["num_experts"])
    sent, att = p.sentences, p.attempts
    keep = np.ones(len(att), bool)
    h = A.sentence_histograms(p.ids, p.sent, len(sent), p.experts)
    edges = A.length_edges(sent)
    questions = sorted(set(att["question"]))
    folds_q = B.make_folds(questions, 8, 1)[0]
    dense = A.a2_run(p, h, keep, None, folds_q=folds_q, edges=edges)
    lag0 = A.LAGS.index(0)
    stats = {"pre": lambda r: (r["pre_sum"], r["pre_cnt"]),
             "pre_lag0": lambda r: (r["sums"][:, lag0, :].mean(-1), r["cnt"][:, lag0])}
    ref = {}
    for k, f in stats.items():
        m, w = A.qmean(*f(dense))
        ref[k] = dict(estimate=float((m * w).sum() / w.sum()), sd=float(np.std(m[w > 0], ddof=1)))
    log(f"dense reference {ref} ({time.time() - t0:.0f}s)")
    out = dict(design=dict(fractions=list(a.fractions), draws=a.draws, min_labels=10,
                           note="supplementary, not registered; same folds/estimator as the registered simulation"),
               dense=ref, by_fraction={}, code_sha256=code_hashes(), tokens_sha256=tk["provenance"]["tokens_sha256"],
               slurm_job=os.environ.get("SLURM_JOB_ID"))
    for frac in a.fractions:
        est = {k: [] for k in stats}
        cov = {k: [] for k in stats}
        nq, ev = [], []
        for d in range(a.draws):
            mask = draw_mask(sent, 2000 + d, fraction=frac, min_labels=10)
            res = A.a2_run(p, h, keep, mask, folds_q=folds_q, edges=edges)
            ps = pair_statistics(sent, mask)
            ev.append(ps["events"])
            for k, f in stats.items():
                sc = A.scalar_ci(*f(res), 500)
                est[k].append(sc["estimate"])
                cov[k].append(covers(sc["ci_lo"], sc["ci_hi"], ref[k]["estimate"]))
                if k == "pre":
                    nq.append(sc["n_questions"])
        out["by_fraction"][str(frac)] = dict(
            mean_events=float(np.mean(ev)), mean_contributing_questions=float(np.mean(nq)),
            **{k: compare(est[k], ref[k]["estimate"], ref[k]["sd"], cov[k]) for k in stats})
        c = out["by_fraction"][str(frac)]["pre"]
        log(f"fraction {frac}: events {np.mean(ev):.0f}, questions {np.mean(nq):.1f}; pre bias/SD "
            f"{c['bias_over_sd']:.3f}, coverage {c['coverage']:.2f} ({time.time() - t0:.0f}s)")
    out["seconds"] = time.time() - t0
    write_json(OUT / "sparse_sweep.json", out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--fractions", type=float, nargs="+", default=[0.07, 0.14, 0.28, 0.5])
    s.add_argument("--draws", type=int, default=20)
    s.set_defaults(fn=run)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

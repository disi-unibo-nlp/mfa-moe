"""D3b sparse-mask simulation on the dense pilot (measurement feasibility; no outcome is read).

The campaign labels ~7% of the sentences of every trace, drawn uniformly without replacement (sample_stratified),
so adjacent labelled pairs arise only by chance: P(both s-1 and s labelled) = m(m-1) / (N(N-1)) for m of N sentences.
Each draw masks the dense labels that way (m = round(0.07 N), at least 10 labels as in the Qwen3.6 campaign
minimum), and B1 (tokens of the labelled sentences only; profiles/tables from the labelled sentences of the
training questions) and A2 (adjacent labelled pairs, class runs and profiles from the labelled sentences only;
unlabelled sentences break runs) are recomputed with the SAME question folds as the dense reference (repeat 0).
Per statistic: bias = mean(sparse - dense) relative to the between-question SD of the dense question-level
values, and coverage = share of draws whose 95% question-bootstrap interval contains the dense estimate.
Gate (registered): |bias| < 0.2 SD and coverage >= 90%.

  python -m dynrt.d3b_sparse run [--draws 50 --max-pairs N --tag T] -> results/pilot/sparse_sim.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from . import d3b_a2 as A
from . import d3b_b1 as B
from .common import RESULTS
from .d3b_tokens import OUT_PILOT, PILOT_SHAPE, load_tokens
from .d3b_util import code_hashes, log, write_json

FRACTION = 0.07
MIN_LABELS = 10
SEED = 20260929
BIAS_GATE = 0.2
COVERAGE_GATE = 0.90
OUT = RESULTS / "pilot"


def draw_mask(sent: pd.DataFrame, draw: int, seed: int = SEED, fraction: float = FRACTION,
              min_labels: int = MIN_LABELS) -> np.ndarray:
    """Bool mask over sentence rows: per trace m = round(fraction N) (>= min_labels, <= N) uniform without replacement."""
    att = sent["att"].to_numpy()
    mask = np.zeros(len(sent), bool)
    starts = np.flatnonzero(np.r_[True, att[1:] != att[:-1]])
    ends = np.r_[starts[1:], len(att)]
    for a, (s, e) in enumerate(zip(starts, ends)):
        n = e - s
        m = int(min(n, max(min_labels, round(fraction * n))))
        rng = np.random.default_rng([seed, draw, int(att[s])])
        mask[s + rng.choice(n, size=m, replace=False)] = True
    return mask


def pair_statistics(sent: pd.DataFrame, mask: np.ndarray) -> dict:
    """Adjacent labelled pairs realised by a mask (diagnostic of the sampled pair structure)."""
    bd = A.build_boundaries(sent, mask)
    return dict(labelled=int(mask.sum()), adjacent_pairs=int(len(bd.row)), events=int((bd.a != bd.b).sum()))


def compare(sparse: list[float], dense: float, sd_between: float, covers: list[bool]) -> dict:
    """Bias (relative to the between-question SD), spread and coverage over the draws."""
    v = np.asarray(sparse, float)
    ok = np.isfinite(v)
    bias = float(np.mean(v[ok] - dense)) if ok.any() else float("nan")
    cov = float(np.mean(np.asarray(covers)[ok])) if ok.any() else float("nan")
    rel = bias / sd_between if sd_between > 0 else float("nan")
    return dict(dense=float(dense), sparse_mean=float(np.mean(v[ok])) if ok.any() else None,
                sparse_sd=float(np.std(v[ok], ddof=1)) if ok.sum() > 1 else None, bias=bias,
                between_question_sd=float(sd_between), bias_over_sd=rel, coverage=cov,
                n_draws=int(len(v)), n_finite=int(ok.sum()),
                passes=bool(np.isfinite(rel) and abs(rel) < BIAS_GATE and cov >= COVERAGE_GATE))


def covers(ci_lo: float, ci_hi: float, dense: float) -> bool:
    return bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo <= dense <= ci_hi)


def run(a) -> int:
    t0 = time.time()
    tk = load_tokens(OUT_PILOT)
    p = A.make_pilot(tk, PILOT_SHAPE["num_experts"])
    sent, att = p.sentences, p.attempts
    all_keep = np.ones(len(att), bool)
    n_sent = len(sent)
    h = A.sentence_histograms(p.ids, p.sent, n_sent, p.experts)
    edges = A.length_edges(sent)
    pairs = B.layer_pairs(PILOT_SHAPE["num_layers"])
    if a.max_pairs:
        pairs = pairs[::max(1, len(pairs) // a.max_pairs)][:a.max_pairs]
    ts, questions = B.make_tokenset(tk, all_keep, PILOT_SHAPE["num_experts"])
    folds = B.make_folds(questions, 8, 1)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    dense_b1 = B.run_pairs(ts, folds, 8, pairs, workers)
    qg_dense = B.question_gain(dense_b1["gains"])
    g_dense = float(np.nanmean(qg_dense))
    sd_b1 = float(np.nanstd(qg_dense, ddof=1))
    log(f"dense B1 (repeat 0): G {g_dense:.6f}, between-question SD {sd_b1:.6f} ({time.time() - t0:.0f}s)")
    folds_q = folds[0]
    dense_a2 = A.a2_run(p, h, all_keep, None, folds_q=folds_q, edges=edges)
    lag0 = A.LAGS.index(0)
    stats = {
        "pre": (dense_a2["pre_sum"], dense_a2["pre_cnt"]),
        "pre_lag0": (dense_a2["sums"][:, lag0, :].mean(-1), dense_a2["cnt"][:, lag0]),
    }
    dense_scalar = {}
    for k, (s, c) in stats.items():
        m, w = A.qmean(s, c)
        dense_scalar[k] = dict(estimate=float((m * w).sum() / w.sum()),
                               sd=float(np.std(m[w > 0], ddof=1)), n_q=int(w.sum()))
        log(f"dense A2 {k}: {dense_scalar[k]}")
    per_draw = []
    series = {"B1_G_joint": [], "A2_pre": [], "A2_pre_lag0": []}
    cover = {k: [] for k in series}
    for d in range(a.draws):
        td = time.time()
        mask = draw_mask(sent, d)
        tok_mask = mask[tk["sent"]]
        sp = B.run_pairs(ts.select(tok_mask), folds[:1], 8, pairs, workers)
        qg = B.question_gain(sp["gains"])
        ci = B.boot_ci(qg, 500, seed=d)
        series["B1_G_joint"].append(ci["mean"])
        cover["B1_G_joint"].append(covers(ci["ci_lo"], ci["ci_hi"], g_dense))
        res = A.a2_run(p, h, all_keep, mask, folds_q=folds_q, edges=edges)
        row = dict(draw=d, **pair_statistics(sent, mask), b1_G=ci["mean"], b1_questions=ci["n_questions"])
        for key, (s, c) in {"A2_pre": (res["pre_sum"], res["pre_cnt"]),
                            "A2_pre_lag0": (res["sums"][:, lag0, :].mean(-1), res["cnt"][:, lag0])}.items():
            sc = A.scalar_ci(s, c, 500)
            series[key].append(sc["estimate"])
            dk = dense_scalar["pre" if key == "A2_pre" else "pre_lag0"]["estimate"]
            cover[key].append(covers(sc["ci_lo"], sc["ci_hi"], dk))
            row[key] = sc["estimate"]
            row[key + "_questions"] = sc["n_questions"]
        per_draw.append(row)
        log(f"draw {d + 1}/{a.draws}: labelled {row['labelled']} pairs {row['adjacent_pairs']} "
            f"B1 {row['b1_G']:.6f} A2pre {row['A2_pre']:.5f} ({time.time() - td:.0f}s)")
    out = dict(design=dict(fraction=FRACTION, min_labels=MIN_LABELS, draws=a.draws, seed=SEED,
                           folds="8 question folds, repeat 0 (identical for dense and sparse)",
                           layer_pairs=len(pairs), gate=dict(abs_bias_over_sd=BIAS_GATE, coverage=COVERAGE_GATE),
                           sampling="per trace, uniform without replacement (sample_stratified)"),
               B1_G_joint=compare(series["B1_G_joint"], g_dense, sd_b1, cover["B1_G_joint"]),
               A2_pre_boundary=compare(series["A2_pre"], dense_scalar["pre"]["estimate"],
                                       dense_scalar["pre"]["sd"], cover["A2_pre"]),
               A2_pre_lag0_secondary=compare(series["A2_pre_lag0"], dense_scalar["pre_lag0"]["estimate"],
                                             dense_scalar["pre_lag0"]["sd"], cover["A2_pre_lag0"]),
               per_draw=per_draw, code_sha256=code_hashes(), tokens_sha256=tk["provenance"]["tokens_sha256"],
               slurm_job=os.environ.get("SLURM_JOB_ID"), seconds=time.time() - t0,
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    out["gate_passes"] = bool(out["B1_G_joint"]["passes"] and out["A2_pre_boundary"]["passes"])
    write_json(OUT / f"sparse_sim{('_' + a.tag) if a.tag else ''}.json", out)
    log(f"done in {out['seconds']:.0f}s; B1 {out['B1_G_joint']}; A2 {out['A2_pre_boundary']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--draws", type=int, default=50)
    s.add_argument("--max-pairs", type=int)
    s.add_argument("--tag", default="")
    s.set_defaults(fn=run)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

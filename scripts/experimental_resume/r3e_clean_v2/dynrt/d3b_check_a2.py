"""D3b verification: A2 event affinities of the first pilot question against a naive per-window computation.

  python -m dynrt.d3b_check_a2   (prints max abs differences per lag; exit 1 when above 1e-5)
"""
from __future__ import annotations

import sys

import numpy as np

from . import d3b_a2 as A
from . import d3b_b1 as B
from . import features as F
from .d3b_tokens import OUT_PILOT, PILOT_SHAPE, load_tokens


def main() -> int:
    tk = load_tokens(OUT_PILOT)
    p = A.make_pilot(tk, PILOT_SHAPE["num_experts"])
    e, k = p.experts, p.ids.shape[2]
    h = A.sentence_histograms(p.ids, p.sent, len(p.sentences), e)
    keep = np.ones(len(p.attempts), bool)
    qs = sorted(set(p.attempts["question"]))
    folds_q = B.make_folds(qs, 8, 1)[0]
    res = A.a2_run(p, h, keep, None, folds_q=folds_q, edges=A.length_edges(p.sentences))
    sent = p.sentences
    a_of_s, s_cls = sent["att"].to_numpy(), sent["cls"].to_numpy().astype(int)
    worst = 0.0
    for att0 in (0, 17):
        f0 = folds_q[p.attempts["q"].iloc[att0]]
        train = [a for a in range(len(p.attempts)) if folds_q[p.attempts["q"].iloc[a]] != f0]
        rows = np.flatnonzero(np.isin(a_of_s, train))
        lr = F.fit_profiles(h[rows], s_cls[rows], a_of_s[rows], p.n_layers, e).log_ratio().reshape(
            F.NC, p.n_layers, e)
        bd = A.build_boundaries(sent, np.ones(len(sent), bool))
        valid = A.window_valid(bd)
        cell = A.match_cell(bd, A.length_edges(sent))
        evt = bd.a != bd.b
        for lag in (-16, 0, 16, 32):
            li = A.LAGS.index(lag)
            ctrl_n = np.bincount(cell[(~evt) & valid[:, li]], minlength=F.NC * A.N_BIN * A.N_TERC)
            sel = np.flatnonzero((bd.att == att0) & evt & valid[:, li] & (ctrl_n[cell] >= A.MIN_CTRL))
            ref = np.zeros(p.n_layers)
            for m in sel:
                s = p.tok_offset[att0] + bd.r[m] + lag - A.WIN
                ids = p.ids[s:s + A.WIN]
                for layer in range(p.n_layers):
                    hw = np.bincount(ids[:, layer, :].ravel().astype(int), minlength=e) / (A.WIN * k)
                    ref[layer] += (hw * lr[bd.a[m], layer]).sum() - (hw * lr[bd.b[m], layer]).sum()
            q = int(p.attempts["q"].iloc[att0])
            got = res["aff_sums"][int(np.flatnonzero(res["questions"] == q)[0]), li]
            diff = float(np.abs(ref - got).max())
            worst = max(worst, diff)
            print(f"attempt {att0} lag {lag}: events {len(sel)}, max abs diff {diff:.3e}, "
                  f"scale {np.abs(ref).max():.3f}", flush=True)
    print("worst", worst)
    return int(worst > 1e-5)


if __name__ == "__main__":
    sys.exit(main())

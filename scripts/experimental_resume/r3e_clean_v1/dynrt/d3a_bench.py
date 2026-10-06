"""Timing benchmark of single multinomial ridge fits (not part of the analysis)."""
import time, numpy as np
from dynrt import d3a_cv
from dynrt.d3a_softmax import fit_softmax, predict_log_proba, weighted_nll
t0=time.time()
d = d3a_cv.load_data(antic_dir=None, text_scale=10.0)
print("load", time.time()-t0, flush=True)
train = np.flatnonzero(d.outer_folds[0] != 0)
test = np.flatnonzero(d.outer_folds[0] == 0)
t0=time.time(); b = d3a_cv.FoldBlocks(d, train, [test]); print("blocks", time.time()-t0, b.xs_tr.shape, b.xs_tr.nnz, flush=True)
xd, xe = b.dense(["base"])
for scale_lam in (100.0, 1.0, 0.01, 0.001):
    for ftol in (1e-10, 1e-8):
        t0=time.time()
        f = fit_softmax(xd, b.xs_tr, d.y[train], d.w[train], scale_lam, 7, ftol=ftol, maxiter=1000)
        lp = predict_log_proba(f, xe[0], b.xs_ev[0])
        print(f"lam {scale_lam} ftol {ftol}: iters {f.n_iter} conv {f.converged} obj {f.objective:.8f} gmax {f.grad_max:.2e} "
              f"test nll {weighted_nll(lp, d.y[test], d.w[test]):.6f} {time.time()-t0:.1f}s", flush=True)

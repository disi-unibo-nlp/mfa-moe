"""P-A1 sensitivity: the primary comparison at another text-block scale (the registered run uses the scale
chosen by the baseline-only calibration; at that scale every inner CV picked the smallest grid penalty).

CLI: python -m dynrt.d3a_sens --scale 10 [--workers 8]  -> results/A1/sens_scale<scale>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import d3a_cv, d3a_eval
from .d3a_run import ANTIC, OUT, _clean, code_hashes, input_hashes, log, verify_freezes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    fz = verify_freezes()
    d = d3a_cv.load_data(antic_dir=ANTIC, text_scale=a.scale)
    models = {"base": ["base"], "aug": ["base", "last"]}
    t0 = time.time()
    res = d3a_cv.run_cv(d, models, workers=a.workers)
    log(f"sens cv done in {time.time() - t0:.0f}s; diag {res['diag']}")
    ev = d3a_eval.evaluate(res["logp"]["base"], res["logp"]["aug"], d.y, d.groups, d.w, n_boot=a.n_boot,
                           per_class=d.a, full=False)
    g = ev["gain"]
    log(f"scale {a.scale}: gain {g['point']:.5f} [{g['ci_lo']:.5f}, {g['ci_hi']:.5f}] p1={g['p_one_sided']:.4f} "
        f"positive repeats {ev['n_positive_repeats']}/5")
    out = dict(scale=a.scale, evaluation=ev, diag=res["diag"], lambdas={m: v.tolist() for m, v in res["lambdas"].items()},
               inner_curves={m: np.asarray(v).tolist() for m, v in res["curves"].items()},
               seconds=time.time() - t0, frozen=fz, inputs=input_hashes(), code=code_hashes())
    (OUT / f"sens_scale{a.scale:g}.json").write_text(json.dumps(_clean(out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

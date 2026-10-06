"""Stage 4 for C6 (lexical Explore-onset mismatch; qwen36 dev+tune cohort A only).

Prefix controls only (log j, difficulty, dataset FE, marker-identity FE); final length is never used.
  python -m dynrt.c6 run [--lam-scale sum]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from . import cv
from . import transitions as T
from .c1 import Prepared, clean, log
from .c2 import _fmt, partial_generic, verify_frozen_d2
from .common import RESULTS

OUT = RESULTS / "C6"
MODELS = {"pbase": ["pcontrols"], "px": ["pcontrols", "onset.x"]}


def run(workers: int, n_boot: int, lam_scale: str) -> dict:
    fz = verify_frozen_d2()
    p = Prepared("qwen36")
    onset = T.load_onset(p.coh, RESULTS / "qwen36" / "A" / "onset")
    valid = p.valid & onset.has
    n_valid_q = pd.Series(valid.astype(float)).groupby(p.groups).transform("sum").to_numpy()
    weights = np.where(valid, 1.0 / np.maximum(n_valid_q, 1.0), 0.0)
    pctrl = T.PrefixControls(onset, p.difficulty, p.coh.attempts["dataset"].to_numpy(), 6)
    blocks = dict(pcontrols=pctrl, onset=T.OnsetBlock(p.coh.table, onset))
    log(f"qwen36 A: {len(p.y)} attempts, {int(onset.has.sum())} with a decision in [64, 8192], "
        f"{int(valid.sum())} scored ({int((p.y[valid] == 0).sum())} wrong)")
    if lam_scale == "sum":
        orig = cv.ridge_logistic
        cv.ridge_logistic = lambda x, y, w, lam, **k: orig(x, y, w, lam / float(w.sum()), **k)
    spec = cv.CVSpec(blocks=blocks, models=MODELS, y=np.where(valid, p.y, 0.0), groups=p.groups,
                     weights=weights, repeats=5, outer=4, inner=3, seed0=0, valid=valid)
    t0 = time.time()
    res = cv.run_cv(spec, workers=workers, log=log)
    v = valid
    c = cv.evaluate({"pbase": res["preds"]["pbase"][:, v], "px": res["preds"]["px"][:, v]}, "pbase", "px",
                    p.y[v], p.groups[v], weights[v], n_boot=n_boot)
    log(f"C6_primary: gain {c['gain']:.5f} [{c['ci_lo']:.5f}, {c['ci_hi']:.5f}] p1={c['p_one_sided']:.4f}")
    out = dict(candidate="C6", model="qwen36", cohort="A", lam_scale=lam_scale, frozen=fz,
               n_attempts=int(len(p.y)), with_decision=int(onset.has.sum()), n_scored=int(v.sum()),
               n_wrong=int((p.y[v] == 0).sum()), n_questions=int(len(set(p.groups[v]))),
               marker_counts={m: int(((onset.marker == i) & v).sum()) for i, m in enumerate(
                   ["can we", "could", "is it possible", "maybe", "or maybe", "what"])},
               primary=c, lambdas={m: dict(median=float(np.median(l))) for m, l in res["lambdas"].items()},
               seconds=time.time() - t0)
    if lam_scale == "mean":
        c6 = pd.read_parquet(RESULTS / "qwen36" / "A" / "c6_oof.parquet")
        if not (c6["attempt_id"].to_numpy() == p.coh.attempts["attempt_id"].to_numpy()).all():
            raise RuntimeError("frozen C6 table is not aligned")
        x = c6["x_mean"].to_numpy(float)
        cmat = pctrl.fit_transform(np.flatnonzero(valid))
        cmat = np.where(np.isnan(cmat), np.nanmean(cmat, axis=0), cmat)
        cmat = cmat[:, cmat[valid].std(0) > 1e-12]
        rows = valid & np.isfinite(x)
        out["partial_correctness"] = partial_generic(x, p.y, cmat, p.groups, rows, n_boot)
        b_rows = onset.has & np.isfinite(onset.burden) & np.isfinite(x)
        from v3an import stats as vs
        bres = vs.partial_corr(x[b_rows][:, None], onset.burden[b_rows], cmat[b_rows], clusters=p.groups[b_rows],
                               n_boot=n_boot, rng=np.random.default_rng(20260929))
        out["burden_association"] = dict(
            n=int(b_rows.sum()), burden_mean=float(onset.burden[b_rows].mean()), r=float(bres["r"][0]),
            ci_lo=float(bres["ci_lo"][0]), ci_hi=float(bres["ci_hi"][0]), p=float(bres["p"][0]),
            raw_r=float(np.corrcoef(x[b_rows], onset.burden[b_rows])[0, 1]),
            note="controls: prefix controls only; rows: scored or not, decision and >= 1 labelled sentence in (j, j+512]")
        out["x_summary"] = dict(mean=float(np.nanmean(x)), sd=float(np.nanstd(x)))
    return out


def summary_md(r: dict) -> str:
    c = r["primary"]
    L = ["# C6 lexical Explore-onset mismatch: results (D2, qwen36 dev+tune A)", "",
         f"{r['with_decision']} of {r['n_attempts']} attempts have a first decision with 64 <= j <= 8192; "
         f"{r['n_scored']} scored ({r['n_wrong']} wrong).", "",
         f"Primary: gain of prefix-C + x over prefix-C = **{_fmt(c['gain'], 5)}** nats/attempt, 95% CI "
         f"[{_fmt(c['ci_lo'], 5)}, {_fmt(c['ci_hi'], 5)}], one-sided p {_fmt(c['p_one_sided'], 4)}; "
         f"dAUC {_fmt(c['dauc']['diff'], 4)} [{_fmt(c['dauc']['ci_lo'], 4)}, {_fmt(c['dauc']['ci_hi'], 4)}]; "
         f"dBrier improvement {_fmt(c['dbrier_improvement']['diff'], 5)}. Verdict and Holm adjustment in "
         "../SUMMARY.md.", ""]
    pc = r.get("partial_correctness")
    if pc:
        L.append(f"Partial association of x with correctness given the prefix controls: r {_fmt(pc['r'])} "
                 f"[{_fmt(pc['ci_lo'])}, {_fmt(pc['ci_hi'])}], p {_fmt(pc['p'], 4)} (n {pc['n']}).")
    ba = r.get("burden_association")
    if ba:
        L.append(f"Association of x with the future Explore burden (Explore share of labelled sentences starting in "
                 f"(j, j+512]; mean {_fmt(ba['burden_mean'], 3)}, n {ba['n']}): partial r {_fmt(ba['r'])} "
                 f"[{_fmt(ba['ci_lo'])}, {_fmt(ba['ci_hi'])}], p {_fmt(ba['p'], 4)}; raw r {_fmt(ba['raw_r'])}.")
    if r.get("sensitivity"):
        s = r["sensitivity"]["primary"]
        L.append(f"Sensitivity (summed-loss ridge scale): gain {_fmt(s['gain'], 5)} [{_fmt(s['ci_lo'], 5)}, {_fmt(s['ci_hi'], 5)}], "
                 f"p1 {_fmt(s['p_one_sided'], 4)}.")
    L.append(f"Marker identity counts among scored attempts: {json.dumps(r['marker_counts'])}.")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--n-boot", type=int, default=1000)
    a.add_argument("--lam-scale", choices=("mean", "sum"), default="mean")
    sub.add_parser("assemble")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "run":
        res = run(args.workers, args.n_boot, args.lam_scale)
        name = "qwen36_A" + ("" if args.lam_scale == "mean" else "_sumscale")
        (OUT / f"{name}.json").write_text(json.dumps(clean(res), indent=1))
    else:
        r = json.loads((OUT / "qwen36_A.json").read_text())
        s = OUT / "qwen36_A_sumscale.json"
        if s.exists():
            r["sensitivity"] = {"primary": json.loads(s.read_text())["primary"]}
        (OUT / "results.json").write_text(json.dumps(clean(r), indent=1))
        (OUT / "summary.md").write_text(summary_md(clean(r)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

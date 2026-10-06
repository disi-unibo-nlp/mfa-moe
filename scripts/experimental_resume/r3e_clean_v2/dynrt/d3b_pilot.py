"""D3b pilot stages (dense Qwen3.6 pilot; exploratory measurement feasibility; no outcome is read).

  python -m dynrt.d3b_pilot overlap                         pilot questions vs steering-v1 split-v1
  python -m dynrt.d3b_pilot b1  [--max-pairs N --repeats R --n-perm P --tag T]     -> b1.json
  python -m dynrt.d3b_pilot a2  [--tag T]                   -> a2.json, a2_curves.csv, a2_curves.png
Subsets: `all64` = every pilot question (measurement feasibility of the pipeline, NOT steering-relevant);
`dev_tune` = the pilot questions in steering-v1 dev/tune (the only numbers that may inform steering);
pilot questions of the steering confirm split are excluded from every dev_tune result (fit and evaluation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import d3b_a2 as A
from . import d3b_b1 as B
from .common import RESULTS, load_split, sha256_file
from .d3b_tokens import OUT_PILOT, PILOT_MANIFEST, PILOT_SHAPE, PILOT_TRACES, load_tokens
from .d3b_util import clean, code_hashes, log, write_json

OUT = RESULTS / "pilot"
SUBSETS = ("all64", "dev_tune")


def workers() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


def subset_masks(attempts: pd.DataFrame) -> dict:
    split = attempts["split"].to_numpy()
    return {"all64": np.ones(len(attempts), bool), "dev_tune": split != "confirm"}


def overlap_report(attempts: pd.DataFrame) -> dict:
    """Pilot questions vs split-v1, matched on (dataset, source_problem_id) from the trace records."""
    split = load_split()
    man = json.loads(PILOT_MANIFEST.read_text())
    ident = {(t["identity"][1], t["identity"][2].rsplit("__sample_", 1)[0]) for t in man["traces"]}
    from_traces = {tuple(q.split("|", 1)) for q in attempts["question"]}
    counts = attempts["split"].value_counts().to_dict()
    per_dataset = attempts.groupby(["dataset", "split"]).size().rename("n").reset_index().to_dict("records")
    return dict(pilot_questions=int(attempts["question"].nunique()), pilot_traces=int(len(attempts)),
                key="(dataset, source_problem_id) from the trace records (manifest identity agrees)",
                manifest_identity_matches_trace_records=bool(ident == from_traces),
                in_split_v1=int(sum(q in split for q in attempts["question"])),
                confirm=int(counts.get("confirm", 0)), dev=int(counts.get("dev", 0)),
                tune=int(counts.get("tune", 0)),
                retained_dev_tune=int(counts.get("dev", 0) + counts.get("tune", 0)),
                per_dataset=per_dataset, split_v1_seal_verified=True,
                confirm_question_ids_withheld=True)


# -------------------------------------------------------------------------------------- B1

def gate_b1(res: dict) -> dict:
    """Feasibility gate: positive G_joint in >= 4/5 fold repeats AND above the 95th percentile of the nulls."""
    obs = res["observed"]
    perm = res.get("permutation")
    pos = obs["positive_repeats"] >= 4
    above = bool(perm and perm["observed_above_p95"])
    return dict(positive_repeats=int(obs["positive_repeats"]), n_repeats=int(obs["n_repeats"]),
                positive_in_at_least_4_of_5=bool(pos), above_permutation_p95=above,
                G_joint=obs["G_joint"]["mean"], permutation_p95=perm["p95"] if perm else None,
                passes=bool(pos and above))


def cmd_b1(a) -> int:
    t0 = time.time()
    tk = load_tokens(OUT_PILOT)
    masks = subset_masks(tk["attempts"])
    pairs = B.layer_pairs(PILOT_SHAPE["num_layers"])
    if a.max_pairs:
        pairs = pairs[::max(1, len(pairs) // a.max_pairs)][:a.max_pairs]
    out = dict(design=dict(n_folds=a.folds, repeats=a.repeats, n_perm=a.n_perm, lambda_=B.LAMBDA, eps=B.EPS_Q,
                           layer_pairs=len(pairs), gaps=list(B.GAPS), experts=PILOT_SHAPE["num_experts"],
                           top_k=PILOT_SHAPE["top_k"], top_lexical=B.TOP_LEX,
                           classes_weighting="question-weighted (each question equal)",
                           permutation_seed=0, tag=a.tag),
               subsets={}, code_sha256=code_hashes(), tokens_sha256=tk["provenance"]["tokens_sha256"],
               time=time.strftime("%Y-%m-%dT%H:%M:%S"), slurm_job=os.environ.get("SLURM_JOB_ID"))
    for name in SUBSETS if not a.subset else (a.subset,):
        ts, questions = B.make_tokenset(tk, masks[name], PILOT_SHAPE["num_experts"])
        res = B.run_b1(ts, questions, pairs, n_folds=min(a.folds, len(questions)), repeats=a.repeats,
                       n_perm=a.n_perm, workers=workers(), tag=name, n_classperm=a.n_classperm)
        res["gate"] = gate_b1(res)
        res["steering_relevant"] = name == "dev_tune"
        out["subsets"][name] = res
        log(f"{name}: gate {res['gate']}")
    out["seconds"] = time.time() - t0
    path = OUT / (f"b1{('_' + a.tag) if a.tag else ''}.json")
    write_json(path, out)
    log(f"wrote {path} in {out['seconds']:.0f}s")
    return 0


# -------------------------------------------------------------------------------------- A2

def plot_curves(frames: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(frames), 2, figsize=(11, 3.6 * len(frames)), squeeze=False)
    for row, (name, df) in enumerate(frames.items()):
        lm = df[df["layer"] == -1].sort_values("lag")
        ax = axes[row, 0]
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(0, color="0.6", lw=0.8, ls="--")
        ax.fill_between(lm["lag"], lm["ci_lo"], lm["ci_hi"], color="#4C78A8", alpha=0.25, lw=0)
        ax.plot(lm["lag"], lm["excess"], color="#4C78A8", marker="o", ms=4)
        ax.set_xlabel("lag of the window end relative to the boundary (tokens)")
        ax.set_ylabel("excess destination-minus-source affinity")
        ax.set_title(f"{name}: layer mean, a->b vs matched a->a (95% question bootstrap)", fontsize=9)
        piv = df[df["layer"] >= 0].pivot(index="layer", columns="lag", values="excess")
        ax = axes[row, 1]
        vmax = float(np.nanpercentile(np.abs(piv.to_numpy()), 98)) or 1.0
        im = ax.imshow(piv.to_numpy(), aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=(piv.columns.min() - 8, piv.columns.max() + 8, -0.5, piv.shape[0] - 0.5))
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("lag (tokens)")
        ax.set_ylabel("layer")
        ax.set_title(f"{name}: per-layer excess", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def cmd_a2(a) -> int:
    t0 = time.time()
    tk = load_tokens(OUT_PILOT)
    p = A.make_pilot(tk, PILOT_SHAPE["num_experts"])
    n_sent = len(p.sentences)
    log(f"pilot: {len(p.att)} tokens, {n_sent} sentences, {len(p.attempts)} attempts")
    h = A.sentence_histograms(p.ids, p.sent, n_sent, p.experts)
    log(f"sentence histograms {h.shape} ({time.time() - t0:.0f}s)")
    masks = subset_masks(p.attempts)
    edges = A.length_edges(p.sentences)
    out = dict(design=dict(lags=list(A.LAGS), window=A.WIN, pre_lags=list(A.PRE_LAGS),
                           post_lags=list(A.POST_LAGS), min_controls=A.MIN_CTRL, n_folds=a.folds,
                           n_boot=A.N_BOOT, length_edges=edges.tolist(), profile_kappa=50.0,
                           matching="source class x absolute-position bin x source-sentence-length tercile",
                           censoring="window inside [source-run start, destination-run end)",
                           window_definition="16 tokens ending at r + lag (exclusive)", tag=a.tag),
               subsets={}, code_sha256=code_hashes(), tokens_sha256=tk["provenance"]["tokens_sha256"],
               time=time.strftime("%Y-%m-%dT%H:%M:%S"), slurm_job=os.environ.get("SLURM_JOB_ID"))
    frames = {}
    for name in SUBSETS if not a.subset else (a.subset,):
        res = A.a2_run(p, h, masks[name], None, n_folds=min(a.folds, int(masks[name].sum())), edges=edges)
        summ = A.summarize_a2(res)
        cv = A.curves(res)
        frames[name] = A.curves_frame(cv, res, name)
        for k in ("pre_boundary", "post_boundary"):
            summ[k].pop("question_values", None)
        out["subsets"][name] = dict(summary=summ, steering_relevant=name == "dev_tune",
                                    n_questions=res["n_q"], sup_t_critical=cv["crit"], seconds=res["seconds"])
        pre = summ["pre_boundary"]
        log(f"{name}: events {res['n_events']} controls {res['n_controls']}; pre-boundary excess "
            f"{pre['estimate']:.5f} [{pre['ci_lo']:.5f}, {pre['ci_hi']:.5f}] "
            f"post {summ['post_boundary']['estimate']:.5f} ({res['seconds']:.0f}s)")
    tag = f"_{a.tag}" if a.tag else ""
    out["seconds"] = time.time() - t0
    write_json(OUT / f"a2{tag}.json", out)
    df = pd.concat(frames.values(), ignore_index=True)
    df.to_csv(OUT / f"a2_curves{tag}.csv", index=False)
    plot_curves(frames, OUT / f"a2_curves{tag}.png")
    log(f"wrote a2{tag}.json, a2_curves{tag}.csv/png in {out['seconds']:.0f}s")
    return 0


def cmd_overlap(a) -> int:
    tk_att = pd.read_parquet(OUT_PILOT / "attempts.parquet")
    rep = overlap_report(tk_att)
    rep["manifest_sha256"] = sha256_file(PILOT_MANIFEST)
    rep["traces_sha256"] = sha256_file(PILOT_TRACES)
    write_json(OUT / "overlap.json", rep)
    print(json.dumps(clean(rep), indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("overlap")
    s.set_defaults(fn=cmd_overlap)
    s = sub.add_parser("b1")
    s.add_argument("--max-pairs", type=int)
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--folds", type=int, default=8)
    s.add_argument("--n-perm", type=int, default=B.N_PERM)
    s.add_argument("--n-classperm", type=int, default=B.N_PERM)
    s.add_argument("--subset", choices=SUBSETS)
    s.add_argument("--tag", default="")
    s.set_defaults(fn=cmd_b1)
    s = sub.add_parser("a2")
    s.add_argument("--folds", type=int, default=8)
    s.add_argument("--subset", choices=SUBSETS)
    s.add_argument("--tag", default="")
    s.set_defaults(fn=cmd_a2)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

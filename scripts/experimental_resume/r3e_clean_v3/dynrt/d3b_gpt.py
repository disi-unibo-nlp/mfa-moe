"""D3b stage 4: P-B1 on GPT cohort A (registered PRIMARY; ANALYSIS_PLAN.md Addendum 3). No outcome is read.

  python -m dynrt.d3b_gpt run [--tag T --max-pairs N --repeats R --n-perm P] -> results/B1/{results.json, summary.md}

Universe: the 1,547 GPT cohort-A attempts (one per question) minus the 37 questions that also form cohort B (B is
reserved), labelled-sentence tokens only. Folds: C8 machinery, 4 outer question folds x 5 repeats (seeds 0-4).
G_joint (nats per token-pair observation) is the held-out composite log-score gain of the class-specific
alternative over the no-3-way-interaction null, averaged over the 43 layer pairs (gaps 1 and 4) and the repeats,
question-weighted; CI = 1,000 paired question bootstraps.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from . import d3b_b1 as B
from .common import CLASSES, RESULTS, V3R2
from .d3b_tokens import OUT_GPT, load_tokens
from .d3b_util import clean, code_hashes, log, write_json

OUT = RESULTS / "B1"
ALPHA = 0.05
HOLM_M = 3
GPT_SHAPE = dict(num_experts=32, num_layers=24, top_k=4)


def verdict(res: dict, threshold: float = B.THRESHOLD) -> dict:
    """Registered decision rule (Addendum 3): Holm p < .05, paired 95% CI above 0, >= 4/5 repeats positive,
    gain >= threshold; an upper CI below the threshold deprioritises; otherwise inconclusive."""
    g = res["observed"]["G_joint"]
    p = g["p_one_sided"]
    cond = dict(
        holm_safe_p_lt_alpha_over_3=bool(p < ALPHA / HOLM_M),
        ci_lower_above_zero=bool(g["ci_lo"] > 0),
        positive_in_at_least_4_of_5_repeats=bool(res["observed"]["positive_repeats"] >= 4),
        gain_at_least_threshold=bool(g["mean"] >= threshold))
    if all(cond.values()):
        label = "ADVANCE (pending the Holm adjustment over the 3-primary family)"
    elif g["ci_hi"] < threshold:
        label = "DEPRIORITISE (CI upper bound below the threshold)"
    else:
        label = "INCONCLUSIVE"
    perm, lex = res.get("permutation"), res.get("lexical")
    diag = dict(
        above_permutation_p95=bool(perm and perm["observed_above_p95"]),
        lexical_G_at_least_threshold=bool(lex and lex["G_joint"]["mean"] >= threshold),
        lexical_ci_lower_above_zero=bool(lex and lex["G_joint"]["ci_lo"] > 0),
        gap_1_ci_lower_above_zero=bool(res["observed"]["gap"]["1"]["ci_lo"] > 0),
        gap_4_ci_lower_above_zero=bool(res["observed"]["gap"]["4"]["ci_lo"] > 0))
    return dict(threshold=threshold, gain=g["mean"], ci=[g["ci_lo"], g["ci_hi"]], p_one_sided=p,
                p_for_holm=p, holm_m=HOLM_M, holm_adjusted_if_smallest=min(1.0, HOLM_M * p),
                conditions=cond, verdict=label, diagnostics=diag)


def check_against_v3an(att: pd.DataFrame) -> dict:
    """Token and sentence counts per attempt must equal v3an's alignment (outcome columns are not read)."""
    v = pd.read_parquet(V3R2 / "gpt/A/attempts.parquet", columns=["attempt_id", "labelled_tokens",
                                                                "labelled_units", "n_reasoning"])
    m = att.merge(v, on="attempt_id", how="left", suffixes=("", "_v3an"))
    return dict(attempts=int(len(att)), missing=int(m["labelled_tokens"].isna().sum()),
                labelled_tokens_mismatch=int((m["n_labelled_tokens"] != m["labelled_tokens"]).sum()),
                labelled_units_mismatch=int((m["n_sentences"] != m["labelled_units"]).sum()),
                n_reasoning_mismatch=int((m["n_reasoning"] != m["n_reasoning_v3an"]).sum()))


def render_summary(res: dict, v: dict, info: dict) -> str:
    o = res["observed"]
    g = o["G_joint"]
    lines = [
        "# D3b P-B1: class-dependent cross-layer co-selection, GPT cohort A (registered primary)", "",
        f"Verdict: **{v['verdict']}**", "",
        f"- G_joint = {g['mean']:.6f} nats per token-pair observation, 95% paired question bootstrap "
        f"[{g['ci_lo']:.6f}, {g['ci_hi']:.6f}] (1,000 resamples of {g['n_questions']} questions); threshold "
        f"{v['threshold']}.",
        f"- One-sided bootstrap p (gain > 0) = {v['p_one_sided']:.4f} (floor {1 / (g['n_boot'] + 1):.4f}); "
        f"p for Holm = {v['p_for_holm']:.4f}; Holm-adjusted if it were the smallest of the 3 = "
        f"{v['holm_adjusted_if_smallest']:.4f}.",
        f"- Positive in {o['positive_repeats']}/{o['n_repeats']} fold repeats: "
        f"{[round(x, 6) for x in o['per_repeat_G']]}.",
        "- Conditions: " + ", ".join(f"{k}={val}" for k, val in v["conditions"].items()) + ".", "",
        "## Data", "",
        f"- Questions in the analysis: {info['n_questions']} (GPT cohort A {info['n_questions_a']} minus "
        f"{info['n_questions_excluded_b']} cohort-B questions); labelled-sentence tokens: {info['n_tokens']:,}; "
        f"sentences: {info['n_sentences']:,}; layer pairs: {res['n_pairs']} (gap 1: 23, gap 4: 20); 32 experts, k = 4.",
        f"- Folds: {res['n_folds']} outer question folds x {res['repeats']} repeats (seeds 0-4).", "",
        "## Diagnostics", "",
    ]
    perm = res.get("permutation")
    if perm:
        lines.append(f"- Within-layer whole-top-k-set permutation null (within class x absolute-position bin, "
                     f"{perm['n_perm']} permutations): mean gain {perm['mean']:.6f}, 95th percentile "
                     f"{perm['p95']:.6f}; observed {'above' if perm['observed_above_p95'] else 'NOT above'} it "
                     f"(rank p = {perm['rank_p']:.3f}).")
    lex = res.get("lexical")
    if lex:
        lines.append(f"- Lexical sensitivity (the {lex['n_frequent_ids']} most frequent token ids dropped, "
                     f"{100 * lex['fraction_dropped']:.1f}% of tokens): G_joint = {lex['G_joint']['mean']:.6f} "
                     f"[{lex['G_joint']['ci_lo']:.6f}, {lex['G_joint']['ci_hi']:.6f}], positive repeats "
                     f"{lex['positive_repeats']}/{lex['n_repeats']}.")
    cp = res.get("class_permutation_within_token")
    if cp:
        lines.append(f"- Additional diagnostic (not registered): class labels shuffled among tokens of the same "
                     f"vocabulary id and position bin ({cp['n_perm']} shuffles): mean gain {cp['mean']:.6f} "
                     f"versus {cp['observed_repeat0']:.6f} observed (repeat 0); retained fraction "
                     f"{cp['retained_fraction']:.2f}.")
    lines += ["", "### Depth-gap strata", "", "| gap | pairs | G_joint | 95% CI |", "|---|---|---|---|"]
    n_gap = {"1": 23, "4": 20}
    for gp, s in o["gap"].items():
        lines.append(f"| {gp} | {n_gap.get(gp, '')} | {s['mean']:.6f} | [{s['ci_lo']:.6f}, {s['ci_hi']:.6f}] |")
    lines += ["", "### Per class (mean held-out gain over the class's tokens)", "", "| class | G |", "|---|---|"]
    for c in CLASSES:
        val = o["per_class"].get(c)
        lines.append(f"| {c} | {'' if val is None else f'{val:.6f}'} |")
    per_pair = sorted(o["per_pair"], key=lambda r: -r["G"])
    lines += ["", "### Layer pairs (top 5 / bottom 5)", "", "| i | j | gap | G |", "|---|---|---|---|"]
    for r in per_pair[:5] + per_pair[-5:]:
        lines.append(f"| {r['i']} | {r['j']} | {r['gap']} | {r['G']:.6f} |")
    ipf = o["ipf"]
    lines += ["", f"IPF: {ipf['fits']} fits, median {ipf['median_iterations']:.0f} iterations, max "
                  f"{ipf['max_iterations']}, unconverged {ipf['unconverged']}.", "",
              "The null-bias caveat of C8 applies: held-out gains of an over-parameterised alternative are "
              "biased downward, so the permutation null is the like-for-like reference.", ""]
    return "\n".join(lines)


def run(a) -> int:
    t0 = time.time()
    tk = load_tokens(OUT_GPT)
    att = tk["attempts"]
    chk = check_against_v3an(att)
    log(f"v3an alignment cross-check: {chk}")
    if chk["missing"] or chk["labelled_tokens_mismatch"] or chk["labelled_units_mismatch"]:
        raise RuntimeError(f"token table differs from the v3an alignment: {chk}")
    b_q = set(pd.read_parquet(V3R2 / "gpt/B/attempts.parquet", columns=["question"])["question"])
    keep = ~att["question"].isin(b_q).to_numpy()
    ts, questions = B.make_tokenset(tk, keep, GPT_SHAPE["num_experts"])
    pairs = B.layer_pairs(GPT_SHAPE["num_layers"])
    if a.max_pairs:
        pairs = pairs[::max(1, len(pairs) // a.max_pairs)][:a.max_pairs]
    info = dict(n_questions=len(questions), n_questions_a=int(att["question"].nunique()),
                n_questions_excluded_b=int(att["question"].isin(b_q).sum()), n_tokens=int(len(ts.cls)),
                n_sentences=int(att.loc[keep, "n_sentences"].sum()),
                tokens_per_class={c: int((ts.cls == k).sum()) for k, c in enumerate(CLASSES)})
    log(f"universe: {info}")
    res = B.run_b1(ts, questions, pairs, n_folds=4, repeats=a.repeats, n_perm=a.n_perm, workers=workers(),
                   n_boot=B.N_BOOT, tag="gptA", n_classperm=a.n_classperm)
    v = verdict(res)
    out = dict(result=res, verdict=v, data=info, v3an_cross_check=chk,
               design=dict(pairs=len(pairs), gaps=list(B.GAPS), lambda_=B.LAMBDA, eps=B.EPS_Q,
                           threshold=B.THRESHOLD, n_boot=B.N_BOOT, n_folds=4, repeats=a.repeats,
                           experts=GPT_SHAPE["num_experts"], top_k=GPT_SHAPE["top_k"],
                           weighting="question-weighted: token-pair mean within question, questions equal",
                           folds="C8 question_folds seeds 0..4 over the cohort-A questions absent from cohort B",
                           tag=a.tag),
               code_sha256=code_hashes(), tokens_sha256=tk["provenance"]["tokens_sha256"],
               slurm_job=os.environ.get("SLURM_JOB_ID"), time=time.strftime("%Y-%m-%dT%H:%M:%S"),
               seconds=time.time() - t0)
    tag = f"_{a.tag}" if a.tag else ""
    write_json(OUT / f"results{tag}.json", out)
    (OUT / f"summary{tag}.md").write_text(render_summary(clean(res), clean(v), info))
    log(f"verdict: {v['verdict']}; G {v['gain']:.6f} [{v['ci'][0]:.6f}, {v['ci'][1]:.6f}] p {v['p_one_sided']:.4f}")
    return 0


def workers() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--max-pairs", type=int)
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--n-perm", type=int, default=B.N_PERM)
    s.add_argument("--n-classperm", type=int, default=B.N_PERM)
    s.add_argument("--tag", default="")
    s.set_defaults(fn=run)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

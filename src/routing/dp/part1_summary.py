"""results/part1/summary.md from the per-model Part 1 JSON files."""
from __future__ import annotations

import json

import numpy as np

from . import common as cm
from . import part1 as p1

OUT = cm.RESULTS / "part1"


def _f(x, nd=3):
    return "NA" if x is None else f"{x:.{nd}f}"


def _thirds(v):
    v = np.array([np.nan if x is None else x for x in v], float)
    return [float(np.nanmean(p)) for p in np.array_split(v, 3)]


def _load() -> dict:
    return {m: json.loads((OUT / f"{m}.json").read_text()) for m in cm.MODELS if (OUT / f"{m}.json").exists()}


def write_summary() -> None:
    R = _load()
    L = ["# Part 1: depth-path structure of top-1 experts (descriptive; no outcomes)", "",
         "Plan `ANALYSIS_PLAN.md` (frozen 2026-09-29). Per model: 200 cohort-A attempts (seed 0; Qwen3.6 = split-v1 "
         "dev+tune only), up to 4,096 uniformly sampled reasoning tokens each; top-1 expert = selected expert with the "
         "largest gate weight; depth = position in the routed-layer list. Nulls: "
         f"{p1.N_NULL} token shuffles within 10 relative-position bins (D-MI, concentration), "
         f"{p1.N_PERM_CLASS} sentence-label permutations within absolute-position bins (class differences).", "",
         "## Sample", "", "| model | layers | experts | top-k | attempts | tokens | median tokens/attempt | "
         "labelled token share |", "|---|---|---|---|---|---|---|---|"]
    for m, r in R.items():
        s = r["spec"]
        L.append(f"| {m} | {s['num_layers']} | {s['num_experts']} | {s['top_k']} | {r['n_attempts']} | "
                 f"{r['n_tokens']:,} | {r['n_tokens_per_attempt']['median']:.0f} | "
                 f"{_f(r['labelled_token_frac'], 2)} |")
    L += ["", "## 1. Depth NMI (top-1 at layer l vs l+1) against the within-position-bin token-shuffle null", "",
          "NMI = MI / mean(H(X), H(Y)). Ratio = observed / null mean; p = share of null layer-mean NMIs >= observed "
          f"(+1 smoothing, {p1.N_NULL} draws, so the floor is 1/{p1.N_NULL + 1}). Thirds are means over the "
          "early / mid / late layer pairs.",
          "", "| model | layer-mean NMI | null | ratio | p | ratio early | ratio mid | ratio late | "
          "min..max layer ratio | layers with z > 3 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in R.items():
        d = r["dmi"]
        ratio = np.array([np.nan if x is None else x for x in d["ratio"]], float)
        z = np.array([np.nan if x is None else x for x in d["z"]], float)
        e, mi, la = _thirds(d["ratio"])
        L.append(f"| {m} | {_f(d['layer_mean_nmi'], 4)} | {_f(d['layer_mean_null'], 4)} | "
                 f"{_f(d['layer_mean_ratio'], 2)} | {_f(d['p_layer_mean'], 3)} | {_f(e, 2)} | {_f(mi, 2)} | "
                 f"{_f(la, 2)} | {np.nanmin(ratio):.2f}..{np.nanmax(ratio):.2f} | "
                 f"{int((z > 3).sum())}/{len(z)} |")
    L += ["", "## 2. Path predictability: training-half bigram argmax vs unconditional mode (held-out attempt half)", "",
          "Accuracy of predicting the layer-(l+1) top-1 expert. Direction 0 = plan split (half A trains, half B is "
          "held out); direction 1 swaps the halves. CI = attempt bootstrap (1,000) of the layer-mean gain.", "",
          "| model | acc bigram | acc mode | gain [95% CI] | gain (swapped halves) | test tokens |",
          "|---|---|---|---|---|---|"]
    for m, r in R.items():
        a, b = r["predictability"]["direction0"], r["predictability"]["direction1"]
        L.append(f"| {m} | {_f(a['layer_mean_bigram'], 4)} | {_f(a['layer_mean_mode'], 4)} | "
                 f"{_f(a['layer_mean_delta'], 4)} [{_f(a['delta_ci95'][0], 4)}, {_f(a['delta_ci95'][1], 4)}] | "
                 f"{_f(b['layer_mean_delta'], 4)} | {a['n_test_tokens']:,} |")
    L += ["", "## 3. Path concentration: share of tokens on the 100 most frequent 3-layer top-1 paths", "",
          "Mean over the (L-2) windows (l, l+1, l+2). `independent` shuffles every layer column separately over all "
          "tokens (the plan's independent-layer expectation, same sample size and selection bias); `binned` shuffles "
          "columns l+1, l+2 within the relative-position bin.", "",
          "| model | observed | independent null | ratio | p | binned null | ratio (binned) | ratio early / mid / late "
          "(independent) |", "|---|---|---|---|---|---|---|---|"]
    for m, r in R.items():
        c = r["concentration"]
        e, mi, la = _thirds(c["ratio_indep"])
        L.append(f"| {m} | {_f(c['layer_mean_observed'], 4)} | {_f(c['layer_mean_null_indep'], 4)} | "
                 f"{_f(c['layer_mean_ratio_indep'], 2)} | {_f(c['p_layer_mean_indep'], 3)} | "
                 f"{_f(c['layer_mean_null_binned'], 4)} | {_f(c['layer_mean_ratio_binned'], 2)} | "
                 f"{_f(e, 1)} / {_f(mi, 1)} / {_f(la, 1)} |")
    L += ["", "## 4. Score trajectories by depth", "",
          "Mean top-1 gate weight (w1) and boundary gap at the first / middle / last routed layer; the per-token "
          "Spearman correlation of w1 with depth (\"commitment slope\"): token-level mean, and the attempt-weighted "
          "mean (each attempt's mean slope averaged over attempts) with an attempt-bootstrap 95% CI.", "",
          "| model | w1 first / mid / last | w1 SD (layer mean) | gap first / mid / last | slope mean (tokens) | "
          "slope mean (attempts) [95% CI] | slope SD (tokens) | share slope > 0 / < 0 | slope by relative position "
          "first -> last bin |", "|---|---|---|---|---|---|---|---|---|"]
    for m, r in R.items():
        s = r["scores"]
        w, g = s["w1_mean"], s["gap_mean"]
        k = len(w) // 2
        rb = [x for x in s["slope_by_rel_bin"] if x is not None]
        L.append(f"| {m} | {_f(w[0], 3)} / {_f(w[k], 3)} / {_f(w[-1], 3)} | {_f(float(np.mean(s['w1_sd'])), 3)} | "
                 f"{_f(g[0], 3)} / {_f(g[k], 3)} / {_f(g[-1], 3)} | {_f(s['slope_mean'], 3)} | "
                 f"{_f(s['slope_mean_attempt_weighted'], 3)} "
                 f"[{_f(s['slope_mean_attempt_boot_ci95'][0], 3)}, {_f(s['slope_mean_attempt_boot_ci95'][1], 3)}] | "
                 f"{_f(s['slope_sd'], 3)} | {_f(s['slope_frac_pos'], 2)} / {_f(s['slope_frac_neg'], 2)} | "
                 f"{_f(rb[0], 3)} -> {_f(rb[-1], 3)} |")
    L += ["", "Depth profile of mean w1 (relative depth deciles, mean of layers in each decile):", "",
          "| model | " + " | ".join(f"d{i}" for i in range(10)) + " |", "|---|" + "---|" * 10]
    for m, r in R.items():
        parts = [float(np.mean(p)) for p in np.array_split(np.array(r["scores"]["w1_mean"]), 10)]
        L.append(f"| {m} | " + " | ".join(f"{p:.3f}" for p in parts) + " |")
    L += ["", "Depth profile of mean boundary gap (same deciles):", "",
          "| model | " + " | ".join(f"d{i}" for i in range(10)) + " |", "|---|" + "---|" * 10]
    for m, r in R.items():
        parts = [float(np.mean(p)) for p in np.array_split(np.array(r["scores"]["gap_mean"]), 10)]
        L.append(f"| {m} | " + " | ".join(f"{p:.3f}" for p in parts) + " |")
    L += ["", "## 5. Class differences in path statistics", "",
          "Pairwise Jensen-Shannon distance (bits, sqrt of JSD) between class-conditional bigram matrices Q_c(a, b) of "
          "layer pair (l, l+1), mean over class pairs and layer pairs, vs 200 sentence-label permutations within "
          "absolute-position bin (classes with >= 1,000 sampled labelled tokens). `joint` = plan statistic (distance "
          "between the class joint matrices). By the chain rule JSD_joint = JSD_marginal + JSD_transition: `marginal` "
          "= class differences in which expert is used at layer l; `transition` = sqrt of the remainder, class "
          "differences in P(next-layer expert | this-layer expert). Plug-in distances are inflated by sparse counts; "
          "read the excess over the permutation null.", "",
          "| model | tokens | joint obs | joint null | excess | p | marginal excess | p | transition obs | "
          "transition null | excess | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in R.items():
        c = r["class_js"]
        if "joint" not in c:
            L.append(f"| {m} | {c.get('n_labelled_tokens')} | skipped: {c.get('skipped')} | | | | | | | | | |")
            continue
        j, mg, t = c["joint"], c["marginal_layer_l"], c["transition"]
        L.append(f"| {m} | {c['n_labelled_tokens']:,} | {_f(j['layer_mean_observed'], 4)} | "
                 f"{_f(j['layer_mean_null'], 4)} | {_f(j['layer_mean_excess'], 4)} | {_f(j['p_layer_mean'], 3)} | "
                 f"{_f(mg['layer_mean_excess'], 4)} | {_f(mg['p_layer_mean'], 3)} | "
                 f"{_f(t['layer_mean_observed'], 4)} | {_f(t['layer_mean_null'], 4)} | "
                 f"{_f(t['layer_mean_excess'], 4)} | {_f(t['p_layer_mean'], 3)} |")
    L += ["", "Share of the layer pairs whose class-difference excess is significant at p < .05 (200 permutations, "
          "unadjusted): "]
    for m, r in R.items():
        c = r["class_js"]
        if "joint" in c:
            L.append(f"- {m}: joint {np.mean(np.array(c['joint']['p']) < .05):.2f}, marginal "
                     f"{np.mean(np.array(c['marginal_layer_l']['p']) < .05):.2f}, transition "
                     f"{np.mean(np.array(c['transition']['p']) < .05):.2f}")
    L += ["", "Class-conditional slope and layer-mean w1 (classes with >= 100 sampled tokens):", ""]
    for m, r in R.items():
        bc = r["scores"]["by_class"]
        L.append(f"- {m}: " + "; ".join(f"{k} slope {v['slope_mean']:+.3f}, w1 {v['w1_layer_mean']:.3f} "
                                        f"(n={v['n_tokens']})" for k, v in bc.items()))
    L += ["", "Largest class-pair transition excess (transition distance minus permutation mean, layer-mean):", ""]
    for m, r in R.items():
        c = r["class_js"]
        if "pair_matrix_transition_excess" not in c:
            continue
        ex = np.array(c["pair_matrix_transition_excess"], float)
        names = c["classes"]
        pairs = sorted(((ex[i, j], f"{names[i]}-{names[j]}") for i in range(len(names)) for j in range(i + 1, len(names))),
                       reverse=True)[:3]
        L.append(f"- {m}: " + ", ".join(f"{n} {v:+.4f}" for v, n in pairs))
    L += ["", "Notes: tokens are sampled uniformly within an attempt, so the statistics are token-level (an attempt "
          "with many tokens contributes at most 4,096). Nemotron's routed-layer list has 23 of its layers (hybrid "
          "model), so `adjacent` layers are adjacent in the routed list. Boundary gaps are in each family's own "
          "selection-score space and are not comparable across models. Per-layer arrays and all null summaries are "
          "in the per-model JSON files.", ""]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT / 'summary.md'}")

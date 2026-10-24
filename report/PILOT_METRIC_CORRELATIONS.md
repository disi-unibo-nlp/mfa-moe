# Qwen3.6 pilot: original routing metrics and reasoning-class dynamics

Both CPU analyses completed successfully: job 58456386 in 2m07s and job 58457518 in 1m50s. No new GPU replay was performed.

The pilot contains 64 distinct MATH-500 questions, one saved attempt each: 39 correct (60.94%), 25 incorrect, including two capped attempts. All 30,187 labeled sentences, 554,738 completion tokens and 640,800 routing windows were retained. This is a pilot result, not a six-benchmark evaluation.

## Main finding

Routing is associated with reasoning class, but this pilot does not establish a routing metric that predicts correctness after correction for the searched comparisons. No original/new routing-correctness family, class-conditioned correctness family, prefix family, or class-dynamics correctness family contains a BH q < 0.05 result. This is a lack of sufficient evidence at this sample size, not proof that all relationships are zero.

Full-trace correlations weaken substantially when controlling for log completion length and caps. These are retrospective sensitivity checks; eventual length and caps must not become inputs to an online policy. The question, not the window or sentence, is the sampling unit.

## Original routing metrics

These use the exact saved full-reasoning and class-view definitions from the existing pipeline, including gap-aware adjacency. Normalized confidence represents entropy without separately testing the same affine transformation. Selected mass and the top-k/non-top-k gap are also affine-equivalent at fixed expert count/top-k; they are retained for continuity and are not independent evidence.

| Metric (mean across 40 layers) | Raw r | Raw 95% CI | Adjusted r | Adjusted 95% CI | Joint raw BH q |
|---|---:|---|---:|---|---:|
| Normalized confidence | +0.257 | [+0.010, +0.485] | -0.078 | [-0.318, +0.177] | 0.188 |
| Top-1 probability margin | +0.202 | [-0.044, +0.442] | -0.246 | [-0.493, +0.006] | 0.213 |
| Selected expert mass | +0.238 | [-0.015, +0.459] | -0.031 | [-0.298, +0.226] | 0.196 |
| Selected/unselected boundary margin | +0.204 | [-0.062, +0.428] | +0.011 | [-0.254, +0.258] | 0.213 |
| Top-k/non-top-k gap | +0.238 | [-0.015, +0.459] | -0.031 | [-0.298, +0.226] | 0.196 |
| Top-1 switching | -0.155 | [-0.367, +0.093] | -0.048 | [-0.268, +0.199] | 0.343 |
| Adjacent top-k overlap | +0.210 | [-0.043, +0.426] | +0.138 | [-0.127, +0.384] | 0.213 |

Hidden-state norm, hidden trajectory distance and hidden/router geometry are unavailable: this router-only pilot did not save hidden states. No scalar routing proxy is substituted.

## New temporal routing metrics

All-layer means of trailing 64/256-token windows include lagged overlap (1/4/16/64), usage-preserving shuffled-overlap controls, turnover, full-router and selected-mixture entropy, entropy differences, JSD and local shuffled JSD, effective expert counts and native selection gaps. Weak-boundary fractions remain unavailable here because this pass does not fit train-only gate thresholds.

| Metric | Raw r | Adjusted r | Joint raw BH q |
|---|---:|---:|---:|
| full/all/W64/overlap_excess_lag1 | +0.343 | +0.215 | 0.117 |
| full/all/W64/full_jsd | +0.325 | +0.132 | 0.117 |
| full/all/W256/overlap_excess_lag64 | -0.322 | -0.264 | 0.117 |

At the 1,024-completion-token landmark, lag-1 excess overlap has r = +0.432, 95% CI [+0.230, +0.593], q = 0.059 (62 eligible questions). It is suggestive, not corrected-significant, and conditions on reaching that landmark. Prefix features never include later routing. No nested held-out prediction is claimed.

## Associations with reasoning classes

For each question, compare windows ending in a class with windows ending in other classes, then average question differences. The table uses 256-token windows and equal layer averages. A window may cross class boundaries; this is an endpoint-class association, not an isolated class-only causal effect. Position, text and question difficulty may explain part of these differences.

| Endpoint class | Supporting questions | Effective-expert-count difference | 95% CI | BH q |
|---|---:|---:|---|---:|
| Read | 53 | +3.642 | [+1.998, +5.218] | 0.005 |
| Analyze | 57 | -2.716 | [-3.855, -1.580] | 0.005 |
| Plan | 62 | -0.005 | [-1.877, +1.820] | 0.987 |
| Implement | 64 | -1.149 | [-2.028, -0.302] | 0.020 |
| Explore | 43 | +2.987 | [+1.651, +4.255] | 0.005 |
| Verify | 58 | +2.240 | [+0.773, +3.611] | 0.005 |
| Monitor | 59 | +3.216 | [+1.570, +4.780] | 0.005 |

Effective expert count is exp(entropy of the window-average executed mixture). It describes diversity across a window; native top-k remains eight per token.

Analyze windows also show higher lag-16 overlap (+0.0090, about +0.90 percentage points; q = 0.0051). Explore windows show lower overlap (−0.00583, about −0.58 percentage points; q = 0.0166). Monitor windows show higher selected-mixture JSD (+0.0107 nats; q = 0.0166). These are descriptive signatures and do not establish beneficial intervention directions.

Across all 336 new class/metric/window contrasts, 188 have BH q < 0.05. Many metrics are dependent transformations or related summaries, so this does not mean 188 independent discoveries. Original class-view contrasts are also saved; their estimand is class minus all reasoning, not class versus rest.

## Class dynamics and inference limits

Transitions, stay/exit rates, transition entropy, observed runs, return intervals and censor-aware exit hazards were analyzed. Class-specific dynamics were also correlated with correctness. None survives its BH family correction. Observed run averages can include censored runs and must not be interpreted as uncensored dwell-time estimates. Motif mining, expert-specific rankings and nested next-class prediction are not part of this compact recovery pass.

Intervals use 1,000 question-bootstrap resamples; adjusted intervals refit the length/cap regression in each resample. Correlation p-values use the conventional t reference, with BH correction per declared family. Original and new global routing metrics additionally receive a joint correction. Separate family definitions and every tested estimate are available in the machine-readable tables.

Generation used FP8, while this pilot replay used NF4 with BF16 compute/routers. These observations therefore require FP8-path confirmation before defining a primary intervention policy. They provide hypotheses about class-linked routing behavior, not a demonstrated accuracy/token improvement.

![Original metrics and class associations](../results/pilot-metrics/pilot-findings.png)

[Download the summary figure as PDF](../results/pilot-metrics/pilot-findings.pdf).

## Verification and artifacts

12 targeted tests passed (six correlation/statistical tests and six existing class-dynamics/alignment/causality tests). An independent scalar Pearson calculation matched the reported overlap correlation within 1e-12. Checks confirmed 64 distinct questions, exact window/label coverage and retention of both caps. Input file hashes match the accepted pilot artifact; per-trace checkpoints support inspection and resume.

- [Combined correlation table](../results/pilot-metrics/original-v1/combined_correlations.jsonl)
- [New class-metric contrasts](../results/pilot-metrics/correlations-v1/class_metric_contrasts.jsonl)
- [Original class-view contrasts](../results/pilot-metrics/original-v1/class_metric_contrasts.jsonl)
- [Transition rates](../results/pilot-metrics/correlations-v1/transition_rates.jsonl)
- [Censor-aware exit hazards](../results/pilot-metrics/correlations-v1/exit_hazards.jsonl)
- [Correlation plot (PNG)](../results/pilot-metrics/correlations-v1/correlations.png)
- [Correlation plot (PDF)](../results/pilot-metrics/correlations-v1/correlations.pdf)
- [Verification](../results/pilot-metrics/verification.json)

## Post-hoc class-exchange follow-up

Question-weighted correct/incorrect comparisons for specific transitions are now
available in [PILOT_REASONING_TRANSITIONS.md](PILOT_REASONING_TRANSITIONS.md).
These exploratory comparisons do not survive BH 5%.

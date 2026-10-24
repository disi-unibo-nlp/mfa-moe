# Intermediate findings from completed GPT and GLM analyses

Inspected 2026-09-24T10:23:48.604420+00:00. GPT: 1,580 unique captured attempts across separate label cohorts, 24 routed layers. GLM: 1,390 attempts, 46 routed layers. The examples below all use their respective stratified label cohorts.

These are selected exploratory associations, not validated prediction rules or causal effects. Correlations are calculated at question level and adjusted for log completion length and truncation/cap status. BH q-values are corrected within each metric family, benchmark, cohort and adjustment—not globally across this table. Related layer, window and duplicate primary/supplement tests are not independent discoveries.

| Model | Benchmark | Metric | Questions | Adjusted r | BH q | 95% question-bootstrap interval |
|---|---|---|---:|---:|---:|---|
| gpt | olympiad | `class_token_fraction/Monitor` | 642 | -0.165 | 0.000194 | [-0.24138018301734127, -0.08418196322087777] |
| glm | olympiad | `class_token_fraction/Monitor` | 539 | -0.337 | 6.49e-15 | [-0.4042940262632579, -0.2729240139043592] |
| gpt | minerva | `temporal_complete_global/W64/overlap_excess_lag1` | 272 | -0.197 | 0.018 | [-0.3167784187860629, -0.06420154572759904] |
| glm | olympiad | `temporal_complete_global/W256/mixture_jsd_excess` | 530 | -0.134 | 0.00929 | [-0.21677644001528207, -0.0576591682535449] |
| glm | minerva | `temporal_complete/L15/W256/mixture_local_entropy` | 250 | +0.414 | 3.04e-08 | [0.31796595024104574, 0.5112796016719511] |
| gpt | math500 | `saved_full/router_confidence_l11` | 496 | -0.245 | 9.55e-06 | [-0.33033000259460105, -0.16023640508252973] |
| gpt | olympiad | `saved_full/router_boundary_margin_l09` | 642 | -0.236 | 8.14e-07 | [-0.3090396238508626, -0.15668220865427585] |
| gpt | math500 | `class_token_fraction/Explore` | 496 | +0.155 | 0.00386 | [0.09126460989128038, 0.21422831589859684] |

Interpretation:

- A greater share of labeled reasoning tokens in Monitor is associated with lower correctness on Olympiad in both models. This may reflect difficulty or unsuccessful progress; it does not show monitoring causes errors.
- GPT on Minerva shows lower adjacent-token expert overlap on correct answers even after subtracting the within-trace shuffled baseline, averaged across layers (64-token windows). This provides a modest temporal association beyond marginal expert frequencies.
- GLM on Olympiad shows smaller changes in expert-weight distributions between window halves on correct answers, after subtracting the shuffled baseline (256-token windows). This is a different temporal property from adjacent-token set turnover, so a universal “more switching is better” claim is unsupported.
- GLM Minerva layer 15 has higher executed-mixture entropy on correct answers: weights are more evenly spread among selected experts. This selected layer result does not establish an all-layer effect.
- Higher GPT router confidence or selected/unselected boundary separation does not universally indicate correctness; the displayed MATH500/Olympiad examples have negative associations.
- GPT Explore token fraction is positively associated with correctness on MATH500, but the same pattern is not significant on its Olympiad or Minerva cohorts.

Small-benchmark check: neither model has an adjusted BH-significant result on AIME24 in these stratified primary or supplemental analyses. AIME25 has a few selected results based on only 23–30 questions; these are fragile exploratory findings. GLM AMC23 Verify sentence fraction is positive (r=0.461, q=0.0219, n=40).

No comparable accepted production findings are available yet for Gemma, Nemotron or the remaining Qwen sources. Gemma preparation completed with exact tokenizer coverage on 1,642 attempts / 14,819,371 completion tokens; its GPU validation is running.

Exact source rows and file hashes are retained in intermediate-findings-20260924.json.

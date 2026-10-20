# W-0002: Whole nomination and an actual GPT control shortfall
- date: 2026-10-20
- commit: 385ea15f395e4b29a4919f545603386e2456ddce
- source: report/whole-reasoning-20261020-r2/GPT_CONTROL_WALKTHROUGH.json; gpt/WHOLE_NOMINATIONS.json, fold 0 / Read
- paths: src/moe_exp/final_routing/selection.py, src/moe_exp/study/whole.py
- keywords: whole, risk difference, quintile, feasibility, controls

**Input:** The actual GPT Read nomination for held-out fold 0, derived in CPU61566298 from discovery folds 1, 2, 3, 4. No whole-arm outcome exists.

## 1. Global nomination (`selection.global_pairs`)

The full fit and all four inner refits retain 363 positive and 322 negative sign-stable pairs, with support at least two families on both sides. Fold 0 contributes to none of these fits.

## 2. Target selection (`whole.select_pairs`)

Promote requests 10 activations and 50 deactivations. The activation cap is one per layer; the deactivation cap is ceil(50/24) = 3. Exact counts are available, so sign-support availability passes.

## 3. Match construction (`selection.layer_matched_controls`)

At layer 14, rank-based native-frequency quintile 4 contains [9, 20, 3, 26, 13, 25]. The targets deactivate [3, 25, 20] in that quintile. The current targets and their opposite-polarity exclusions remove [20, 3, 13, 25]. Only [9, 26] remain for three required distinct controls. Layer 20 has a further shortfall.

## 4. Observed result

CPU61566298 exits FAILED 1:0 after 43 seconds on `whole control pool exhausted ... /deactivate/14/4`. Its pending canaries and supervisor were cancelled without allocation. Every cost is preserved. An exhaustive outcome-free check finds all 35 GPT full-dose promote cells infeasible under quintiles, while all 20 suppress cells pass. Tertiles make 35/35 promote cells feasible; that scientific amendment requires the user's decision. The sampler has not weakened matching, reused an excluded pair, selected half dose, or read an outcome.

## Worth noticing

Sign-stable target availability does not guarantee a disjoint matched random set. This is a finite expert-pool constraint that must be resolved before inference, independent of efficacy or fluency.

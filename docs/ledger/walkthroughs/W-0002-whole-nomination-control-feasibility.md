# W-0002: Whole nomination and an actual control-feasible GPT cell
- date: 2026-10-20
- commit: 16cf59973f27a066d6f1a47a620d02c5c519928e
- source: report/whole-reasoning-20261020-r3-batched/deterministic/gpt/WHOLE.json; WHOLE_NOMINATIONS.json; NOMINATION_AUDIT.json; fold0 / Read / promote / main
- paths: src/moe_exp/final_routing/selection.py, src/moe_exp/study/whole.py
- keywords: whole, risk difference, quintile, feasibility, controls

**Input:** The actual GPT Read nomination for held-out fold0, rebuilt in
CPU61581053 from discovery folds1,2,3,4. No whole-arm outcome was read.

## 1. Global nomination (`selection.global_pairs`)

The full fit and all four inner refits retain 363 positive and 322 negative sign-stable pairs, with support at least two families on both sides. Fold 0 contributes to none of these fits.

## 2. Capped ranking (`selection.select_pairs`)

Promote requests10 activations and50 deactivations. The activation cap is one
per layer; the deactivation cap is ceil(50/24)=3. The exact-rank deactivation
set has total absolute RD5.992919759516524.

## 3. Reserve distinct controls (`selection.select_pairs`)

The r2 exact-rank set exhausted layer14/quintile4: [9,20,3,26,13,25]. Targets
[3,25,20] and the opposite exclusion13 left only [9,26] for three controls.
The v3 greedy walk reserves both role demands and skips a target that cannot
leave a distinct control. A new target consumes its own available pool slot
plus a control; an already excluded target consumes only the control.

## 4. Match the selected roles (`selection.layer_matched_controls`)

The actual layer14 target deactivations are [3,25]; their distinct controls
are [9,20]. Across the cell, two deactivation pairs change, all10 activations
remain unchanged, and all50 deactivations have controls under the same
quintiles and exclusions. Retained absolute RD is5.933801208792859,
fraction0.9901352674329091. Mean native frequency across steered experts is
0.361816186798991 for target and0.2953428770321593 for random.

## 5. Allocated qualification (`whole.audit_nominations`)

CPU61581053 completed0:0 after293seconds. All190 study tests passed; the
rebuilt manifests match the saved DP targets and controls in all70 GPT
promote cells across both doses. Every one of the210 model/dose cells is
qualified. Both doses retain3096 GPT and2838 Qwen executable rows. The prior
95d4cb0 and3a848b4 qualifications and every receipt remain preserved.

## Worth noticing

Control capacity changes target nomination before outcomes. The original r2
failure, source and proposal remain in the r2 report. The accepted v3 rule
retains quintile matching and makes its RD loss explicit; targets still have
the larger native-frequency perturbation. Allocated CPU qualification does
not establish GPU canary success or an efficiency result.

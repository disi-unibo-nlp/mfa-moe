# W-0011: Outcome-selected steering, the GPT-OSS expert set
- date: 2026-10-24
- commit: 308969e40638d01287c016aa5d61306f00fc57c4
- source: report/outcome-steering-20261022-r1 (gpt/OUTCOME_SELECTION.json sha256 6a283ef2, gpt/WHOLE.json, analysis/EFFECTS_OUTCOME.json 9e0f3f73, analysis/DECISION.json), the per-attempt usage /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs/outcome-steering/gpt/USAGE.{json,npz} (usage sha256 afcca894) and the class nominations of report/whole-reasoning-20261021-r7; every value below comes from a run on 2026-10-24 of the code at this commit on those files
- paths: src/steering/outcome/select.py, src/steering/outcome/manifest.py, src/steering/outcome/decide.py, scripts/steering/outcome_usage.sh, tests/steering/outcome/test_select.py
- keywords: outcome, nudge, expert selection, within-question z-score, extension rule, D-0003

**Input:** GPT-OSS-20B in the first wave of the outcome-selected gentle-steering experiment (feature F-0004,
decision D-0003): the experts that its wrong attempts use more than its right attempts of the same question,
chosen from the 276 cohort-B attempts of 37 AIME and AMC questions. `scripts/steering/outcome_usage.sh`
(a CPU job running `python -m steering.outcome usage`) made the usage rates; the steps below rebuild the
selection from them, the rows that use it, and the declared decision.

## 1. Usage per attempt (`select.usage`, read here from its receipt)

Every cohort-B capture is reduced to a rate: the fraction of its reasoning tokens that selected each (layer,
expert). `USAGE.npz` holds `rates` of shape (276, 24, 32), float32; each layer of an attempt sums to 4.0, the
native top-k. The outcome of each attempt is the re-scored strict correctness: 147 right, 129 wrong, 0 unknown.
All 37 questions have both a right and a wrong attempt. The usage receipt hash matches the one the selection
records (afcca894).

## 2. Protected experts (`select.protected_pairs`)

Two groups are never nudged. The shared core is the top 5 percent of experts by native frequency in every layer
(2 of 32 per layer, 48 pairs). The execution-specific experts are the top 10 percent of the positive Implement and
Analyze candidates of the class nominations (71 pairs). Their union is 86 pairs; the saved selection records the same
three numbers.

## 3. Scores and stability (`select.question_scores`, `select.stable_scores`)

Within each question the usage of every (layer, expert) is z-scored across that question's attempts, so difficulty
and length cannot leak in. The score is the mean over questions of (mean z of the wrong attempts) minus (mean z of
the right ones). A pair is stable when its sign survives leaving out any one question (37 refits).
For the set built on all questions, 349 pairs have a stable positive score.

## 4. The chosen set (`select.choose`, `select.selection`)

Stable positive pairs outside the protected group are taken in score order until they carry 1.2 percent of native
routing (between 5 and 40 pairs), each only if it still has a feasible matched control from
`steering.selection.layer_matched_controls`. For the set `all`:

| | Value |
|---|---|
| target pairs / matched random controls | 20 / 20 |
| share of native routing slots: targets / controls | 0.01208 / 0.01319 |
| unstable fills | 0 |
| strongest pairs (layer, expert, score) | (9, 13, 0.4992), (10, 14, 0.4722), (20, 30, 0.4572), (14, 11, 0.4289), (12, 12, 0.4244) |
| layers holding targets | 1, 3, 4, 5, 9, 10, 11, 12, 14, 17, 19, 20 |

Evaluation families whose question is in cohort B get a set selected without it: 13 of the 129 evaluation
questions are in cohort B, giving 14 sets in all. The set without `aime24|aime24_65` has 19 targets and shares 17 with
`all`. The rebuilt selection (all 14 sets, the protected counts and the set id 7e70103b) equals the saved
`OUTCOME_SELECTION.json` field by field.

## 5. Rows (`manifest.nudge_actions`, `manifest.outcome_manifest`)

Each evaluation family gets a `target_nudge` and a `random_nudge` row at delta 0.4, 0.8 and 1.6, and a length-limit
`budget` row at 50, 70 and 85 percent of its native reasoning. A nudge is one `score_shift` action per layer. The first
`target_nudge` row of `gpt/WHOLE.json` has delta 0.4, expert set `all` and 12 actions, the first being
`{layer: 1, shift: [7], delta: 0.4, operator: score_shift}`: expert 7 of layer 1 has 0.4 subtracted from its router
logit at every prompt and reasoning token, and the router still takes its own top-4 (`operators.score_shift_masked`).

## 6. Decision (`decide.qualifying`, `decide.decide`)

The extension rule was fixed before any outcome (D-0003): extend when a nudge is shorter than matched random experts
with the interval above zero and accuracy against native not below -1 point, or more accurate than random with the
interval above zero. Applied to the saved `EFFECTS_OUTCOME.json` it finds no qualifying cell, as
`DECISION.json` records (`extend` False). GPT-OSS, 129 families:

| Delta | Tokens, target vs random (log scale, positive = shorter) | Accuracy, target minus random | Accuracy, target minus native |
|---|---|---|---|
| 0.4 | +0.006 [-0.131, 0.143] | -0.006 [-0.096, 0.085] | +0.017 |
| 0.8 | -0.080 [-0.269, 0.109] | +0.028 [-0.059, 0.115] | +0.011 |
| 1.6 | -0.110 [-0.587, 0.368] | -0.069 [-0.192, 0.053] | -0.039 |

Intervals are the simultaneous 95 percent ones. Native accuracy is 0.736 at a mean of 4,118 reasoning tokens; the target
nudge at 0.4 gives 0.736 and 4,078 tokens. Tests: `tests/steering/outcome` 40 passed (3.9 s, including the r2 tests).

## Worth noticing

- The whole selection reruns in 3 s on the CPU because the expensive part, reading 276 captures, is a separate
  step with its own receipt.
- Every interval includes zero. The 0.4 nudge reproduces native behaviour almost exactly (accuracy 0.736, 4,078
  against 4,118 tokens), and the larger nudges change neither tokens nor accuracy beyond the intervals. The null
  result was recorded under a rule fixed in advance.
- Protecting 86 of the 768 (layer, expert) pairs (11 percent) keeps the shared core and the execution experts out of
  both the targets and the matched controls, before any score is looked at.
- Only 13 of the 129 evaluation questions appear in cohort B, so most families use the set built on everything and
  the rest use a leave-one-question-out set.

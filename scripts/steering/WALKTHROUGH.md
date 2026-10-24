# steering: walkthroughs

## whole-nomination-trace: Whole nomination and an actual control-feasible GPT cell
- commit: 308969e40638d01287c016aa5d61306f00fc57c4
- source: report/whole-reasoning-20261020-r3-batched/deterministic/gpt/WHOLE_NOMINATIONS.json (sha256 11e2324f), gpt/WHOLE.json (8b78678e), gpt/WHOLE_HALF.json (99764d19), qwen36/WHOLE.json (83a966ae), NOMINATION_AUDIT.json (215cdfb7), the DP reference CONTROL_FEASIBLE_RD_BOTH_DOSES.json (e3577e2f); every value below comes from a run on 2026-10-24 of the code at this commit over those saved files
- code: src/steering/selection.py, src/steering/nominations.py, src/steering/whole.py, tests/steering/test_whole.py

**Input:** The GPT-OSS-20B Read nomination of the whole-reasoning release for held-out fold 0 (main dose,
promote), rebuilt from discovery folds 1, 2, 3 and 4 only. No whole-arm outcome is read anywhere in this
trace. This is the retrace of the walkthrough first written on 2026-10-20 at commit 16cf599, when the code
was `final_routing.selection` and `study.whole`; the modules are now `steering.selection` and `steering.whole`.

### 1. Global nomination (`nominations.discovery_inputs`, `selection.global_pairs`)

The discovery inputs are 1,547 GPT attempts of 1,486 question families: a counts array
`[attempt, class, layer, expert]` of shape (1547, 9, 24, 32) and token denominators (1547, 9). A pair
(layer, expert) is a candidate for a class when its risk difference between the class and all other
classes has support from at least two families on both sides and keeps its sign in the full fit and in all
four inner refits that leave one more fold out.

| Fold 0, class Read | Value |
|---|---|
| training folds | [1, 2, 3, 4] |
| sign-stable positive pairs / negative pairs | 363 / 322 |
| strongest positive pair | layer 8, expert 29, +0.18327 |
| strongest negative pair | layer 9, expert 15, -0.19581 |
| activations and deactivations requested (GPT) | 10 and 50, both available and qualified |

`global_pairs` over all five folds and seven classes returns 35 records, and they equal the saved
`WHOLE_NOMINATIONS.json` records exactly (`True`).

### 2. Capped ranking (`selection.select_pairs`, no controls)

`whole.dose_counts('gpt', ..., 'main', 'promote')` gives (10, 50). The activation cap is one pair per layer,
the deactivation cap is ceil(50/24) = 3. Walking the candidates in order of absolute risk difference, the
exact-rank deactivation set has total absolute RD 5.992919759516524. The opposite-polarity exclusion (the
10 negative activations and 50 positive deactivations the other direction would pick) has 60 pairs.

### 3. Reserve distinct controls (`selection.select_pairs` with `frequency=` and `excluded=`)

Controls must come from the same layer and the same native-frequency quintile as their target. At layer 14
the most frequent quintile holds the experts [9, 20, 3, 26, 13, 25]. The exact-rank deactivation targets at
that layer are [3, 25, 20] and the opposite exclusion there is [4, 13, 28], so expert 13 is also taken.
That leaves only [9, 26] to serve three targets. The greedy walk charges a new target its own pool slot plus
one distinct control and skips the target that cannot leave one.

### 4. Match the selected roles (`selection.layer_matched_controls`, called by `whole.cell_actions`)

| Layer 14, deactivation | Experts |
|---|---|
| targets after the walk | [3, 25] |
| matched random controls | [9, 20] |

Across the cell, 2 deactivation pairs change against the exact-rank set, all 10 activations are unchanged
(`activation_changed` False) and all 50 deactivations have controls under the same layer, quintile and
exclusions. Retained absolute RD is 5.933801208792859, a fraction of 0.9901352674329091. Mean native
frequency of the steered experts is 0.361816186798991 for the targets and 0.2953428770321593 for the random
controls. The rebuilt cell equals the cell stored in `gpt/WHOLE.json` (`True`).

### 5. Qualification of the whole release (`whole.cell_actions` for every cell, the comparisons of `whole.audit_nominations`)

All cells of the deterministic release were rebuilt from the saved nominations and compared with what the
release stored. For GPT the nominations themselves were recomputed too; for Qwen3.6 the saved candidate
lists were used (recomputing them did not finish inside the 550 s wall limit of the first, login-node run).

| Check | GPT-OSS | Qwen3.6 |
|---|---|---|
| cells rebuilt (2 doses x 5 folds x classes) | 110 | 100 |
| qualified | 110 | 100 |
| equal to the cell in the saved manifest | 110 | 100 |
| equal to the record in `NOMINATION_AUDIT.json` | 110 | 100 |
| promote cells whose targets and controls equal the saved DP reference | 70 of 70 | not in the DP reference |
| rows planned = executable, per dose | 3,096 | 2,838 |

The unit tests `tests/steering/test_whole.py` and `tests/steering/test_stream_batch.py` ran in a CPU job
in the vLLM environment: 33 tests, OK, 14 s. They include the sign-survival, half-availability and
control-matching tests behind steps 1 to 4.

### Worth noticing

- The new code reproduces every nomination receipt of 2026-10-20 without a difference: 35 GPT records, 210
  cells and 70 DP matches. The move from `final_routing` and `study` into `steering` changed no result.
- Control capacity decides the nomination before any outcome. In this cell expert 20 is replaced to keep
  distinct controls, at a cost of 1 percent of the retained RD; the lowest retained fraction over the GPT main promote cells is 0.9426.
- Targets still sit on more frequently used experts than their controls (0.362 against 0.295), so the
  matching is by quintile, not by value.
- The release's original allocated qualification (job 61581053, 190 tests) is a Slurm stage of every release
  (`python -m steering.whole validate`); the CPU-job run above is a spot check of the same code, not a replacement.

## whole-canary-trace: One saved whole-routing canary unit
- commit: 308969e40638d01287c016aa5d61306f00fc57c4
- source: report/whole-reasoning-20261020-r3-batched/deterministic/stores/gpt/main/canary/d110fdc51c1f33901e79e28217594b4654f1ae3a69f15a03aee8435067a09618/ (AUDIT.npz, CAPTURE.json, OUTPUT.json, PARTIAL.json, RESULT.json), gpt/WHOLE.json (8b78678e), gpt/CANARY.json, GPT_SOFTWARE_FAILURE.json; native job 61581237; every value below comes from a run on 2026-10-24 of the code at this commit over those saved files
- code: src/steering/whole.py, src/steering/pulse.py, src/steering/stream.py, src/steering/operators.py, scripts/steering/gpu.sh, tests/steering/test_stream_batch.py

**Input:** The saved unit of the actual Minerva family
60e8b726e6566124e257983eb62d37173a5df88f0c7196d6f76052195338f2ed, fold 4, arm `target_suppress`, class
Read, from the GPT canary of the whole-reasoning release (job 61581237). Its original prompt has 191 tokens,
its seed is 2186601647 and its allowance 130,881. The unit is a canary: it never enters a production
analysis. This is the retrace of the walkthrough first written on 2026-10-20 at commit 16cf599, when the
modules were `study.whole`, `study.driver`, `study.stream` and `final_routing.operator`.

### 1. Admission (`whole.canary_rows`, `pulse.intervention_builder`, `pulse.Engine.admit`)

`canary_rows` on the saved GPT main manifest picks 8 families (one per benchmark plus two more) and, per
family, the two native arms and the Verify promote and Read suppress cells: 48 rows, of which 32 steer.
Today's row for this unit has uid e70dffcb, the saved unit has uid d110fdc5. The only difference is the
audit flag: the 2026-10-20 canary forced `audit_capture` on every row, today only the 32 production units
chosen in advance carry it (32 of 3,096). Setting the flag on the manifest row gives uid d110fdc5 and the
assignment hash 1949593e that `RESULT.json` stores.

| Row of the unit | Value |
|---|---|
| layers with an action | 20 (all but 0, 2, 3 and 19), one deactivated expert each, no activation |
| layer 1 | deactivate expert 17 |
| prompt, from `prompt_prefix` | 191 tokens, its digest equals `prompt_sha256` |
| seed of segment 0, `max_tokens` | 2186601647, 130,881 (the whole allowance) |
| closure marker | token ids [200005, 17196, 200008] = `<|channel|>final<|message|>` |

The builder returns the request that `Engine.admit` registers on both tensor-parallel ranks (fields
`uid, prompt, prefix, closure, actions, capture_layers, window, whole, prompt_scope, audit_capture`). The
engine itself needs vLLM and was not run: `PARTIAL.json` on disk records the same seed, `max_tokens`, history 0
and job 61581237.

### 2. Scope (`stream.Clock.observe`)

The clock sees every position the engine runs. Replaying the 191 prompt tokens and the saved 805 output
tokens (the last one is never fed, so 995 positions) gives one pass, `closed` True, and the closure marker
completes at generated position 738. The hook keeps every prompt row and every generated row before
closure: 191 + 738 = 929 rows. The stored layer-1 counter `active_positions` is also 929 (191 prompt, 738
generated), and the stored capture says `observed_tokens` 995, `passes` 1, `closed` True.

### 3. Layer execution (`StreamCapture._whole_batch`, `operators.score_override_dual_masked`)

For every row the layer applies the dual score override: the removed expert's log-score goes to the row
minimum minus 0.01, the router takes its own top-4 and renormalizes. The router logits are not kept, only the
audited top-4 ids and weights, so the operator cannot be re-run on this unit; the operator test
(`test_dual_matches_extrema_reference_and_preserves_inactive_bits`) passes instead. What the archive shows at
generated position 0 of layer 1:

| | ids | weights |
|---|---|---|
| native | [18, 29, 12, 6] | 0.5410026908, 0.2987700403, 0.0962416083, 0.0639856309 |
| executed | [18, 29, 12, 6] | 0.5410026908, 0.2987700701, 0.0962416083, 0.0639856383 |

Expert 17 was absent, so the membership is unchanged; renormalizing over a changed score vector moves the
weights in the eighth digit. Across the 738 stored generated positions of layer 1, native routing had expert
17 in the top 4 at 284 positions and the executed routing at none.

### 4. Counters (`stream.batched_dose`)

The additive counters of one request are summed from the layer's rows. Feeding the 738 stored generated rows
of layer 1 to `batched_dose` with the role "deactivate 17" gives the generated part of the stored totals:

| Counter, layer 1 | Recomputed from the archive (generated rows) | Stored in `CAPTURE.json` (prompt + generated) |
|---|---|---|
| active positions | 738 | 929 (191 + 738) |
| changed membership | 284 | 325 (so 41 are prompt rows) |
| native deactivated weight | 61.9238586 | 70.9533920 |
| executed deactivated weight | 0 | 0 |
| L1 distance of the weights | 123.8477249 | 141.9067688 |
| membership failures | 0 | 0 |

The archive holds no prompt rows, so those totals are checked in part. The stored finite and observation
counters are 739 on every layer; layers without an action (0, 2, 3, 19) store zero active positions. These
are same-state routing measurements, not effects on correctness or length.

### 5. Release and archive (`Engine.release`, `pulse.compress_audit_capture`)

The unit finishes after 805 output tokens with `stop`; `RESULT.json` says `complete`, one segment, no
stochastic redraw, and binds the raw output hash e53ba782 that `CAPTURE.json` repeats. The audit is a
compressed archive of 144 arrays (24 layers times 6 fields: positions int32, ids uint8, weights float32,
active bool), layer 1 covering generated positions 0 to 737. Its 620,456 bytes hash to c3caac87…3533588,
which equals `full_capture.sha256_file` in `CAPTURE.json`.

### Worth noticing

- A saved canary store cannot be matched by rebuilding the 48 rows with today's code: only 1 of the 48 saved
  uids is reproduced (a row whose audit flag was already on). With the flag forced on, all 48 are reproduced.
- The scope is checkable on a CPU: replaying saved tokens through `Clock` gives exactly the 929 rows the
  GPU job counted, with the closure at generated position 738.
- 41 of the 325 changed positions are prompt positions (the intervention covers the prompt); the audit
  archive keeps generated rows only, so the prompt part of each counter rests on the stored totals.
- This unit completed, but job 61581237 failed after 545 s because the capture of another unit
  disagreed between tensor-parallel ranks (`GPT_SOFTWARE_FAILURE.json`: 3 completed units, 45 partial). The
  later cleanup error (`unknown released request`) hides that first error. Three saved units do not qualify the
  48-row canary, and GPT production was withheld; the failure and every charge remain recorded.
- The tests of the batched counters (`tests/steering/test_stream_batch.py`, with `test_whole.py`) ran in a
  CPU job: 33 tests, OK, 14 s. The canary now starts with `sbatch scripts/steering/gpu.sh steering.whole
  canary --release report/NAME --model gpt`.

## outcome-selection-trace: Outcome-selected steering, the GPT-OSS expert set
- commit: 308969e40638d01287c016aa5d61306f00fc57c4
- source: report/outcome-steering-20261022-r1 (gpt/OUTCOME_SELECTION.json sha256 6a283ef2, gpt/WHOLE.json, analysis/EFFECTS_OUTCOME.json 9e0f3f73, analysis/DECISION.json), the per-attempt usage /leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs/outcome-steering/gpt/USAGE.{json,npz} (usage sha256 afcca894) and the class nominations of report/whole-reasoning-20261021-r7; every value below comes from a run on 2026-10-24 of the code at this commit on those files
- code: src/steering/outcome/select.py, src/steering/outcome/manifest.py, src/steering/outcome/decide.py, scripts/steering/outcome_usage.sh, tests/steering/outcome/test_select.py

**Input:** GPT-OSS-20B in the first wave of the outcome-selected gentle-steering experiment (feature outcome-steering-r1,
decision gentle-routing-nudge): the experts that its wrong attempts use more than its right attempts of the same question,
chosen from the 276 cohort-B attempts of 37 AIME and AMC questions. `scripts/steering/outcome_usage.sh`
(a CPU job running `python -m steering.outcome usage`) made the usage rates; the steps below rebuild the
selection from them, the rows that use it, and the declared decision.

### 1. Usage per attempt (`select.usage`, read here from its receipt)

Every cohort-B capture is reduced to a rate: the fraction of its reasoning tokens that selected each (layer,
expert). `USAGE.npz` holds `rates` of shape (276, 24, 32), float32; each layer of an attempt sums to 4.0, the
native top-k. The outcome of each attempt is the re-scored strict correctness: 147 right, 129 wrong, 0 unknown.
All 37 questions have both a right and a wrong attempt. The usage receipt hash matches the one the selection
records (afcca894).

### 2. Protected experts (`select.protected_pairs`)

Two groups are never nudged. The shared core is the top 5 percent of experts by native frequency in every layer
(2 of 32 per layer, 48 pairs). The execution-specific experts are the top 10 percent of the positive Implement and
Analyze candidates of the class nominations (71 pairs). Their union is 86 pairs; the saved selection records the same
three numbers.

### 3. Scores and stability (`select.question_scores`, `select.stable_scores`)

Within each question the usage of every (layer, expert) is z-scored across that question's attempts, so difficulty
and length cannot leak in. The score is the mean over questions of (mean z of the wrong attempts) minus (mean z of
the right ones). A pair is stable when its sign survives leaving out any one question (37 refits).
For the set built on all questions, 349 pairs have a stable positive score.

### 4. The chosen set (`select.choose`, `select.selection`)

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

### 5. Rows (`manifest.nudge_actions`, `manifest.outcome_manifest`)

Each evaluation family gets a `target_nudge` and a `random_nudge` row at delta 0.4, 0.8 and 1.6, and a length-limit
`budget` row at 50, 70 and 85 percent of its native reasoning. A nudge is one `score_shift` action per layer. The first
`target_nudge` row of `gpt/WHOLE.json` has delta 0.4, expert set `all` and 12 actions, the first being
`{layer: 1, shift: [7], delta: 0.4, operator: score_shift}`: expert 7 of layer 1 has 0.4 subtracted from its router
logit at every prompt and reasoning token, and the router still takes its own top-4 (`operators.score_shift_masked`).

### 6. Decision (`decide.qualifying`, `decide.decide`)

The extension rule was fixed before any outcome (gentle-routing-nudge): extend when a nudge is shorter than matched random experts
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

### Worth noticing

- The whole selection reruns in 3 s on the CPU because the expensive part, reading 276 captures, is a separate
  step with its own receipt.
- Every interval includes zero. The 0.4 nudge reproduces native behaviour almost exactly (accuracy 0.736, 4,078
  against 4,118 tokens), and the larger nudges change neither tokens nor accuracy beyond the intervals. The null
  result was recorded under a rule fixed in advance.
- Protecting 86 of the 768 (layer, expert) pairs (11 percent) keeps the shared core and the execution experts out of
  both the targets and the matched controls, before any score is looked at.
- Only 13 of the 129 evaluation questions appear in cohort B, so most families use the set built on everything and
  the rest use a leave-one-question-out set.

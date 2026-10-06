**Scope.** I audited only the supplied stdin packet, treated every file as data, and did not assume omitted artifacts are valid. Paths below are relative to `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo` unless the path is explicitly under `tools/tmp/`.

## Bottom line

The supplied implementation and evidence **cannot yet answer the objective**. The protocol itself says no intervention outcomes exist (`report/experimental-resume-v1/PROTOCOL_v0.1.md:3-4`). What exists is:

- observational expert-routing contrasts and proposed actions;
- prospective prices for a 13-family, 256-token micro-screen, explicitly below the registered 48-family discovery (`CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v2.json:22`);
- immediate binary semantic-rating machinery, not ordered seven-class trajectory measurement (`CAUSAL_ELIGIBLE_IMMEDIATE_SEMANTIC_RUBRIC_v1.md:31-35`);
- a batched neighbor qualification whose own zero condition shows routing/top-k divergence (`CAUSAL_BATCHED_NEIGHBOR_QUAL_AUDIT_v2.json:28-47`, `440-480`).

So the current packet can support at most a small exploratory engineering/measurement pilot after repairs. It does not support a causal claim about ordered reasoning-class trajectories, accuracy, or token-cost utility.

---

## Ranked findings

### 1. [PROVEN GAP] No causal intervention result exists yet

**Evidence:** `PROTOCOL_v0.1.md:3-4` says “No intervention outcomes exist for this study.” The price files are prospective (`CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v2.json:22`, `CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_PRICE_v3.json:22`), and the blind rating price is explicitly pre-generation (`CAUSAL_ELIGIBLE_DEACTIVATION_BLIND_RATING_PRICE_v3.json:23`). No generation, rating, accuracy-grading, or analysis result for the new routing study is supplied except observational/legacy artifacts.

**Why it matters:** Every causal claim about sparse routing edits changing reasoning trajectories is currently unsupported by executed outcomes.

**Shortest corrective check:** Execute and seal a frozen causal stage with explicit ITT receipts, or present the work as a protocol/feasibility study only.

---

### 2. [PROVEN GAP] The measurement pipeline does not measure ordered seven-class trajectories

**Evidence:** The protocol requires all seven classes and a 1,024-token trajectory endpoint (`PROTOCOL_v0.1.md:44-46`, `60-63`, `141-148`). The semantic rubric instead rates only `candidate_to_verify` and `approach_to_commit` as a binary target within a 256-token continuation (`CAUSAL_ELIGIBLE_IMMEDIATE_SEMANTIC_RUBRIC_v1.md:9-26`). The rating driver outputs only `target: bool` (`rate_eligible_immediate_semantics_v1.py:96-132`). The blind frame contains only the continuation text, not dense seven-class sentence labels (`build_eligible_micro_blind_frame_v2.py:156-170`). `analysis.py` contains trajectory/class helpers (`analysis.py:12-34`, `57-94`) but no supplied driver connects generated continuations to dense seven-class labels, correctness, or token receipts.

**Why it matters:** A binary “did the immediate next behavior happen” outcome is not an ordered reasoning-class trajectory. It also cannot measure answer accuracy or full token-cost utility from a 256-token generation cap.

**Shortest corrective experiment:** Freeze a dense seven-class sentence labeler for generated continuations, require token/offset/routed-array agreement, and apply `trajectory_completion`/`class_summary` at 1,024 tokens. Otherwise rescope the paper to immediate binary transition control.

---

### 3. [PROVEN BUG] The blind-rating driver cannot consume the supplied rating price

**Evidence:** `rate_eligible_immediate_semantics_v1.py:64-93` requires schema `eligible-immediate-blind-rating-price-v1`, status `PASS_COMPLETE_STAGE`, and fields `frame_sha256`, `frame_file_sha256`, `driver_sha256`, `gradeable_requests`, `ratings`, and `max_decode_tokens`. The supplied `CAUSAL_ELIGIBLE_DEACTIVATION_BLIND_RATING_PRICE_v3.json:2` has a different schema (`routing-eligible-deactivation-blind-rating-price-v3`), its status is `RESOURCE_PRICE_BEFORE_GENERATION...` (`:23`), and it instead has `assigned_continuations`, `maximum_ratings`, and `maximum_decode_tokens_total` (`:7-12`). With the supplied pair, validation raises at `rate_eligible_immediate_semantics_v1.py:92`.

There is also a cap mismatch: the driver sets `MAX_TOKENS = 1024` (`rate_eligible_immediate_semantics_v1.py:28`), while the supplied price budgets 256 decode tokens per rating (`CAUSAL_ELIGIBLE_DEACTIVATION_BLIND_RATING_PRICE_v3.json:11-12`).

**Why it matters:** The semantic rating stage cannot run or be priced correctly from the supplied artifacts.

**Shortest corrective check:** Create the required postgeneration-bound price file with the exact driver schema and frame hash, bind the judge output cap to the priced value, or update the driver to consume the v3 schema and reprice for 1,024-token generations.

---

### 4. [PROVEN GATE DEFECT] Batched neighbor qualification does not establish routing isolation

**Evidence:** In `CAUSAL_BATCHED_NEIGHBOR_QUAL_AUDIT_v2.json`, the zero/native condition already shows `first_token_unordered_topk_equal: false` in all eight paired family-seed comparisons (`:28-38`) and `target_mask_equal_on_same_text_prefix: false` in four of eight (`:37-38`, `440-447`). Edited conditions show same-text-prefix target-mask discordance of 6/8, 4/8, 5/8, and 2/8 (`:448-475`). Yet `audit_boundary_neighbor_qual_v2.py:121-130` defines `behavioral_sensitivity_gate_pass` using only first-token target-mask discordance and first-emitted-token discordance; it ignores same-text-prefix mask discordance and unordered top-k discordance. The artifact then sets `accepted_for_limited_batched_behavioral_discovery: true` (`CAUSAL_BATCHED_NEIGHBOR_QUAL_AUDIT_v2.json:477-480`).

**Why it matters:** If zero-condition runs already differ in top-k sets and later target membership on identical text, behavioral differences in edited conditions cannot be cleanly attributed to the sparse routing edit. The audit’s own interpretation says non-rejection cannot establish engine equivalence (`:480`).

**Shortest corrective check:** Either use the serial/eager profile for causal claims, or rerun the qualification with explicit zero-condition determinism gates: same-text-prefix top-k equality or a tight bound, target-mask equality, recompute equivalence, pulse-order checks, and more than four pilot families. Do not use the current `accepted_for_limited_batched_behavioral_discovery` flag as a causal qualification.

---

### 5. [PROVEN GAP] The serial qualification audit is omitted and unverified

**Evidence:** The packet explicitly omits `CAUSAL_MICRO_SERIAL_QUAL_AUDIT_v1.json` (bytes=110171, sha256=`284365cd...`). The serial price gate requires “serial 59200002 audit” (`CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v2.json:21`), and the serial driver admits the serial/eager profile was exercised in a separate four-family engineering qualification (`run_eligible_micro_serial_v2.py:3-4`). The qualification itself is not in the packet.

**Why it matters:** The claimed serial batch isolation, pulse, recompute, and native-routing qualification cannot be independently checked. The serial path is the cleanest fallback, and its qualification is therefore central.

**Shortest corrective experiment:** Supply and audit the omitted serial qualification on the final code tree, both TP ranks, with actual batch isolation, pulse timing, recompute, closure, native routing, inactive-row checks, and no sentinel active rows.

---

### 6. [PROVEN SCOPE/LEAKAGE RISK] The 13-family micro-screen is not the registered 48-family discovery and is discovery-internal

**Evidence:** The protocol requires 48 discovery families and says insufficient eligibility fails feasibility (`PROTOCOL_v0.1.md:24-31`, `106-109`). The micro price says maximum support is 13 globally distinct families and calls it “a labeled small causal screen, not validation” (`CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v2.json:22`). The action dictionary’s expert proposals come from the 48 discovery families (`CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json:28`, `101`), and the micro-screen driver restricts rows to the discovery pool (`run_eligible_micro_serial_v2.py:66-69`).

**Why it matters:** The 13-family screen uses the same discovery population that produced the expert proposals. It can be exploratory, but it cannot be called independent discovery or validation. Any positive result there is still observational-selection-adjacent.

**Shortest corrective check:** Report the 13-family screen as a pilot. For confirmation, use the registered mechanism/utility pools or a prospectively defined expansion with a pre-specified eligibility rule. Do not pool the 13-family screen with the registered 48/128/96 stages.

---

### 7. [PROVEN CODE RISK] The 13-family selection rule does not enforce per-transition coverage

**Evidence:** `run_eligible_micro_serial_v2.py:32-47` builds the chosen set by iterating `candidate_to_verify` first, then `approach_to_commit`, using a single global `used` family set. It only checks `len(chosen) == 13`. It does not require a minimum number of families for each transition. The random-balance check loops over `members` per transition (`:113-119`); if a transition has no rows, the counts are `[0,0,0,0]` and the check passes. The same selection code appears in `run_eligible_deactivation_serial_v1.py:33-48` and `run_eligible_deactivation_serial_v3.py:33-48`.

**Why it matters:** The screen could contain only `candidate_to_verify` families, leaving `approach_to_commit` unevaluated, while still passing manifest validation. The objective includes both transitions.

**Shortest corrective check:** Inspect the realized 13 rows, require a pre-specified minimum per transition, and fail feasibility for any unsupported transition rather than silently omitting it.

---

### 8. [PROVEN BUG] Detector version mismatch, candidate-sentence association bug, and display-math parsing bug

**Evidence:**

- `live_gate_v1.py:16` imports `StreamingTransitionDetectorV2` from `.transitions_v2`.
- `transitions_v2.py:15` identifies that detector as `prefix-transition-candidates-v2.1-discovery`.
- `transitions_v2.py:119-121` associates a complete candidate with the current sentence using an end-offset window with a four-character tolerance: `item.end >= end - len(sentence) - 4`. This can attach a candidate ending just before the current sentence to the current sentence. `transitions_v22.py:15` identifies v2.2 and fixes this with true containment: `start <= item.start and item.end <= end` (`transitions_v22.py:119-121`).
- `transitions_v21_frozen.py` is byte-identical to `transitions_v2.py` (same hash in the packet), so the frozen v2.1 file carries the same association logic.
- `build_eligible_micro_blind_frame_v2.py:20` references `TRANSITION_V22_FULL_PREFIX_START_FRAME.json`, so the rated frames are built from v2.2 data while the online gate uses v2.1.
- The display-math regex is broken in all three detector files: `transitions_v2.py:19-21`, `transitions_v21_frozen.py:19-21`, and `transitions_v22.py:19-21` contain `\\\[(.{1,240}?)\\\)`—the `\[` alternative closes with `\)` instead of `\]`. Math in `\[...\]` is therefore not detected as a math span.

**Why it matters:** Online trigger identity, sentence boundary, and transition coverage can differ from the frozen v2.2 frame used for semantic ratings.

**Shortest corrective check:** Make the live gate use v2.2, or port the containment condition into v2.1 and re-freeze. Fix the `\[...\]` regex. Add regression tests with adjacent candidate sentences and display-math expressions.

---

### 9. [PROVEN BUG] `dense_window` can count transitions beyond the horizon and can silently drop the first post-trigger sentence

**Evidence:** `analysis.py:37-54` validates only gaps between provided rows (`:47-49`) and returns any row with `token_start < horizon` (`:54`). It does not require `min(sentence_index) == triggering_sentence + 1`, and it does not require `token_end <= horizon`. Therefore a sentence that starts before 1,024 tokens but ends after 1,024 tokens can satisfy a transition, and a missing sentence immediately after the trigger can be silently omitted from the window. The docstring claims the transition must be “within the horizon” (`analysis.py:15-16`).

**Why it matters:** This can inflate apparent semantic success and biases the primary controllability endpoint.

**Shortest corrective check:** Require the trigger row and contiguous coverage from `triggering_sentence + 1`; either require full sentence containment within the horizon or freeze an explicit truncation rule. Add boundary tests for a sentence spanning token 1,024 and for a missing first post-trigger index.

---

### 10. [PROVEN CONSTRUCT GAP] The approach action targets a proxy, not the semantic “identified approach” transition

**Evidence:** The action dictionary itself says the approach proposal is an “Explore→Plan/Implement class proxy, not identified approach→semantic commitment” (`CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json:351-352`) and that the source population uses direct LLM seven-class labels (`:101`). The online gate discovery shows that among 35 lexical approach fires, Qwen two-reader acceptance is 12/35 and rejection is 21/35; among 48 nonfires, one Qwen accepted start was found (`APPROACH_ONLINE_GATE_DISCOVERY_v1.json:9-45`). Its interpretation states this stratified sample cannot estimate live recall (`:115`).

**Why it matters:** The selected expert may be associated with a broad class transition, not with the semantic criterion the paper claims to control. A null or positive intervention result would be hard to interpret as semantic approach control.

**Shortest corrective experiment:** Re-derive or validate expert-use contrasts against the frozen semantic rubric on discovery data, and report proxy-to-semantic agreement. If agreement is weak, scope claims to the proxy transition or choose a different supported transition.

---

### 11. [PROVEN RESOURCE GAP] Utility and accuracy measurement are unpriced and likely infeasible under the registered stage ceiling

**Evidence:** The utility scout is a 96-family, native-vs-policy, 16,384-token stage (`PROTOCOL_v0.1.md:135-139`). The worst-case utility decode cap is 6,291,456 tokens (`:190`). At the serial micro-screen stress rate of 8 tokens/s (`CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v2.json:13`), that is roughly 218 hours of decode on one GPU, or roughly 437 GPU-hours on two—far above the 4.75 GPU-hour utility ceiling in `src/moe_exp/routing_control/design.py:14-16`. Even at the X3 pilot rate of 838.61 tokens/s (`X3_COMPLETE_STAGE_PRICE_v3.json:28`), the decode alone is about 4.17 GPU-hours on two GPUs before prefill, loads, retries, and overhead. The protocol explicitly holds utility until relevant timing establishes fit (`PROTOCOL_v0.1.md:190-193`).

**Why it matters:** The objective includes accuracy and generated-token cost, but the current plan has no complete utility price and no demonstrated batched throughput that fits the declared stage.

**Shortest corrective experiment:** Measure a complete context-matched 16k native-vs-policy stage on qualified batched execution, price all loads/prefill/decode/retries, and either revise the budget prospectively or rescope the utility claim. Do not use the serial 8-token/s profile for the 16k stage.

---

### 12. [PROVEN BUG] Retry and failure costs are undercounted

**Evidence:** In `rate_eligible_immediate_semantics_v1.py`, failed rating attempts write an exception receipt without `prompt_tokens`, `generated_tokens`, or elapsed time (`:233-239`), while the summary totals only successful batch records (`:255-264`). The protocol requires all loads, retries, failures, and actual device-hour expenditure to be charged (`PROTOCOL_v0.1.md:195-203`). `QueryCostLedger` similarly rejects a duplicate `request_id` (`live_gate_v1.py:107-119`), so a retried side query cannot be recorded under the same identity.

**Why it matters:** Pricing and ITT cost accounting can be too optimistic, and the resource ceiling can be exceeded silently.

**Shortest corrective check:** Record actual prompt/generated tokens and elapsed time for every attempt, including failures, using attempt-specific IDs; include them in summaries and device-hour totals. Test a forced retry path.

---

### 13. [PROVEN BUG] Random-control distinctness guard is too weak

**Evidence:** `design.py:141-172` promises four random sets but accepts the result if `len({a.experts for a in sets}) >= 2` (`:170`). It does not require four distinct expert sets.

**Why it matters:** Mechanism validation could compare against duplicate random controls, weakening target-specificity claims.

**Shortest corrective check:** Require exactly `n_sets` distinct expert sets, or fail feasibility. Add a deterministic test with a seed/pool that would otherwise duplicate.

---

### 14. [PROVEN CONDITIONAL BUG] `paired_binary_bounds` does not implement the registered question-weighted estimand when seed counts vary

**Evidence:** `uncertainty.py:6-16` accepts arbitrary question-by-seed arrays and computes `difference = float(np.mean(left - right))` (`:16`), pooling all question-seed cells. The docstring/assumptions claim “equal question weights and seed means” (`:33`), but the function does not enforce equal seed counts per question. The protocol requires averaging seeds within question and then weighting questions equally (`PROTOCOL_v0.1.md:153-154`).

**Why it matters:** With missing seeds, unequal seed counts, or censored cells, the estimate and Hoeffding bound target a different estimand than registered.

**Shortest corrective check:** Enforce equal seed counts per question or compute per-question means first, then average questions/families. Add a test with unequal seed counts.

---

### 15. Additional concrete faults and integration gaps

These are lower-ranked individually but collectively matter for reproducibility:

- **[PROVEN INCOMPATIBILITY]** The v1 deactivation driver attaches ordered `routing_control` metadata to negative policies (`run_eligible_deactivation_serial_v1.py:175-177`), but `OrderedPulse.load` rejects any operator with `sign != 1` (`worker_adapter.py:45-46`). The v3 driver correctly removes ordered metadata and uses the base hook (`run_eligible_deactivation_serial_v3.py:54-55`, `198-199`). Treat v1 deactivation artifacts as stale.
- **[PROVEN VERSION MISMATCH]** `ordered_vllm.py:19` reports `ordered-worker-v1`, while the adapter it installs reports `ordered-worker-v2` (`worker_adapter.py:17`).
- **[PROVEN VERIFICATION GAP]** `manifests.py:22-44` checks that a supplied `code_digest` string equals the qualification’s `worker_code_digest`, but does not compute or verify the running code tree/file hashes. A manifest can bind a code version that is not the executing code.
- **[PROVEN INCOMPATIBILITY IF USED]** `receipts.py:22-30` requires 32- or 64-character lowercase-hex UIDs, while the eligible drivers generate UIDs such as `eligible-v2|<24 hex>` (`run_eligible_micro_serial_v2.py:147`) and `deactivate-v3|<24 hex>` (`run_eligible_deactivation_serial_v3.py:186`).
- **[PROVEN RECOVERY BUG]** `build_eligible_micro_blind_frame_v2.py:111-123` creates/writes `ARM_MAP.json` before writing `BLIND_FRAME.json` at `:193`. If a frame already exists but the map is missing, a new salt can be written and then the frame write fails, leaving an orphan map that blocks recovery.
- **[PROVEN ROBUSTNESS BUG]** `counterfactual.py:55-68` validates only `force_positive` and `force_negative`; an unknown operator string silently returns hits without enforcing either membership rule.
- **[PROVEN INPUT-VALIDATION BUG]** `design.py:112-116` skips only when `array.shape[2] < 8`; it accepts a truncated expert dimension rather than requiring the registered 256 experts.
- **[PROVEN GOVERNANCE GAP]** `live_gate_v1.py:53-57` accepts any caller-supplied `screen` callable and does not bind a sealed screen digest/version. The protocol requires a discovery-frozen screen before side queries.
- **[PROVEN EDGE-CASE BUG]** `rate_eligible_immediate_semantics_v1.py:66-68` rejects zero gradeable rows (`not 1 <= len(rows)`), even though the protocol requires all-assigned ITT handling for complete failure.
- **[PROVEN INTERFACE RISK]** `analysis.paired_itt` defaults to every non-reference arm (`analysis.py:154-157`); the micro-screen has a `native_duplicate` arm (`run_eligible_micro_serial_v2.py:81-89`). Without a supplied analysis driver passing exact `comparison_pairs`, the duplicate-native execution-noise arm can enter efficacy contrasts and multiplicity.
- **[PROVEN ARTIFACT GAPS]** The packet references but does not supply `run_boundary_micro_screen.py`, the micro/deactivation manifests, `FAMILY_FREEZE`, `JOINT_QWEN_NATIVE_EXACT_POOL_v1.json`, `TRANSITION_V22_FULL_PREFIX_START_FRAME.json`, the batched-neighbor manifest/gates/run directories, `RESOURCE_AUTHORIZATION_2026-10-02_v2.json`, or a utility-stage price. None should be assumed valid.

---

## Data leakage and blinding

**Strengths:**

- `PrefixInput.from_record` reads only `problem`, `emitted_token_ids`, and `emitted_text` (`prefix.py:140-147`).
- `LivePrefixGate.observe_record` passes only `problem` and `emitted_token_ids` to the detector (`live_gate_v1.py:63-66`).
- `test_live_gate_v1.py:32-42` explicitly checks that future fields cannot change the live decision.
- The blind frame excludes arm, policy, seed, family, expert route, future native text, and correctness fields (`build_eligible_micro_blind_frame_v2.py:1-4`, `156-170`), and the rating driver accepts only that frame (`rate_eligible_immediate_semantics_v1.py:1-5`, `64-81`).

**Risks:**

- The 13-family screen is drawn from the same discovery families used to select the expert IDs, so it is not independent confirmation. This is the main leakage/independence issue in the current design.
- The blind salt and `ARM_MAP.json` are stored in the output directory (`build_eligible_micro_blind_frame_v2.py:178-183`). Operational control must keep that map away from readers; the code does not enforce reader isolation.
- No correctness grader or its blinding is supplied, so answer-accuracy leakage cannot be audited.

---

## Statistical and claim-discipline strengths

The packet is unusually explicit about limits:

- It separates observational association from causal importance (`PROTOCOL_v0.1.md:8-11`, `90-98`).
- It uses ITT language and requires missing/failed cells to be retained (`:150-164`).
- It specifies simultaneous intervals and exact comparison pairs (`:155-164`).
- It forbids claims of retention without a noninferiority margin (`:166-169`).
- It labels the 13-family screen as not validation (`CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v2.json:22`).
- It labels the immediate 256-token rubric as not establishing the 1,024-token endpoint (`CAUSAL_ELIGIBLE_IMMEDIATE_SEMANTIC_RUBRIC_v1.md:31-35`).
- The action dictionary explicitly calls the approach action a proxy and not confirmatory (`CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json:351-352`).
- The batched audit admits observed non-rejection cannot establish engine equivalence (`CAUSAL_BATCHED_NEIGHBOR_QUAL_AUDIT_v2.json:480`).
- The legacy precision exercise reports substantial undercoverage of nominal intervals (`R3E_B4_PRECISION_INTERPRETATION_v0.1.md:7-17`), which is a useful warning against overclaiming calibrated uncertainty elsewhere.

These are real strengths, but they do not substitute for missing causal outcomes.

---

## Shortest credible route to a defensible result

### If the paper is limited to an exploratory immediate-semantic pilot

1. **Repair the rating and endpoint pipeline.**
   - Create the postgeneration-bound rating price required by the driver, or update the driver to the v3 schema.
   - Align the judge decode cap with the price (256 vs 1,024).
   - Add dense seven-class labeling only if the paper claims ordered trajectories; otherwise explicitly rescope to immediate binary target behavior.
   - Add correctness grading and actual token-cost receipts for every assigned request, including failures and retries.

2. **Repair triggering and endpoint logic.**
   - Use the v2.2 detector in the live gate, or fix v2.1 and re-freeze.
   - Fix the `\[...\]` regex.
   - Fix `dense_window` horizon and first-sentence coverage.
   - Enforce per-transition family representation in the 13-family screen.

3. **Qualify execution on the final code tree.**
   - Supply and audit the omitted serial qualification.
   - Keep serial/eager for the small causal screen unless a stricter batched qualification passes zero-condition same-text determinism and target-mask/top-k bounds.

4. **Run the 13-family serial screen as a pilot.**
   - Use the positive target, native duplicate, matched random, and both transitions.
   - Record actual dose, pulse rows, executed expert identities, route arrays, closure, and token costs.
   - Rate arm-blind with two Qwen3.8 readers.
   - Analyze with exact frozen comparison pairs, ITT, family-clustered simultaneous intervals, and no claim of retention.

5. **Report a null pilot honestly.**
   - A null result is informative if the intervals, actual dose, and measurement error are reported.
   - Without a pre-specified equivalence/noninferiority margin, a null is not evidence of absence.

### If the paper must answer the full objective

Add the following before any confirmatory claim:

- a separately frozen 1,024-token trajectory study on family-disjoint validation families, with dense contiguous seven-class sentence labels and the trigger sentence excluded;
- the ordered template with pulses at `[0,256)` and `[512,768)`, plus matched random and reversed-order controls;
- accuracy grading and actual generated/injected token accounting at the 16k utility cap;
- a measured complete utility-stage price that fits a declared budget, or a prospectively revised resource protocol;
- a qualification artifact proving the worker adapter’s batch isolation, pulse order, recompute behavior, and native routing on the exact execution profile.

Until those exist, the defensible conclusion is narrow: the packet defines a plausible protocol, but the causal ordered-trajectory and utility questions remain unanswered.
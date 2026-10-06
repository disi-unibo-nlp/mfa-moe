## Verdict

The new facts change several conclusions. The held 256-token positive screen is **not blocked** by the batched-neighbor defect, the preliminary rating-price mismatch, the detector bugs, or the `dense_window` bugs, because the held job is serial/eager, uses precomputed starts and fixed action slots, and does not run the rating or trajectory stages.

However, the serial stochasticity result means the screen must not be analyzed or described as an exact hidden-state pairing. It is an ITT policy-assignment pilot. The current source also leaves a separate control-assignment risk: arm order within each six-arm batch is fixed, not randomized or counterbalanced.

**Classification summary**

| Prior finding | Class | Source-level counterexample / why | Smallest corrective test |
|---|---|---|---|
| Serial/eager screen vs batched neighbor defect | **(iii) caveat/design choice** for current screen | `run_eligible_micro_serial_v2.py:22`, `:28-29`, `:187-190` forces `max_num_seqs=1`, `enforce_eager=True`; the batched audit is not exercised. | Verify the actual runtime status reports V1 runner, serial profile, and no batched neighbor dependencies. No change required for release. |
| 13 rows: 8 candidate / 5 approach | **(iii) current caveat; (ii) future guard** | Observed 8/5 resolves the current concern, but `run_eligible_micro_serial_v2.py:32-47` does not enforce per-transition minimums. A 13-candidate-family synthetic pool passes `len(chosen)==13`; `:113-119` passes for an empty transition. | Add a synthetic-pool test requiring a pre-specified minimum per transition or explicit feasibility failure. Do not block the frozen held manifest if its 8/5 allocation is sealed. |
| Four matched random sets | **(iii) current caveat; (ii) future guard** | The action dictionary contains four distinct sets. The generic guard remains weak: `design.py:170` accepts only two distinct sets for four requested. | Add a deterministic collision test requiring exactly four distinct sets. Current dictionary is not blocked. |
| Preliminary rating-price mismatch | **(ii) before rating/later claim** | `CAUSAL_ELIGIBLE_DEACTIVATION_BLIND_RATING_PRICE_v3.json:23` is explicitly pre-generation/non-executable. The packet’s rating driver still expects a frame-bound schema and checks fields at `rate_eligible_immediate_semantics_v1.py:64-93`; it also hardcodes `MAX_TOKENS=1024` (`:28`) while v3 prices 256 (`:11-12`). | CPU preflight: v3 price must be rejected; a frame-bound price with the driver’s exact schema and `max_decode_tokens = rows × readers × 1024` must pass. Bind the hash and verify `SamplingParams(max_tokens=1024)`. |
| Batched neighbor interference | **(ii) before later batched mechanism/utility claim** | The batched audit’s zero condition already shows same-text-prefix target-mask discordance 4/8 (`CAUSAL_BATCHED_NEIGHBOR_QUAL_AUDIT_v2.json:440-447`), while the gate ignores it (`audit_boundary_neighbor_qual_v2.py:121-130`). | Do not use batched execution for causal claims until zero-condition same-text target-mask/top-k equality and interference bounds pass, or keep those stages serial. |
| Omitted serial qualification audit | **(i) release-gate artifact check** | The price gate requires the serial audit (`CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v2.json:21`). The raw audit is not in the packet, though its summary is now supplied. | Hash-bind the sealed audit to the exact manifest/code and verify the eight-pair metrics and profile in the raw file. If it is not bound, do not release. |
| Detector v2.1/v2.2 and display-math handling | **(iii) for the precomputed-start 256 screen; (ii) before triggered/online trajectory claim** | The held driver does not import or run the live gate. The bug is real for later use: `live_gate_v1.py:16` imports v2.1, `transitions_v2.py:119-121` uses loose end-offset association, v2.2 fixes it at `transitions_v22.py:119-121`, and `transitions_v2.py:19-21` has the `\[...\]` closing-token bug. | Regression tests: a candidate ending before the next sentence must not attach to it; a `\[ x = 5 \]` sentence must be recognized. Only needed if the next test uses the live detector. |
| `dense_window` first-sentence/horizon behavior | **(ii) before later 1,024 trajectory claim** | `analysis.py:54` filters only `token_start < horizon`, so a sentence from 1,000 to 1,100 is included at horizon 1,024. `:47-49` checks only gaps between supplied rows, so rows starting at trigger+2 pass even if trigger+1 is missing. | Unit tests: missing `trigger+1` must raise; a sentence crossing the horizon must be excluded or handled by a frozen truncation rule. This does not affect the immediate 256-token rubric. |
| Seven-class ordered trajectory measurement | **(ii) before ordered-trajectory claim** | The rating driver returns only `target: bool` (`rate_eligible_immediate_semantics_v1.py:121-132`); `trajectory_completion` expects dense `behavioral_transition`/`substantive` rows (`analysis.py:12-34`). No dense seven-class labels for generated continuations are supplied. | Feed current rating output to `trajectory_completion` and show it is insufficient; add dense seven-class sentence labels for the later 1,024 test. |
| Approach expert uses a class proxy | **(iii) caveat for pilot interpretation; (ii) before semantic approach claim** | The action dictionary says it is an “Explore→Plan/Implement class proxy, not identified approach→semantic commitment” (`CAUSAL_DISCOVERY_ACTION_DICTIONARY_v1.json:351-352`). Passing native start checks validates starts, not the expert-selection proxy. | Compare proxy-selected starts with the frozen semantic approach rubric on the same discovery rows. If agreement is weak, scope the claim to the proxy transition. |
| Utility/accuracy cost | **(ii) before utility claim** | Utility worst case is 6,291,456 decode tokens (`PROTOCOL_v0.1.md:190`); at the serial stress rate of 8 token/s (`CAUSAL_ELIGIBLE_MICRO_SERIAL_PRICE_v2.json:13`) this is ~437 GPU-hours on two GPUs, far above the registered 4.75 GPU-hour utility ceiling (`design.py:14-16`). | Measure a complete batched 16k native-vs-policy stage, reprice all loads/prefill/decode/retries, and either revise the budget prospectively or rescope utility. |
| Retry/failure cost accounting | **(ii) before rating/cost claims** | `rate_eligible_immediate_semantics_v1.py:233-239` writes failure receipts without token/elapsed cost; `:255-264` sums only successful batches. | Forced judge-failure test must record and include actual failed-attempt tokens and elapsed time. |
| `paired_itt` default comparisons can include `native_duplicate` | **(i) pre-analysis freeze** | `analysis.py:154-157` defaults to every non-reference arm; the micro-screen arms include `native_duplicate` (`run_eligible_micro_serial_v2.py:81-89`). | Freeze exact `comparison_pairs` before unblinding and assert `native_duplicate` is a noise/replication control, not an efficacy contrast. |
| Fixed arm order inside the six-arm batch | **(i) conditional release check** | `run_eligible_micro_serial_v2.py:81-89` fixes arm order as native, native_duplicate, target0.5, target1, random0.5, random1; `:144-163` iterates that order; `:179-183` requires six-arm batches. Target and random arms therefore occupy different fixed positions. | Use the serial qualification/raw run to test whether native-duplicate or first-token route differences correlate with batch position. If order effects are present, randomize/counterbalance arm order before release; if absent, document fixed order as a caveat. |

---

## Does serial stochasticity invalidate randomized ITT?

**No, not by itself.** It invalidates exact hidden-state pairing and equivalence claims. It does not invalidate a randomized policy-assignment comparison that averages outcomes over runs.

The supplied analysis structure is consistent with that distinction:

- `run_eligible_micro_serial_v2.py:144-163` uses the same `common_seed` across arms, but this is common random numbers, not a guarantee of identical trajectories.
- `paired_itt` groups by question and arm, averages seeds within question, then questions equally, and resamples families (`analysis.py:151-188`). It does not require identical per-run native routes.
- The protocol’s ITT language requires averaging seeds within question and questions equally (`PROTOCOL_v0.1.md:153-154`), not exact token-level pairing.

The serial audit metrics should be read as follows:

- **1/8 text equality** means native duplicate runs are not exact replications. Treat `native_duplicate` as a second stochastic control, not as a deterministic duplicate.
- **0/8 unordered top-k equality at token zero** means there is no exact hidden-state/routing pairing even at the first token.
- **8/8 first-token target-mask equality** is a useful sanity check: target-expert membership at the first token was stable in these eight pairs. It does not establish trajectory determinism.
- **5/8 target-mask equality before text divergence** means in 3/8 pairs the target mask differed even on the same text prefix. Token-level “same-prefix” paired counterfactuals after divergence are therefore unsafe.

So the defensible interpretations are:

- **Valid:** randomized ITT of policy assignment vs native/random, averaged over seeds and families, with uncertainty that reflects between-run variation.
- **Invalid:** exact hidden-state equivalence, exact native-duplicate replication, token-level counterfactual pairing that assumes identical native routes, or claims that a routing difference alone explains the full trajectory difference.
- **Valid and still required:** within-run inactive-routing identity. `worker_adapter.py:83-97` compares actual vs native routing for inactive rows in the same forward pass; `audit_boundary_neighbor_qual_v2.py:82-85` checks zero inactive expert/weight mismatches. Between-run differences do not contradict that.
- **Valid and important:** realized dose telemetry. `ordered_action_dose`/`action_dose` records actual per-run active-row routing displacement (`worker_adapter.py:156-159`; `build_eligible_micro_blind_frame_v2.py:158-163`). Report it per run. Do not condition the primary ITT on dose; use dose as first-stage/secondary evidence.

The smallest statistical check is not more families. It is to quantify between-seed versus between-family variance from existing qualification/screen data and report a seed-level sensitivity interval alongside the prescribed family-clustered interval. If seed variance dominates, the pilot is noisy but still an ITT pilot.

---

## Minimum viable next 1,024-token test

The smallest defensible 1,024-token pilot reuses the frozen 13 rows and precomputed prefixes. It does **not** need the live detector.

1. **Inputs and execution**
   - Same 13 frozen rows and prefixes.
   - Serial/eager `max_num_seqs=1`.
   - Same two seeds, common-seed construction, presence restoration, and `routed_start`.
   - No online side queries and no live detector.

2. **Action**
   - Use the action/template selected by the 256-token pilot.
   - Do not run an ad hoc dose sweep after seeing outcomes.
   - If the pilot selects a two-action template, test native, frozen template, matched random, and reversed. For a one-action template, reversed is a replication per `PROTOCOL_v0.1.md:119-122`.

3. **Minimum arm set**
   - Native, native duplicate, frozen target, matched random.
   - For 13 rows × 2 seeds × 4 arms, that is **104 requests** and at most **106,496 decode tokens** at 1,024 tokens. If both biases and both random sets are retained, it is 156 requests and ~159,744 tokens—still small.
   - No extra families to chase significance. Freeze the pilot as a pilot.

4. **Measurement**
   - Dense contiguous seven-class sentence labels on each generated continuation, trigger sentence excluded.
   - Use the corrected `dense_window`: require `triggering_sentence + 1` coverage and require full-sentence containment within the 1,024-token horizon, or freeze an explicit truncation rule.
   - If dense seven-class labels are not produced, the test is only an extended immediate-target pilot, not an ordered-trajectory test.

5. **Analysis**
   - ITT with exact frozen comparison pairs.
   - Treat `native_duplicate` as a stochastic/replication control, not an efficacy comparator.
   - Report actual dose, executed expert identities, token cost, and family-clustered simultaneous intervals.
   - Add a seed-level variance/sensitivity check because serial execution is stochastic.
   - If null, report a null pilot. Do not enlarge the discovery pilot or run repeated 1,024 variants until significance. A powered confirmation, if needed, uses observed variance and the pre-registered disjoint mechanism families (`PROTOCOL_v0.1.md:205-207`).

---

## Changed conclusions from the prior audit

1. **Batched neighbor interference is no longer a blocker for the held 256 screen.** The job is serial/eager. It remains a blocker for later batched mechanism/utility claims.
2. **The rating-price mismatch is not a blocker for generation.** It is a real must-fix before the rating stage; the v3 price is explicitly non-executable and calls for a later frame-bound 1,024-token price.
3. **The random-control distinctness concern is resolved for the actual dictionary.** Four distinct sets exist. The generic code guard is still too weak for future regeneration.
4. **The per-transition omission concern is resolved for the actual 8/5 manifest.** The selection code still lacks a per-transition feasibility guard for future screens.
5. **Detector and `dense_window` faults do not block the held precomputed-start screen.** They must be fixed before triggered/online or 1,024-trajectory claims.
6. **Serial stochasticity does not invalidate ITT.** It rules out exact hidden-state pairing/equivalence and requires treating `native_duplicate` and seed variance honestly.
7. **Fixed arm order is a new conditional release concern.** The source does not randomize run order; if order effects exist, target-vs-random ITT may be confounded. Check position effects before release; otherwise counterbalance or document.
8. **The ordered seven-class trajectory claim still cannot be made from the 256-token screen.** That remains the main later-stage gap, together with utility pricing and the serial 16k throughput problem.
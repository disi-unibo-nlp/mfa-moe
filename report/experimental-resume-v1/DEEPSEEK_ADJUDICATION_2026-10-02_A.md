## Direct adjudication

1. **v3 rating price:** This is an **incomplete future gate**, not a defect in the submitted generation stage. The v3 file itself says pre-generation status and requires a post-generation, hash-bound price (`CAUSAL_ELIGIBLE_DEACTIVATION_BLIND_RATING_PRICE_v3.json:23`). No generation or rating was submitted. The held positive v2 generation does not need this price to run. The later newly constructed price must match the executing driver’s schema and 1,024-token reader cap.

2. **8 candidate / 5 approach:** For the current frozen manifest, this is **not a selection error**. Exact whole-row-list validation means the sealed 8/5 composition cannot silently change. The absence of an explicit per-transition minimum is a **future validator robustness risk** only if the selection routine is rerun on a changed pool.

3. **Serial nondeterminism:** The audit supports an **exploratory randomized/ITT-style contrast**, not exact hidden-state pairing. Exact native top-k determinism is not required for ITT. The minimal additional control is the already-present `native_duplicate` arm, analyzed as a stochastic reference, plus a pre-specified order-effect check. The 8/8 first-token target-mask equality is useful; the 0/8 top-k equality and 1/8 text equality mean native duplicate is a second stochastic draw, not a deterministic replication.

4. **Endpoint critique:** It applies to **both the paper objective and any over-interpretation of the pilot**, but not to the pilot’s narrow purpose. The 13-family, 256-token run is correctly labeled exploratory immediate-target discovery. It cannot establish ordered 1,024-token trajectory control or 16k utility. The registered validation/utility pools remain the confirmatory resource.

5. **Patched source:** The dense-window patches, display-math v2.3 detector, and exact-`n_sets` random-control guard do **not block the held fixed-prefix 256 screen**, because that screen uses precomputed prefixes and fixed action slots. Live gate v1 still importing v2.1 blocks only later online/triggered work.

6. **Fixed arm order:** Family/seed pairing plus a native duplicate is **not sufficient** to rule out systematic run-order drift. It estimates stochastic variance but not the full six-position order effect. The estimand is the average target-vs-native/random policy contrast under randomized or prospectively fixed arm assignment. The key assumption is order ignorability/stationarity. Because the job is held with zero GPU time, the cheapest fix is to re-seal with randomized or counterbalanced arm order before release.

---

## Corrected adjudication table

| Finding | Revised classification | Required before held 256 screen | Required before 1,024 mechanism / 16k utility |
|---|---|---|---|
| v3 rating price vs later rating price | **Incomplete future gate; not a submitted-stage defect** | None for generation. Record that rating is blocked until a new post-generation price exists. | Construct frame-bound price with driver/rubric/model hashes and the actual 1,024-token cap; validate with CPU preflight. |
| 8/5 pilot composition | **Current selection correct; future validator robustness risk** | None. Keep exact-list validation and sealed 8/5 manifest. | If selection is rerun, require a pre-specified per-transition minimum or explicit feasibility failure. |
| Serial nondeterminism | **Caveat for exact pairing; not an ITT defect** | Keep `native_duplicate`; freeze ITT interpretation: stochastic reference, not exact replication. | Add seed-level variance sensitivity. Do not add families merely to chase significance. |
| Batched neighbor qualification | **Irrelevant to held serial screen; later batched-stage blocker** | None; held screen uses `max_num_seqs=1` serial/eager (`run_eligible_micro_serial_v2.py:22`, `:28-29`, `:187-190`). | Use serial, or rerun batched qualification with zero-condition same-text top-k/target-mask isolation gates. |
| 13-family pilot vs objective | **Pilot scope is intentional; objective gap remains** | None if labeled as a 256-token exploratory immediate-target pilot. | Run a separate dense-label 1,024-token trajectory stage and a separately priced 16k utility stage. |
| `dense_window` full-containment/first-sentence fixes | **Later trajectory-stage fix; not used by held screen** | None for execution; ensure the held manifest still binds the executing code. | Bind the patched version; retain regression tests for missing trigger+1 and sentences crossing the horizon. |
| Detector v2.3 vs live gate v2.1 | **Later online-stage defect** | None; fixed-prefix screen does not call the live gate. | Wire v2.3 into `live_gate_v1` or explicitly freeze v2.1 with known limits; add adjacent-sentence and display-math tests. |
| Random-control distinctness | **Resolved for actual frozen dictionary; future guard now fixed** | None. Four distinct sets are present. | Keep exact-`n_sets` distinctness test in the bound code for any future control generation. |
| Fixed arm order within family/seed | **Conditional release blocker for a clean randomized/ITT claim** | Randomize/counterbalance arm order, or pre-specify and pass an order-effect check. Because zero GPU time has been used, re-sealing order is the cheapest option. | Carry counterbalanced order into 1,024; include order as a pre-specified stratifier/sensitivity. |
| Seven-class ordered trajectory measurement | **Not a pilot defect; required for objective claim** | None for the immediate 256 pilot. | Dense seven-class sentence labels, trigger excluded, corrected `dense_window`, token/routed-array alignment. |
| Utility/accuracy cost | **Later-stage gate** | None. | Measure complete batched 16k native-vs-policy throughput and reprice; do not use serial 8-token/s for utility. |
| Rating robustness: retry cost, map/frame binding, zero-gradeable ITT | **Later rating-stage fixes** | None. | Forced-failure test must charge failed attempts; frame/map write ordering must be crash-safe; zero gradeable rows must still produce an all-assigned ITT ledger. |
| Approach expert proposal | **Pilot caveat; later semantic-claim fix** | None; interpret as an Explore→Plan/Implement proxy. | Validate proxy-selected starts against the frozen semantic approach rubric, or narrow the claim. |
| Serial audit omitted from initial packet | **Release-evidence gate, not necessarily a scientific defect** | Hash-bind the sealed audit to the exact manifest/code and verify the raw eight-pair metrics. | Use it to justify serial execution or require fresh qualification for any new engine profile. |

---

## Most important remaining causal ambiguity

The most important ambiguity for the held 256 screen is **not** whether serial runs are deterministic. It is whether the fixed arm order confounds the target-vs-native/random contrast with serial drift.

Source-level counterexample:

- `run_eligible_micro_serial_v2.py:81-89` fixes arm order as native, native_duplicate, target0.5, target1, random0.5, random1.
- `:144-163` executes seeds then arms in that fixed order.
- `:179-183` requires a six-arm batch.

So target arms always occupy earlier fixed positions than random arms. The native duplicate tests one adjacent pair (`native` vs `native_duplicate`), not drift across all six positions. If the serial engine drifts with repeated runs, family/seed pairing removes prompt/seed differences but does not remove position effects. Native duplicate then measures noise, not the full order confound.

**Shortest resolving experiment:** re-seal the held 256 screen with randomized or counterbalanced arm order within each family/seed block, keeping the same 13 rows, same two seeds, same `native_duplicate`, and the same fixed-prefix actions. Analyze ITT with the exact frozen contrasts and a pre-specified order-effect check. This resolves the order confound without adding families or outcomes and uses the already-held, zero-GPU-time stage. If no order effect is detected, fixed order can be retained as a documented assumption; if it is detected, counterbalanced order is required for the 1,024 mechanism stage.

For the later paper objective, the minimum next trajectory experiment remains: same frozen starts, serial/eager execution, native + native_duplicate + frozen target + matched random, 1,024-token cap, dense seven-class labels, corrected `dense_window`, and ITT analysis. If the 256 pilot selects a two-action template, add the reversed-order arm; for one action, reversed is a replication.

---

## Unchanged conclusions

- These facts improve stage readiness and clarify intended gates. They do **not** prove causal steering.
- The 256-token pilot cannot establish ordered 1,024-token trajectory control or 16k utility.
- The approach expert set remains a proxy until validated against the semantic approach rubric.
- Utility remains infeasible under the serial 8-token/s path and has no complete 16k price in the supplied artifacts.
- Exact hidden-state pairing, exact native-duplicate replication, and token-level counterfactual equivalence claims remain unsupported by the serial audit.
- The live gate’s v2.1 import remains a real blocker for later online/triggered work, even though it does not block the held fixed-prefix screen.
# Routing study v2: 16k local-context execution

The implementation is additive: every new scientific continuation has a 16,384-token cap and a separate first-1,024-token mechanism endpoint. Historical 256/1,024-token results retain their original inference. No new target or independent outcome is qualified yet.

Observed on 2026-10-06T13:20:52.744359+00:00 as lmolfett on login05.leonardo.local. Existing recovery, operator and high-cap pricing jobs are preserved. Their current states are:

| Job | State | Exit |
|---|---|---|
| 59400762 | PENDING | 0:0 |
| 59400764 | PENDING | 0:0 |
| 59400766 | PENDING | 0:0 |
| 59405556 | PENDING | 0:0 |
| 59405561 | PENDING | 0:0 |
| 59405635 | PENDING | 0:0 |
| 59493889 | RUNNING | 0:0 |
| 59493892 | RUNNING | 0:0 |
| 59493902 | RUNNING | 0:0 |
| 59493908 | RUNNING | 0:0 |
| 59493911 | RUNNING | 0:0 |
| 59493914 | RUNNING | 0:0 |
| 59493917 | RUNNING | 0:0 |
| 59493921 | RUNNING | 0:0 |
| 59493925 | RUNNING | 0:0 |
| 59493930 | RUNNING | 0:0 |
| 59493932 | RUNNING | 0:0 |
| 59493935 | RUNNING | 0:0 |
| 59493937 | RUNNING | 0:0 |
| 59493939 | RUNNING | 0:0 |
| 59493943 | RUNNING | 0:0 |
| 59493945 | RUNNING | 0:0 |
| 59493947 | RUNNING | 0:0 |
| 59493949 | RUNNING | 0:0 |
| 59493952 | RUNNING | 0:0 |
| 59493954 | RUNNING | 0:0 |
| 59493956 | RUNNING | 0:0 |
| 59493958 | RUNNING | 0:0 |
| 59493962 | RUNNING | 0:0 |
| 59493965 | RUNNING | 0:0 |
| 59493967 | RUNNING | 0:0 |
| 59493969 | RUNNING | 0:0 |
| 59493972 | RUNNING | 0:0 |
| 59493974 | RUNNING | 0:0 |
| 59493979 | RUNNING | 0:0 |
| 59493981 | RUNNING | 0:0 |
| 59493984 | RUNNING | 0:0 |
| 59493986 | RUNNING | 0:0 |
| 59493992 | RUNNING | 0:0 |
| 59494010 | RUNNING | 0:0 |
| 59494061 | RUNNING | 0:0 |
| 59494066 | RUNNING | 0:0 |
| 59494071 | PENDING | 0:0 |
| 59495690 | PENDING | 0:0 |

All 187,524 dense identities remain bound to the historical frame. Complete identity reconciliation is pending the already queued recovery/closeout. `dense-ledger` creates a new identity ledger from committed receipts and explicit unresolved assignments; it never resubmits work. Pending calls are not outcomes.

The v2 code implements fold-local eight-component PCA and K=1/2/4 clustering, nested family prediction, ten-family association support and the three-outer-fold gain rule. Multiple contexts fall back to K=1 when support fails. Native ID composition and saved execution weights remain separate signals. Actual local-context fits and 64/256-token anticipation remain unavailable until their prediction frame qualifies.

The first qualified opportunity is final for each transition: context declines and random skips cannot retry later. Skip probabilities are frozen from discovery strata of source state, prefix status and position. Delayed early acts, closure, ambiguous decisions and overlapping composition pulses retain no-fire receipts. Controller generation is charged to policy cost. Snapshots bind request, prefix, controller ledger, capture and exact sampler/engine state.

New manifests bind recognizer, context, sampler/resume, all-40-layer ID/selected-weight capture, 16k cap, early endpoint and immutable code. Target/operator/timing/context effects and achieved dose remain unobserved. Exploratory advancement requires verified matched dose, positive semantic point effects versus both controls, three positive family folds and reported simultaneous intervals; intervals may remain inconclusive.

The existing 32/96 family reservation, two seeds and source revision are unchanged. Native validation runs first. State-only and context-aware counterparts use the same actions and settings. The incumbent must qualify before validation; its forced L28/E9+E189 and +1 L24/E159 actions remain fixed and unproven. Final enrollment is at most 768 assignments. Accuracy–cost frontiers remain unavailable.

Measured stage prices require generation, prefill, model loads, controller, replay, capture/storage, annotation, grading, runtime and bounded recovery. Short or mismatched pilots cannot price 16k cells. No new jobs are submitted. Stage resource authorization follows a complete measured price.

Native-finalization qualification failure, failed counterfactual-NLL qualification and deferred X3 remain separate limitations; qualified free generation can proceed independently. CPU tests validate contracts and information boundaries; GPU fidelity, capture, RNG continuity and semantic recognition still require observed passing receipts.

The following historical report is preserved evidence from its original observation time. Its progress counts are historical snapshots; the job table above is the current scheduler observation.

# Routing study: evidence and remaining execution

Historical A/B/C/FRESH estimates and their saved 50,000-draw intervals were reproduced exactly. Each population, cap, endpoint and inference remains separate. Capped mechanism results do not establish final-answer utility.

| Study | Cap | Assigned | Valid semantic pairs | Interpretation |
|---|---:|---:|---:|---|
| A | 1024 | 128 | 110 | Discovery mechanism evidence |
| B | 1024 | 156 | 141 | Discovery mechanism evidence |
| C | 256 | 208 | 200 | Discovery mechanism evidence |
| FRESH | 1024 | 3480 | 2920 | Discovery mechanism evidence |

Dense recovery inventory: 11,200 committed of 187,524; 176,324 missing; 8 old calls without committed outputs retained explicitly.

A does not establish an advantage of paired verification experts over either singleton. B does not establish additional benefit of the second pulse. C shows an exploratory reweighting effect in its short horizon; the degenerate bias estimate does not establish equivalence. These are substantive reader outcomes, distinct from the seven-class sentence descriptions.

| Study/scope | Contrast | Substantive success change (pp) | Frozen simultaneous 95% interval (pp) | Families |
|---|---|---:|---|---:|
| A/all | pair − native | 6.25 | [-25.00, 50.00] | 8 |
| A/all | expert189 − native | 0.00 | [-25.00, 25.00] | 8 |
| A/all | expert9 − native | 6.25 | [-25.00, 37.50] | 8 |
| A/all | pair − expert189 | 6.25 | [-25.00, 50.00] | 8 |
| A/all | pair − expert9 | 0.00 | [-25.00, 25.00] | 8 |
| B/all | once − native | -7.69 | [-30.77, 15.38] | 13 |
| B/all | repeat − native | -11.54 | [-42.31, 0.00] | 13 |
| B/all | repeat − once | -3.85 | [-26.92, 15.38] | 13 |
| B/candidate_to_verify | once − native | -12.50 | [-43.75, 25.00] | 8 |
| B/candidate_to_verify | repeat − native | -18.75 | [-62.50, 0.00] | 8 |
| B/candidate_to_verify | repeat − once | -6.25 | [-37.50, 25.00] | 8 |
| B/approach_to_commit | once − native | 0.00 | [0.00, 0.00] | 5 |
| B/approach_to_commit | repeat − native | 0.00 | [0.00, 0.00] | 5 |
| B/approach_to_commit | repeat − once | 0.00 | [0.00, 0.00] | 5 |
| C/all | bias − native | 0.00 | [0.00, 0.00] | 13 |
| C/all | force − native | 0.00 | [-23.08, 23.08] | 13 |
| C/all | reweight − native | 3.85 | [0.00, 19.23] | 13 |
| C/all | force − bias | 0.00 | [-23.08, 23.08] | 13 |
| C/all | reweight − bias | 3.85 | [0.00, 19.23] | 13 |
| C/candidate_to_verify | bias − native | 0.00 | [0.00, 0.00] | 8 |
| C/candidate_to_verify | force − native | 6.25 | [-25.00, 37.50] | 8 |
| C/candidate_to_verify | reweight − native | 6.25 | [0.00, 31.25] | 8 |
| C/candidate_to_verify | force − bias | 6.25 | [-25.00, 37.50] | 8 |
| C/candidate_to_verify | reweight − bias | 6.25 | [0.00, 31.25] | 8 |
| C/approach_to_commit | bias − native | 0.00 | [0.00, 0.00] | 5 |
| C/approach_to_commit | force − native | -10.00 | [-40.00, 0.00] | 5 |
| C/approach_to_commit | reweight − native | 0.00 | [0.00, 0.00] | 5 |
| C/approach_to_commit | force − bias | -10.00 | [-40.00, 0.00] | 5 |
| C/approach_to_commit | reweight − bias | 0.00 | [0.00, 0.00] | 5 |

The complete saved contrasts, token effects, and missing-reader identification bounds are retained in HISTORICAL_INTEGRATION.json. Small-family and degenerate bootstrap limitations remain applicable.

![Historical pulse trajectories](../routing-study-v1/historical-pulse-trajectories.png)

The 0–256 and 512–768 windows are the first and repeated pulse positions. Post-pulse descriptions use only exact saved expert IDs and sentence token ownership. Unknown labels break adjacency; closure competes with state exits; caps censor unobserved exits. The complete window table separately reports routing selection, class occupancy, unknown labels, observed reasoning duration and closure. Window comparisons are descriptive and carry no new significance claims.

Bias and forced inclusion are compared using the frozen inclusion-per-intended-opportunity endpoint. Reweighting also requires executed selected-weight displacement: sparse L1/2 and target mass from saved execution-local telemetry. The dose table retains TP ranks separately and excludes padded inactive layers. Its weights are normalized executed top-k weights, not full router probabilities. Exposure, displacement and semantic effects are not treated as equivalent.

![Global and conditional expert profiles](../routing-study-v1/global-conditional-heatmap.png)

44,442 expert-condition rows meet the ten-family/three-fold direction rule for class-transition proxies. No new singleton is selected: candidate completeness and checked status still need a prefix-only semantic recognition audit. Global versus conditional signs, ranks and cosine comparisons are recorded. The prospective predictive interface fits feature selection and regularization inside family folds, but held-out predictive gain is unavailable without a qualified prefix baseline and trailing feature frame. Full router confidence and native selection-boundary gaps are unavailable.

Unchecked-candidate verification and approach commitment remain hypotheses requiring qualified prefix opportunities. Already-checked candidate closure and failed-check revision have insufficient qualified support. No causal specialist or original-prompt accuracy claim is made.

The separate native-finalization prompt-control diagnostic has zero of 32 outcomes. Native repeat variability and execution-local capture were audited; the v4 exact native-token repeat gate failed. It remains an engineering qualification failure. Counterfactual-NLL qualification remains failed; X3 remains deferred. Qualified free-generation analysis can proceed independently.

The full 5,000-question MATH test source is pinned at revision 21a5633873b6a120296cce3e2df9d5550074f4a3. All 500 prior MATH-500 texts matched. The duplicate graph preserves prior family connections and leaves 4,479 eligible new families. INDEPENDENT_FAMILY_MANIFEST.json now freezes 32 validation and 96 final families with disjoint identities and subject/level stratification. The approved CPU work completed in jobs 59493569 and 59494618: the initial job wrote a valid sealed graph but failed a tuple/list equality check; the bounded completion reused that graph. Total elapsed allocation was 178 seconds on two CPUs, within the approved two-core-hour limit. No new-family outcomes have been generated or used for selection.

The independent analysis implements seed averaging within family, equal family weights, paired simultaneous 95% accuracy–cost intervals, missing-grade sensitivity, and the stated deterministic validation choice. No acceptable accuracy-loss margin is imposed. The frontier is unavailable until qualified policies and independent outcomes exist; a nonsignificant accuracy difference will not be reported as accuracy retention.

| Resource component | Observed / authorized status |
|---|---|
| Existing dense labels | 36 packed jobs submitted and verified; maximum 1707.24 GPU-hours including bounded recovery; dependent analysis job 59494071 |
| Packaging qualification | Original 59490750 failed at the IPC path limit; corrected 59491564 completed with 64/64 valid stopped labels and exact label agreement across two original shards in one load |
| Existing utility pilots | Jobs 59405556, 59405561, 59405635 retained and reconciled |
| Existing operator pilots | Jobs 59400762, 59400764, 59400766 retained and reconciled |
| Offline integration | Two CPUs, at most one hour; historical authorized analysis |
| Independent family freeze | Approved and completed within two CPU-core-hours, zero GPUs; 32/96 family freeze verified |
| New target / operator / timing / evaluation | Not submitted; complete measured generation, prefill, controller, labeling, grading, load and bounded-recovery prices required |

Policy cost includes controller-generated tokens. Processed input, controller overhead, runtime, reasoning tokens, and offline labeling/grading expenditure will be itemized separately. Accuracy damage and net correct-to-incorrect flips per exposure remain unavailable without paired final-answer grades; near-zero-dose ratios will stay descriptive.

Current machine-readable inputs: STUDY_INVENTORY_v3.json, CANDIDATE_TABLE.json, INTERVENTION_MANIFEST_v2.json. Earlier immutable versions preserve their original observations. Historical source artifacts remain unchanged. This study is incomplete while label recovery, queued pilots, prefix qualification and independent evaluation are outstanding.

![Transition-aligned native routing](../routing-study-v1/native-transition-aligned.png)

In this plot, source sentence 0 precedes destination sentence +1. Every included offset has a contiguous valid label chain; events and seeds are averaged within family before families receive equal weight. This separates native pre-transition profiles from routing in the destination. These sentence-scale class proxies do not qualify a 64-token timing trigger or establish a substantive verification. Token-scale anticipation remains unavailable until raw prediction-row alignment and a prefix-only baseline are qualified.

![Native state-transition diagram](../routing-study-v1/native-state-transitions.png)

The diagram shows up to two most frequent observed exits per state. Edge labels are destination probabilities conditional on an observed state exit. Self transitions, all exit counts, dwell, re-entry and loops remain in NATIVE_BOUNDARY_PROFILES.json. Missing labels break adjacency. Selected-window ends censor termination, so no closure probability is inferred from them.

The corrected packaging changed only private runtime path length and model-load sharing across existing shards. Original prompts, sampler, assignment identities and valid saved labels are preserved. All 36 GPU receipts and the dependent analysis receipt are in submissions-v5/RECOVERY_CHAIN.json; pending jobs are not completed label outcomes.

The new intervention_design_v1.py and policy_stages_v1.py contracts freeze at most two singleton targets, signed screens, operator comparisons, calibrated controls, delayed assignments and supported composition. The online controller derives text from already emitted IDs, records controller expenditure, and retains no-fires after substantive acts, closure or caps. These interfaces still need qualified recognizer and GPU capture/executor bindings before new stages can run. Worker-specific price components and dose-normalized answer-flip diagnostics are implemented separately in policy_stages_v1.py and resource_analysis_v1.py.

Verification: 54 relevant tests passed. Saved historical effects and 50,000-draw intervals reproduced exactly; frozen recovery source hashes remain unchanged; every packed submission has an exact verified Slurm job ID; independent source and family seals were read back. The independent accuracy–cost frontier remains unobserved.

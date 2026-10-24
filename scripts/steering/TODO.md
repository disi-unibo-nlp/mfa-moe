# steering: to do

## whole-reasoning-sweep: Execute the whole-reasoning efficiency experiment
- status: active
- asked: "Alignment v2: stop the 512-token rescue and run the whole-reasoning (SteerMoE-style) efficiency experiment"; amended by "Alignment v3: fix four launch blockers, then run the whole experiment tonight". (2026-10-20)
- why: Measure strict accuracy and reasoning length against matched random and native, retaining capped and missing outcomes as bounds.
- done when: 129 frozen families; up to 3096 GPT and 2838 Qwen rows, with non-nominated cells excluded; source-bound fresh execution; allocated full CPU suite/prepare and passing 48-row per-model canaries; six shared simultaneous contrasts and mandatory companions/manipulation/dose reporting. Continuous 64-slot admission, 2 GPT/7 Qwen unheld shards, relative fluency gates and a monitor whose deadline/errors never stop production. New GPU-hours at most 72, retained r3 0.346111 separately; CPU-only core-hours at most 16; science cutoff 2026-10-23 05:30 Europe/Rome.
- decisions: whole-reasoning-design
- code: src/steering, tests/steering, scripts/steering
- [x] Freeze r3 terminal accounting and supersede its unfinished 512-token plan; preserve all historical receipts; evidence: native sacct readback 2026-10-20, unchanged r3 TERMINAL.json, and report/whole-reasoning-20261020/README.md. Queue empty; r3 GPU 0.346111 hours and prior CPU 0.903333 core-hours remain recorded.
- [x] Implement D1 control-feasible global sign-stable RD nominations with exact layer/quintile/role controls, matching all70 saved GPT DP cells and preserving Qwen/GPT suppress exact ranks; rebuild both doses and record retained RD per cell; evidence: source95d4cb0, CPU61577861 COMPLETED0:0, report/whole-reasoning-20261020-r3/NOMINATION_AUDIT.json and NOMINATION_CHANGES.csv. All210 cells qualified; 3096GPT/2838Qwen rows executable at both doses.
- [x] Implement dual operator, absolute prompt/reasoning scope, closure/recompute behavior and aggregate dose/full audit captures; focused tests including real short Unix socket; evidence: report/whole-reasoning-20261020/LOCAL_VALIDATION.json and FINAL_FOCUSED_TESTS.log. The >2048-token chunk, re-prefill, prompt Harmony suffix, native answer and generated-position-zero contracts pass; immutable-copy allocated verification remains whole-reasoning-sweep (step 6).
- [x] Correct bounded metric inference, missing benchmarks, signed matrices, occupancy companions and main log-total link; threaded scoring and replay-based reasoning counts; evidence: 17 effects regressions, 11 final focused checks and three replay/closeout checks passed, with six bounded primary exports even for entirely missing production. See LOCAL_VALIDATION.json, REPLAY_CLOSEOUT_TESTS.log and tests/study/test_effects.py. Fresh real analyses remain whole-reasoning-sweep (step 8).
- [x] Record section4 endpoints, v3 decisions, within-family repeat-noise SD/MDE using driver accuracy (SUSPECTED), AGENTS/whole-reasoning-sweep/whole-reasoning-design and CPU/GPU reservations; commit and materialize immutable whole r3 release; evidence: source95d4cb0, RELEASE.json, RESOURCE_PLAN.json and allocated SD_MDE.json. Main manifests/outcomes contain zero native_repeat pairs; no paired SD is estimable and no between-family substitute is used. Planning does not gate science.
- [ ] Run one allocated full study suite plus prepare with WORK staging, then the same eight-family/48-row full-allowance canaries; synthetic relative gates and declared fluency-only half-dose fallback must pass.
- [ ] Submit 2 GPT165-minute/7 Qwen240-minute two-GPU shards, all unheld afterok at 64 in flight without family barriers; native/efficiency tier precedes other promote rows; enforce relative early gates, quarter projection, storage and cutoff.
- [ ] Queue eight-CPU/90-minute extraction with each model's shards and eight-CPU/120-minute analysis afterany both extractions; separate worker tokenizers, visible extraction failures, not_in_design cells, point-observed closed reasoning and unit-level coverage failures; fresh corrected main/strong analyses and link qualification disclosure.
- [ ] Supervise to terminal accounting and report actual nominations, completeness, six primaries, companions, dose, exploratory link and occupancy with per-model verdicts.
- [ ] Record real component walkthrough and final checks without manuscript edits or remote pushes.
- [x] Test and deploy the one-CPU/2-GB/16-hour monitor: deadline and exceptions never publish STOP or cancel production, transient accounting/quota errors retry after60s, frozen projections use actual planned maxima; queue dependent work independently and guard actual CPU use; evidence:184 allocated tests, source95d4cb0 and accepted/running monitor61578500. CPU-only workers enforce cumulative15.5core-hours with0.5 shutdown margin; actual final accounting remains whole-reasoning-sweep (step 9).
- [ ] Correct the measured all-layer callback bottleneck with batched row-exact operator/counter/audit work; preserve the original r3 qualification/failures and charge0.842222GPU-hours; validate the new source on CPUs and allocated canaries, then admit the same production population under remaining72/16limits. Original165/240-minute shard limits are maxima; bind any budget-required reduction before submission. Preserve purged-dependency semantics with native proofs.
- [ ] Diagnose the actual GPT rank-capture disagreement with lossless per-rank records and first-error-preserving cleanup; allocate at most2GPUs/15minutes after a fresh full CPU qualification. This engineering command cannot qualify or launch production. Keep Qwen's existing owner graph and reserve every cross-release actual/maximal charge before any further GPT allocation; evidence: five focused lifecycle/admission/resource checks pass, allocated diagnostic pending.

whole-reasoning-sweep (step 6) allocated validation is complete: CPU61577861 COMPLETED0:0 after288s,
184tests passed in67.626s after both-dose preparation. Canary outcomes remain
pending. The allocated audit confirms70 exact DP matches; full-dose GPT
promote replaced185 deactivation nominations across35cells, retaining
94.25754–99.73353% RD; half-dose replaced3 across35cells, retaining
99.91228–100%. Other targets and all activation selections are unchanged.
Evidence: report/whole-reasoning-20261020-r3/CPU_VALIDATION.json,
NOMINATION_AUDIT.json and NOMINATION_CHANGES.csv.
The batched source3a848b4 also passed all190allocated tests in CPU61580365
(149s,0.165556core-hours); no GPU was released from it. Its final engineering
correction uses deterministic contiguous segment sums, because floating
atomic groups could disagree between ranks. Seven focused graph/operator/
counter/audit/recompute tests pass. Continue the deterministic subdirectory;
its1.637778CPU/0.842222GPU-hour prior retains every allocation and local charge.
Final source16cf599 then passed all190allocated tests in CPU61581053; its
73.767-second test stage follows fresh main/half preparation for both models.
All70 GPT promote DP matches and all210qualified cells remain unchanged.
Evidence: report/whole-reasoning-20261020-r3-batched/deterministic/CPU_VALIDATION.json.
Final canaries GPT61581237/Qwen61581238 and monitor61581239 started23:42:29
Rome. Native resource readbacks are preserved in that root's JOBS.json and
OBSERVED_LAUNCH.json. Passing CPU tests do not establish a passing canary.
whole-nomination-trace now traces the final source and allocated nomination evidence.
GPT61581237 then failed1:0 after545s on strict tensor-parallel capture
agreement. Three canary units completed; none is production. GPT extraction
61581597 completed0:0 in227s with zero production units/failures. Qwen's
canary61581238 continues; the software gate prevents GPT production and the
fluency-only half-dose fallback. whole-canary-trace traces one actual completed GPT canary
unit and explicitly preserves the later allocation failure.
The separate diagnostic source6d4341f passed all194allocated tests in
CPU61583800 (170s,0.188889core-hours); both-dose preparation preserves all70
DP matches. GPT61584000 is accepted for2GPUs/15minutes and cannot launch
production. Its admission reserves63.645278GPU-hours with Qwen's optional
half-dose and full production maxima, without admitting further GPT science.

### Results

| Measure | Data | Before | After | Evidence |
|---|---|---|---|---|
| R3 engineering | Superseded 512-token rescue | 61554207/61554215 | Both FAILED 1:0; 312/311 seconds; 0.346111 GPU-hours | Native sacct readback 2026-10-20; original r3 TERMINAL.json |
| Focused whole contracts | Synthetic nomination/operator/prompt and all-missing scientific closeout fixtures | New design lacked contracts | 22 tests passed in 41.281 seconds; additional horizon/gate/signed checks and three replay/closeout checks passed | report/whole-reasoning-20261020/LOCAL_VALIDATION.json; repeated test counts are not added |
| Actual staging and entitlement | login05.leonardo.local, lmolfett | Required live authorization evidence | Native iscrc_miosr/normal association, empty queue, project quota20149688; no new jobs at snapshot | report/whole-reasoning-20261020/PREFLIGHT.json and HISTORICAL_ACCOUNTING.json |
| First whole allocated validation | Immutable abbcffd release; 165 study tests | Required before discovery and canaries | FAILED 61565142: one incomplete-sentence fixture assertion and one archived data-path fixture error. Canary jobs cancelled with zero allocation; all seven jobs terminal, 0 GPU-hours / 0.407778 CPU core-hours retained | report/whole-reasoning-20261020/TERMINAL.json and logs/whole-validate-61565142.err |
| Focused correction | Same frozen whole experiment | Failed fixtures and supervisor path | Ten tests passed, including cumulative usage/projection and immediate stop on validation failure; pooled native planning retains paired family covariance | report/whole-reasoning-20261020-r2/LOCAL_VALIDATION.json and PRIOR_USAGE.json |
| Actual whole control feasibility | Frozen discovery nominations; no whole outcomes | Requested within-layer quintile matching with cell-local exclusions | GPT main promote 0/35 feasible, suppress 20/20; three frequency bins would make 35/35 promote cells feasible, requiring a user amendment. Qwen all 50 cells exact-count feasible at both doses; 2838 unique executable rows per manifest | report/whole-reasoning-20261020-r2/FEASIBILITY.md and CONTROL_FEASIBILITY.json; Qwen CPU61566702 COMPLETED 0:0 |
| Completed-output extraction correction | GPT Harmony replay fixture, native usage reasoning count zero | Helper import was outside worker scope | Seven replay reasoning tokens and strict-correct answer extracted; nine extraction/closeout/gate tests passed in 41.043 seconds | tests/study/test_whole.py; report/whole-reasoning-20261020-r2/extraction-fix.time |
| Current whole accounting | All accepted jobs in both whole attempts plus Qwen feasibility | No failures reset | New GPU 0; allocated CPU 0.541667 core-hours plus conservative local charge 0.1. No canary or production outcome; matching amendment and concurrent-session clarification pending | report/whole-reasoning-20261020-r2/CURRENT_STATUS.json; both TERMINAL.json records and qwen-feasibility/TERMINAL.json |
| Inactive control-feasible nomination proposal | Original GPT discovery candidates; fixed positive activations and opposite-polarity exclusions | Exact-ranked main promote cannot admit quintile controls | All 35 main and all 35 half promote cells admit exact layer/quintile/role controls at unchanged doses and layer caps. Main retains 94.258–99.734% of original deactivation RD; half retains 99.912–100%. Exact dynamic-program optima independently match all main integer-program optima. No nomination rule changed or approved | report/whole-reasoning-20261020-r2/CONTROL_FEASIBLE_RD_REVIEW.md and CONTROL_FEASIBLE_RD_BOTH_DOSES.json |
| Canary historical feasibility review | Eight outcome-independent family hashes per model; original saved native traces only | Fixed canary allowance 4096, at most two capped steered units | Original native completion length exceeds 4096 in 2/8 GPT and 7/8 Qwen families. Disclosed risk only; selection and gates unchanged, no new canary outcome | report/whole-reasoning-20261020-r2/CANARY_NATIVE_LENGTH_FEASIBILITY.json; final twelve-job readback and 0.641667/16 CPU charge in CONTROL_REVIEW_ACCOUNTING.json |
| V3 allocated validation | Rebuilt GPT/Qwen main and half-dose manifests; full study suite | No allocated qualification of the corrected nominations | CPU61577861 COMPLETED0:0, 288s/0.32CPU core-hours; 184tests pass; 210cells qualified; all70 GPT promote DP targets/controls match | report/whole-reasoning-20261020-r3/CPU_VALIDATION.json; NOMINATION_AUDIT.json; native sacct |
| V3 nomination changes | 35GPT promote cells per dose; same frozen discovery data | Exact-ranked deactivation RD | Main185 pair replacements total, retained RD94.25754–99.73353%; half3 replacements, retained RD99.91228–100%; every Qwen/GPT suppress target and every activation unchanged | report/whole-reasoning-20261020-r3/NOMINATION_CHANGES.csv; whole-nomination-trace |
| Paired planning evidence | Main driver manifests and outcome tables, GPT/Qwen | Requested native_repeat−native noise | Zero historical native_repeat assignments/pairs; paired SD/MDE unavailable, marked SUSPECTED; no fallback SD or science gate | report/whole-reasoning-20261020-r3/SD_MDE.json; historical driver ASSIGNMENTS.json arm counts |
| Whole-hook engineering throughput |48GPT canary rows; checkpoints22:49:55/22:51:56 | Main-study per-sequence speed assumed transportable to all-layer instrumentation |3216tokens/121s =26.5785aggregate tok/s,0.5537per sequence; zero completed units. GPT/Qwen/monitor canceled at758s;0.842222GPU-h and0.210556monitor core-h retained; no production | report/whole-reasoning-20261020-r3/PERFORMANCE_HALT.json and TERMINAL.json |

## outcome-steering-r1: Outcome-selected gentle steering experiment
- status: active
- asked: "define the implementation plan ... human readable code/repo"; approved plan 2026-10-22 (outcome-selected gentle steering on 3 models, length-limit baseline, three nudge strengths, scripts/ cleanup after launch). (2026-10-22)
- why: Test whether a small set of experts chosen by outcome (used more in wrong than right runs of the same question) and nudged down gently at every reasoning token shortens reasoning without losing accuracy, against matched random experts, unedited routing and a plain stop-thinking-earlier baseline.
- done when: For gpt, qwen330b and nemotron at delta 0.4/0.8/1.6: bounded target-random, target-native, random-native and native_repeat-native contrasts, the budget_50/70/85 curve and the target-versus-budget comparison in a sealed EFFECTS_OUTCOME.json; manipulation check shows the nudged experts used less with zero membership failures; all GPU work before 2026-10-23 23:00 Europe/Rome within an 82 GPU-hour ceiling.
- decisions: gentle-routing-nudge
- code: src/steering/outcome, src/steering/operators.py, scripts/steering, tests/steering/outcome, src/steering
- [x] Outcome selection: per-attempt expert usage on campaign-v3 cohort B, within-question wrong-minus-right score, leave-one-question-out stability, shared-core and execution exclusions, mass truncation, holdout sets and matched random controls; tests in tests/outcome_steering/test_select.py; evidence: usage job 61764186 (gpt 276x24x32, qwen330b 219x48x128, nemotron 96x23x128); preview sets gpt 20 pairs 1.21% mass, nemotron 40 pairs 1.07%, qwen330b 40 pairs 0.39% (cap), no unstable fill.
- [x] Shift operator in final_routing/operator.py and stream dispatch with shift counters; existing dual path unchanged; tests in tests/study/test_stream_batch.py; evidence: 9 tests in tests/study/test_stream_batch.py pass (5 new NudgeTests).
- [x] Budget-prefix rows (cut native reasoning, forced closing suffix, answer-only allowance) in runtime/driver/extraction; tests in tests/study/test_whole.py; evidence: tests/outcome_steering/test_manifest.py budget-prefix tests pass.
- [x] Manifest builder and release (OUTCOME.json sweep release reusing base natives); tests in tests/outcome_steering/test_manifest.py; evidence: local dry run built 1161/1152/1152 rows (gpt 12 budget rows source_unavailable); allocated job 61796625 ran 99 whole tests, one message assertion updated.
- [x] Analysis: bounded contrasts, budget-curve comparison, manipulation check; tests in tests/outcome_steering/test_analysis.py; evidence: tests/outcome_steering/test_analysis.py passes.
- [x] READMEs for src/moe_exp/outcome_steering and scripts/outcome_steering; launch, supervise to terminal accounting, report per-model results; evidence: release r1 shards all COMPLETED, report/outcome-steering-20261022-r1/analysis/EFFECTS_OUTCOME.json and DECISION.json (extend: false).

### Results

| Measure | Data | Before | After | Evidence |
|---|---|---|---|---|
| Nudge vs matched random, accuracy (pp) | 129/128/128 families | random experts | no interval excludes 0 (gpt -6.9..+2.8, nemotron -2.5..+2.5, qwen330b -2.5..+2.5) | EFFECTS_OUTCOME.json target_random |
| Nudge vs matched random, tokens | same | random experts | no interval excludes 0; largest nemotron delta 1.6 -12.8% [+5.9, -28.1] | same |
| Nudge vs unedited, accuracy (pp) | same | native | -5.0..+1.7, all intervals include 0 | target_native |
| Stop-thinking-early baseline, accuracy (pp) | same | native | -6 to -47 (qwen330b answers often hit the 2048-token answer cap) | budget_native |
| Nudge vs length-limit curve at matched length (pp) | same | budget curve | +7.4 [+0.5, +14.3] nemotron delta 0.8; others include 0 | target_budget |
| Manipulation: nudged experts' top-k slots kept | same | native | 0.24-0.34 at 0.4, 0.07-0.18 at 0.8, 0.00-0.05 at 1.6 | models.manipulation |

## whole-states-heatmap: Complete the whole-reasoning per-state heatmap with the missing suppress cells
- status: active
- asked: "SBatche all the remaining runs to compelte Figure 9. I don't want not-run cells" (2026-10-23)
- why: The base releases suppress only the states tied to longer reasoning, so 25 of the 112 model-by-state cells of the per-state heatmap (Fig. 9, whole_states) were never run. Run them at each model's base strength on the same families and seeds, joined to the base rows by family, so every cell has a targeted-minus-random estimate.
- done when: A sealed complement release runs the 25 suppress cells (6,372 rows: gpt Im/Ve/Mo, nemotron Ex/Ve/Mo, qwen35 Re/Im/Ve/Mo, qwen36 Re/Ex/Ve/Mo, gemma Im/Ve/Mo, glm Ex/Pl/Ve/Mo, qwen330b Ve/Mo, lfm25 Ve/Mo); a new combined EFFECTS_WHOLE.json (r7-r17 plus the complement) has a cell_effects entry for every model x operation x state at base strength; primary, companions and dose response are unchanged in definition (complement cells are not efficiency-directed); GPU work before 2026-10-24 06:00 Europe/Rome within a 610 GPU-hour ceiling.
- decisions: whole-reasoning-design
- code: src/steering, scripts/steering, tests/steering
- [x] whole.complement_manifest builds the missing suppress cells from the parent's nominations and native rows; whole_effects joins a source's `complements` by family and keeps them out of arm levels; tests in tests/study/test_whole_complement.py; evidence: 4 new tests pass; dry run on r7/r8 data built 25 cells, all qualified, 6,372 rows, row fields identical to base suppress rows.
- [ ] Seal and launch the complement release with its combined analysis and staged paper export; evidence: report/whole-reasoning-20261021-r18/JOBS.json.
- [ ] Supervise to terminal accounting; report per-cell results and remaining unknown rows.

## outcome-steering-r2: Class-conditioned outcome steering on eight models
- status: active
- asked: "the paper must show how we can detect meaningful signal connected to each reasoning class and how we can leverage that to shape the effectiveness of the reasoning"; "let's start with OSS for control but then we need to do all models"; "Let's try to do our best and beat random"; plan approved 2026-10-23. (2026-10-23)
- why: Find, per model and reasoning class, the experts used more in wrong than in right attempts of the same question, and test whether nudging them down shortens reasoning without losing accuracy against matched random experts.
- done when: Per-class signal table (real score against 200 shuffled-label nulls, split-half replication, length correlation) for 7 classes x 8 models; GPT-OSS half-A grid (8 sets x 3 shifts x target + 2 random) and half-B confirmation in a sealed EFFECTS receipt with manipulation checks (nudged experts lose top-k slots, zero membership failures) and native accuracy near r1's 0.736; the other seven models only after the joint review.
- decisions: class-conditioned-outcome-design
- code: src/steering/outcome, scripts/steering, tests/steering/outcome, src/steering/stream.py, src/steering/pulse.py, src/steering/whole.py, src/steering/release.py
- [x] Native-id capture in the stream hooks (uint8 ids, float16 weights per reasoning token, hashed beside the receipt), per-row token cap, job-name prefix, locked store binding; evidence: tests/study/test_stream_batch.py 10 pass (new native-id test); commits a3aaa67, 607789b.
- [x] Capture manifest and release (roster of 300 shared questions, 6 attempts, 32,768 cap, stores on FAST); evidence: tests/outcome_steering/test_capture.py; dry run 1,800 rows for each of the 8 models.
- [x] Launch the capture release for all eight models; evidence: r1 (c68b233) failed allocated validation 61994112 on one pinned test message, no shard ran (VALIDATION_FAILED.json); r2 (cf82b90) validate 62112740, 55 oc-run shards, monitor 62113122 (JOBS.json).
- [ ] Class labels and per-class usage (classes.py) on each model's captures; queued as oc-derive 62113144-62113158 (snapshot cf82b90).
- [ ] Class-conditioned selection with window layers, footprint target, protections and two control draws; signal table.
- [ ] Sweep manifest, per-half analysis, pick rule and the GPT-OSS half-A and half-B releases; code in 8dcb2a0 (tests/outcome_steering/test_sweep.py), os-launch-A 62113159 queued after the GPT-OSS derive.
- [ ] Joint review of the GPT-OSS receipt; decide the other seven models' grid and ceiling.

## attempt-selection-by-routing: Routing-based selection among finished attempts
- status: open
- asked: "Grilling answer Q1 (a and c): use routing also to choose among finished attempts; Q15 approved as a CPU analysis on the captures." (2026-10-23)
- why: Test whether routing read from finished attempts picks a correct attempt more often than majority vote, a random pick, or the shortest or longest attempt.
- done when: ATTEMPT_SELECTION.json per model with family-bootstrap intervals for routing consensus, outcome-profile and class-mix picks against majority vote, random (expected accuracy), shortest and longest, on all questions and on mixed-outcome questions.
- decisions: class-conditioned-outcome-design
- code: src/steering/outcome/attempts.py, tests/steering/outcome
- [ ] attempts.py with the three routing picks and four baselines, leave-question-out profiles.
- [ ] Run on every model's captures after its class usage.

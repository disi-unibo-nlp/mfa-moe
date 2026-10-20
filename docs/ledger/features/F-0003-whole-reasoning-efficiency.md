# F-0003: Execute the whole-reasoning efficiency experiment
- status: active
- date: 2026-10-20
- request: "Alignment v2: stop the 512-token rescue and run the whole-reasoning (SteerMoE-style) efficiency experiment"
- purpose: Measure strict accuracy and reasoning length against matched random and native, retaining capped and missing outcomes as bounds.
- acceptance: 129 frozen families; up to 3096 GPT and 2838 Qwen rows, with predeclared availability drops; source-bound fresh execution; allocated CPU validation and passing 32-unit per-model canaries; six shared simultaneous contrasts and mandatory companions/mechanism/dose reporting. New GPU-hours at most 72, retained r3 0.346111 separately; CPU-only core-hours at most 16; science cutoff 2026-10-23 05:30 Europe/Rome.
- keywords: whole, efficiency, SteerMoE, routing, canary, budget
- paths: src/moe_exp/study, src/moe_exp/final_routing, tests/study, scripts/study
- decisions: D-0002

## Tasks

- [x] F-0003.1 Freeze r3 terminal accounting and supersede its unfinished 512-token plan; preserve all historical receipts; evidence: native sacct readback 2026-10-20, unchanged r3 TERMINAL.json, and report/whole-reasoning-20261020/README.md. Queue empty; r3 GPU 0.346111 hours and prior CPU 0.903333 core-hours remain recorded.
- [ ] F-0003.2 Implement global sign-stable RD nominations, layer/quintile/role controls and main/half-dose original-prompt manifests with audit flags.
- [x] F-0003.3 Implement dual operator, absolute prompt/reasoning scope, closure/recompute behavior and aggregate dose/full audit captures; focused tests including real short Unix socket; evidence: report/whole-reasoning-20261020/LOCAL_VALIDATION.json and FINAL_FOCUSED_TESTS.log. The >2048-token chunk, re-prefill, prompt Harmony suffix, native answer and generated-position-zero contracts pass; immutable-copy allocated verification remains F-0003.6.
- [x] F-0003.4 Correct bounded metric inference, missing benchmarks, signed matrices, occupancy companions and main log-total link; threaded scoring and replay-based reasoning counts; evidence: 17 effects regressions, 11 final focused checks and three replay/closeout checks passed, with six bounded primary exports even for entirely missing production. See LOCAL_VALIDATION.json, REPLAY_CLOSEOUT_TESTS.log and tests/study/test_effects.py. Fresh real analyses remain F-0003.8.
- [ ] F-0003.5 Record section 4 endpoints verbatim, debate/evidence and SD/MDE receipt before production; commit and materialize immutable release.
- [ ] F-0003.6 Run one allocated CPU suite with WORK staging, then per-model 32-unit canaries and declared fluency-only half-dose fallback if needed.
- [ ] F-0003.7 Submit 2 GPT/5 Qwen family-block shards, at most six running; enforce early fluency checks, 25-percent projection and absolute spend stop.
- [ ] F-0003.8 Extract with afterany, retain unknowns, analyze with 10000 shared draws; fresh corrected main/strong analyses from the same code.
- [ ] F-0003.9 Supervise to terminal accounting and report actual nominations, completeness, six primaries, companions, dose, exploratory link and occupancy with per-model verdicts.
- [ ] F-0003.10 Record real component walkthrough and final checks without manuscript edits or remote pushes.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|
| R3 engineering | Superseded 512-token rescue | 61554207/61554215 | Both FAILED 1:0; 312/311 seconds; 0.346111 GPU-hours | Native sacct readback 2026-10-20; original r3 TERMINAL.json |
| Focused whole contracts | Synthetic nomination/operator/prompt and all-missing scientific closeout fixtures | New design lacked contracts | 22 tests passed in 41.281 seconds; additional horizon/gate/signed checks and three replay/closeout checks passed | report/whole-reasoning-20261020/LOCAL_VALIDATION.json; repeated test counts are not added |
| Actual staging and entitlement | login05.leonardo.local, lmolfett | Required live authorization evidence | Native iscrc_miosr/normal association, empty queue, project quota20149688; no new jobs at snapshot | report/whole-reasoning-20261020/PREFLIGHT.json and HISTORICAL_ACCOUNTING.json |

# F-0004: Outcome-selected gentle steering experiment
- status: active
- date: 2026-10-22
- request: "define the implementation plan ... human readable code/repo"; approved plan 2026-10-22 (outcome-selected gentle steering on 3 models, length-limit baseline, three nudge strengths, scripts/ cleanup after launch).
- purpose: Test whether a small set of experts chosen by outcome (used more in wrong than right runs of the same question) and nudged down gently at every reasoning token shortens reasoning without losing accuracy, against matched random experts, unedited routing and a plain stop-thinking-earlier baseline.
- acceptance: For gpt, qwen330b and nemotron at delta 0.4/0.8/1.6: bounded target-random, target-native, random-native and native_repeat-native contrasts, the budget_50/70/85 curve and the target-versus-budget comparison in a sealed EFFECTS_OUTCOME.json; manipulation check shows the nudged experts used less with zero membership failures; all GPU work before 2026-10-23 23:00 Europe/Rome within an 82 GPU-hour ceiling.
- keywords: outcome, nudge, steering, routing, budget, early-exit, Beyond the Trace
- paths: src/moe_exp/outcome_steering, scripts/outcome_steering, tests/outcome_steering, src/moe_exp/final_routing/operator.py, src/moe_exp/study
- decisions: D-0003

## Tasks

- [x] F-0004.1 Outcome selection: per-attempt expert usage on campaign-v3 cohort B, within-question wrong-minus-right score, leave-one-question-out stability, shared-core and execution exclusions, mass truncation, holdout sets and matched random controls; tests in tests/outcome_steering/test_select.py; evidence: usage job 61764186 (gpt 276x24x32, qwen330b 219x48x128, nemotron 96x23x128); preview sets gpt 20 pairs 1.21% mass, nemotron 40 pairs 1.07%, qwen330b 40 pairs 0.39% (cap), no unstable fill.
- [x] F-0004.2 Shift operator in final_routing/operator.py and stream dispatch with shift counters; existing dual path unchanged; tests in tests/study/test_stream_batch.py; evidence: 9 tests in tests/study/test_stream_batch.py pass (5 new NudgeTests).
- [x] F-0004.3 Budget-prefix rows (cut native reasoning, forced closing suffix, answer-only allowance) in runtime/driver/extraction; tests in tests/study/test_whole.py; evidence: tests/outcome_steering/test_manifest.py budget-prefix tests pass.
- [x] F-0004.4 Manifest builder and release (OUTCOME.json sweep release reusing base natives); tests in tests/outcome_steering/test_manifest.py; evidence: local dry run built 1161/1152/1152 rows (gpt 12 budget rows source_unavailable); allocated job 61796625 ran 99 whole tests, one message assertion updated.
- [x] F-0004.5 Analysis: bounded contrasts, budget-curve comparison, manipulation check; tests in tests/outcome_steering/test_analysis.py; evidence: tests/outcome_steering/test_analysis.py passes.
- [ ] F-0004.6 READMEs for src/moe_exp/outcome_steering and scripts/outcome_steering; launch, supervise to terminal accounting, report per-model results.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|

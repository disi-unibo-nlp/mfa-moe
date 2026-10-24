# F-0013: Class-conditioned outcome steering on eight models
- status: active
- date: 2026-10-23
- request: "the paper must show how we can detect meaningful signal connected to each reasoning class and how we can leverage that to shape the effectiveness of the reasoning"; "let's start with OSS for control but then we need to do all models"; "Let's try to do our best and beat random"; plan approved 2026-10-23.
- purpose: Find, per model and reasoning class, the experts used more in wrong than in right attempts of the same question, and test whether nudging them down shortens reasoning without losing accuracy against matched random experts.
- acceptance: Per-class signal table (real score against 200 shuffled-label nulls, split-half replication, length correlation) for 7 classes x 8 models; GPT-OSS half-A grid (8 sets x 3 shifts x target + 2 random) and half-B confirmation in a sealed EFFECTS receipt with manipulation checks (nudged experts lose top-k slots, zero membership failures) and native accuracy near r1's 0.736; the other seven models only after the joint review.
- keywords: outcome, nudge, steering, class, capture, selection, Beyond the Trace
- paths: src/moe_exp/outcome_steering, scripts/outcome_steering, tests/outcome_steering, src/moe_exp/study/stream.py, src/moe_exp/study/driver.py, src/moe_exp/study/whole.py
- decisions: D-0006

## Tasks

- [x] F-0013.1 Native-id capture in the stream hooks (uint8 ids, float16 weights per reasoning token, hashed beside the receipt), per-row token cap, job-name prefix, locked store binding; evidence: tests/study/test_stream_batch.py 10 pass (new native-id test); commits a3aaa67, 607789b.
- [x] F-0013.2 Capture manifest and release (roster of 300 shared questions, 6 attempts, 32,768 cap, stores on FAST); evidence: tests/outcome_steering/test_capture.py; dry run 1,800 rows for each of the 8 models.
- [x] F-0013.3 Launch the capture release for all eight models; evidence: r1 (c68b233) failed allocated validation 61994112 on one pinned test message, no shard ran (VALIDATION_FAILED.json); r2 (cf82b90) validate 62112740, 55 oc-run shards, monitor 62113122 (JOBS.json).
- [ ] F-0013.4 Class labels and per-class usage (classes.py) on each model's captures; queued as oc-derive 62113144-62113158 (snapshot cf82b90).
- [ ] F-0013.5 Class-conditioned selection with window layers, footprint target, protections and two control draws; signal table.
- [ ] F-0013.6 Sweep manifest, per-half analysis, pick rule and the GPT-OSS half-A and half-B releases; code in 8dcb2a0 (tests/outcome_steering/test_sweep.py), os-launch-A 62113159 queued after the GPT-OSS derive.
- [ ] F-0013.7 Joint review of the GPT-OSS receipt; decide the other seven models' grid and ceiling.

## Results

| Metric | Dataset | Baseline | Result | Evidence |
|---|---|---|---|---|

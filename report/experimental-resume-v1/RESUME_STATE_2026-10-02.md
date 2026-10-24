# Resume state — 2026-10-02 (after the interrupted OpenAI Codex session)

The previous actual-Codex session (`01a0fce9-6ddb-7d81-a6c0-55aa6dc634fa`,
last event 16:37:48 CEST) stopped while waiting on three subagents.  Its
scientific state is preserved; this note records what is canonical, what was
finished during the resume, and what still gates the later stages.

## Canonical queued work

- **59216285** `st-eligible-micro-v3` — positive discovery pilot.  Manifest
  `CAUSAL_ELIGIBLE_MICRO_SERIAL_MANIFEST_v3.json`, seal
  `320ecca116c7ab6397740876fb680ad666b3df9696adc2ec8a30e1fc9154f052`.
  13 discovery families, 156 requests, six counterbalanced arms, two seeds,
  256-token cap, serial/eager, 2×A100, 3 h ceiling, `normal` QoS.
- **59216601** `st-deactivate-v4` — negative-bias/force-off necessity
  diagnostic.  Manifest `CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v4.json`,
  seal `85b7f90ae7295d854d32aa3bee935a5feeb53ee8810cc6657b34ce40028eeb3d`.
  Bound to the positive-v3 manifest and to the same 13 starts/control schedule.
- The independently submitted positive-v4 **59216368** is cancelled at zero
  elapsed and must not be resubmitted.  The v3 `_draft_unrun_superseded`
  manifest is a different artifact and must not be used.

Both jobs were pending for priority with zero elapsed at the last check.
The standing authorization in `RESOURCE_AUTHORIZATION_2026-10-02_v2.json`
authorizes the necessary measured stages without the earlier 10.5 GPU-hour
ceiling; per-stage complete pricing and artifact verification still apply.

## Verified by the resume audits

- All seals, driver/entry-driver hashes, prices, sbatch paths and output
  directories agree; no competing jobs or pre-existing outputs were found.
- The two jobs have 26 six-arm batches each and no GPU spend while pending.
- `build_eligible_micro_blind_frame_v3/v4.py` are complete, hash-pinned and
  produce `ARM_MAP.json` plus `BLIND_FRAME.json` after generation.
- `rate_eligible_immediate_semantics_v1.py` (Qwen3.8-27B, two arm-blind
  readers, 1,024-token judge cap) and
  `price_eligible_immediate_semantics_v1.py` are complete and tested.
- `raw_native` duplicate variability forbids exact hidden-state pairing or
  engine-equivalence claims; the design remains an intention-to-treat
  comparison with native duplicates as a noise control.

## Completed during this resume

- Added the missing
  `scripts/experimental_resume/analyze_eligible_immediate_semantics_v1.py`:
  it joins `ARM_MAP.json` to the two-reader ratings by blind ID, retains every
  assigned cell in the ITT denominator, reports both-reader-positive and
  at-least-one-reader endpoints, family-clustered paired intervals, native
  duplicate noise, realized dose and execution-position sensitivity.
- Extended `src/moe_exp/routing_control/analysis.py::paired_itt` with an
  explicit metric selection while preserving the previous default.
- Added the three missing execution wrappers:
  `price_eligible_immediate_semantics.sbatch`,
  `rate_eligible_immediate_semantics.sbatch`,
  `analyze_eligible_immediate_semantics.sbatch` (`KIND=v3|v4`).
- Added focused tests:
  `tests/experimental_resume/test_analyze_eligible_immediate_semantics_v1.py`
  (9 passing) and re-ran `tests/test_routing_control.py` (41 passing).
- Replaced the byte-identical 1,024-token driver copy with a fail-closed stub
  (`run_eligible_trajectory_serial_v1.py`); the old copy is preserved as
  `run_eligible_trajectory_serial_v1.py.superseded-v3-copy`.  This removes the
  risk of accidentally re-creating the queued 256-token positive pilot.
- Added the pool-general native-unit builder
  `prepare_dense_pool_units.py` and `prepare_dense_pool_units.sbatch`;
  smoke-tested on 2 mechanism families (240 sentences).  CPU job **59222035**
  completed `0:0` in 2m45s and sealed `dense-mechanism/MECHANISM_UNITS_v1.json`
  with 128 families / 128 questions / 14,869 sentences, seal
  `97adaf1f8719e05bdf87c92e6fdd8fa05649217963a3cc8fbce258b96c7aa917`.
  CPU job **59222416** is the frozen-v2-detector pass over those units
  (`MECHANISM_DETECTOR_AUDIT_v1.json`); it completed `0:0` in about three
  minutes and sealed 14,510 contiguous pairs with 3,715 candidate-to-verify
  fires across all 128 families, 108 approach-to-commit fires across 58
  families and 10 failed-check fires across 10 families, seal
  `bc3b839e50751ac87cc83faa63ff384726c3cf1d7588fa636f110189e58811ff`.
  CPU job **59222589** completed `0:0` in 1m30s and sealed the matching
  96-family utility units (`dense-utility/UTILITY_UNITS_v1.json`, 96 families /
  10,932 sentences, seal
  `89346a0811336dda70144a333077ada874f4cf83be0e8042b0327fc0f95151b3`).  All of
  these are free serial-partition jobs, not GPU work.

## Mechanism/utility pool data availability

The existing `v3_analysis/results-r2/qwen36/A/attempts.parquet` already contains
all 1,547 native questions.  Coverage is **48/48 discovery, 128/128 mechanism
and 96/96 utility representative questions**, and all six referenced trace
files exist (224 mechanism+utility rows checked, zero missing paths).  The
disjoint mechanism and utility pools therefore do **not** need new GPU native
generation; they need CPU unit extraction, a frozen-detector pass, a
two-reader start audit and then the separately sealed 1,024-token generation.
The first extraction step is job 59222035.

Before any mechanism start audit, freeze a deterministic selection rule over
these candidates (for example, at most three hash-ordered candidate starts per
family and supported transition, first two-reader-accepted start wins).  The
failed-check transition has only ten candidate families, so it remains
exploratory; the 1,024-token mechanism stage should use the supported
candidate-to-verify and approach-to-commit strata.

## Exact next execution order

The two automatic dependency chains are already queued, so no manual step is
needed after the pilot jobs finish successfully:

- v3: 59216285 → **59222756** frame → **59222757** price → **59222759**
  rating → **59222760** analysis.
- v4: 59216601 → **59222762** frame → **59222763** price → **59222764**
  rating → **59222765** analysis.

Each link uses `afterok`.  The price wrapper exits non-zero on a non-
`PASS_COMPLETE_STAGE` result, and the rating wrapper exits non-zero unless
`SUMMARY.json` exists, so a held price or partial rating stops the chain
instead of triggering a bad downstream job.  The next manual action is
monitoring the two pilot jobs and reading back the sealed analysis JSON in
`report/experimental-resume-v1/`.

1. Wait for 59216285 and 59216601 to finish; verify final Slurm state, exit
   code, elapsed/AllocTRES and both 26-batch output directories.  Fail closed
   on missing batches, seal mismatches or non-zero exit codes.
2. Read back `ARM_MAP.json` (156 assigned rows) and `BLIND_FRAME.json` (only
   gradeable rows) from the chained frame job and verify its printed counts.
3. Read back the chained price and proceed only on `PASS_COMPLETE_STAGE`; a
   hold status is a stop-and-reprice gate, not a reason to raise the walltime.
4. Read back the chained 2-GPU rating: 26 batch files, both readers, attempt
   accounting and `COMPLETE_ARM_BLIND_LLM_AUDIT`.
5. Read back the chained CPU analysis JSON in `report/experimental-resume-v1/`.
6. Freeze the pilot interpretation before unblinding anything else:
   a positive screen is a local discovery result only; it is not an ordered
   1,024-token trajectory, accuracy, token-utility or engine-equivalence
   result.

## Known later-stage gaps (do not launch yet)

- `run_eligible_trajectory_serial_v1.py` is byte-identical to the 256-token v3
  driver and would reject a 1,024-token manifest.  There is no trajectory
  manifest, sealer, price or sbatch.
- The dense seven-class analyzer and its eight tests exist, but there is no
  producer that joins generated batches/routes to sentence labels and two
  behavior votes, and no trajectory blind frame/arm map.
- No disjoint mechanism-family start pool exists yet (only the 13 discovery
  families), so a family-disjoint 1,024-token validation cannot launch before
  its parent pool, detector and two-reader accepted starts are frozen.
- The online detector v2.4/live-gate v2 are CPU-tested only; the serial
  ordered-pulse machinery was qualified for slots 0/512 and horizon 1,024, but
  the controller has not been GPU-qualified.
- The 16k utility stage has no frozen 96-family pool, no grading/cost
  accounting and no engine-bound price.  Its original 4.75 GPU-hour ceiling is
  not credible under the serial 8-token/s stress price; reprice after the
  pilot readout.

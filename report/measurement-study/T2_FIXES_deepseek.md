# T2 review fixes — Codex (deepseek)

Updated 2026-09-25. All edits are in `P/routing-study`; `P/repo` and
`src/moe_exp/correlation_pipeline` / `src/moe_exp/utils.py` were not touched. No
GPU job was submitted, no production bundle was frozen, and nothing was
committed or pushed. Final CPU suite: **Slurm job 58622773, COMPLETED 0:0,
00:01:50 elapsed, billing=2 (2 CPUs, 16 GiB, `lrd_all_serial`), 147 measurement
+ 74 guiding tests passed with `source_hashes_before == source_hashes_after`.**

## Findings to fix and what each fix does

### B1 — outcome sidecar schema and pointer (hidden by fabricated fixtures)

`campaign_v3_plan.read_outcomes` now:

- accepts either the content-addressed parquet/JSON sidecar **or the mutable
  pointer** (`outcomes_v3.latest.json`), verifies pointer sha256 == sidecar
  sha256 == manifest `sha256`, and re-checks every hash after the read;
- filters `corpus=='card-v3'` and the cohort model **before validation**, so the
  real 32,529-row historical slice (`capped`, `pending`, nullable
  `completion_tokens`) can never reject a card-v3 plan;
- accepts every documented `adjudication_status` including `capped`, accepts any
  non-empty termination string (vLLM `finish_reason`, e.g. `abort`) and maps
  `null` to the explicit string `unknown`;
- accepts nullable `Int64` for `sample_id` and `completion_tokens`, and rejects
  floats/bools instead of coercing them;
- keys rows by `(corpus, model, dataset, problem_id, sample_id)` and rejects
  duplicate identities.

Evidence: `test_outcome_sidecar_pointer_filters_and_explicit_nullable_rows`,
`test_real_cohort_and_real_outcome_contract` (reads the live pointer), and the
unchanged `test_unresolved_outcome_must_remain_null`.

### B2 — cohort binding (removed `outcome_sidecar_sha256`)

`load_cohort_manifest` recomputes the `manifest_sha256` that
`sample_cohorts.write` embeds (digest of `schema_version, model, seed,
label_total, per_b, floor, traces_sha256`) and pairs it with the cohort file
sha256; `validate_cohort_binding` re-reads both at capture/gate time. No code
reads `cohort['outcome_sidecar_sha256']` any more.

Evidence: `test_cohort_manifest_sha256_and_file_tampering_fail_closed`,
`test_capture_manifest_rejects_outcome_binding_and_checks_cohort`.

### B3 — cohort-B inclusion probability

Cohort A still requires `0 < p <= 1`; cohort B requires exactly `null` (paired
case-control selection has no population weight) and any other value fails
closed. Evidence: `test_cohort_b_null_probability_and_cohort_a_range`.

### B4 — outcomes are analysis-time only

- Capture manifests, validation bindings, capture results and packets contain no
  outcome field at all. `validation_binding` digests only
  `spec + code_binding + cohort_binding + maximum_input_tokens`, so cohorts A
  and B (and a re-planned A after the label gate) share one verify receipt.
- `build_manifests(..., outcomes_path=...)` is now an **optional planning-time
  consistency reference**: if a card-v3 row exists it must agree on
  `trace_sha256`, `completion_tokens`, `source_problem_id` and termination, and
  nothing from it is written into a manifest.
- `production_analyze` requires an explicit `--outcomes PATH` (pointer or
  sidecar) plus optional `--outcome-manifest`, joins rows at analysis time by
  `(corpus='card-v3', model, dataset, problem_id, sample_id)`, fails closed when
  a planned attempt has no row or the sidecar trace identity/tokens/termination
  disagree, and puts the sidecar `path/sha256/manifest/pointer` hashes into the
  analysis binding. A new outcome version therefore only needs a fresh analysis
  output root (enforced by `analysis-binding.json` / `complete.json`).

Evidence: `test_label_free_capture_binds_only_cohort_trace_and_code`,
`test_planner_bootstrap_receipt_capture_marker_gate_handshake`,
`test_ab_share_global_label_free_verifier_when_a_is_longer`,
`test_real_cohort_and_real_outcome_contract`.

### M1 — analysis bundle vs capture code binding

`production_analyze` no longer requires the capture manifest `code_binding` to
equal the analysis bundle hash. The capture code binding is recorded as
`capture_code_binding` provenance (artifact hashes and the capture gate already
established that the captures came from that code); analysis runs from its own
frozen bundle. Evidence:
`test_analysis_bundle_need_not_equal_capture_code_but_capture_code_is_recorded`.

### M2 — GLM HF-vs-vLLM differential

`capture_buffers.compare_hf_vllm_capture` no longer demands identical expert
sets at every (layer, token) for bf16 HF vs Triton/Marlin MoE. It now:

- requires zero flips where the **reference selection gap** (`min score among
  selected − max score among unselected`) is `>= eps` (`eps = 1e-3`);
- gates the near-tie flip rate (`gap < eps`) at `<= 1e-3`;
- reports per-layer statistics (tokens, agreeing tokens, flips, near-ties,
  near-tie flips/rates, per-layer effective-weight relative-L2 and max abs
  error) and enforces a per-layer weight relative-L2 bound of `5e-3` plus the
  existing global rounding check;
- still requires exact set equality when both sides declare
  `identical_kernels` (or the caller passes `identical_kernels=True`).

Evidence: `test_near_tie_flip_rate_gate_and_per_layer_weight_bound`,
`test_glm_differential_respects_audited_output_scaling_and_fp16_rounding`,
`test_differential_rejects_true_mismatch_long_trace_and_missing_provenance`.

### m1 — shard sizing with a fixed per-attempt cost

`fit_attempt_runtime` fits `startup + per_attempt + seconds_per_token * tokens`
from the two measured verification traces (least squares with the measured
startup; a negative intercept clamps to zero with a conservative rate).
`token_budget_shards` packs with that full cost, so many short attempts can no
longer overrun a shard; `production_plan --verification-receipt` fits from the
accepted verify receipt, or explicit measured values
`--startup-seconds/--per-attempt-seconds/--p90-seconds-per-token/--bytes-per-token`
are required. `verification_pair` now requires the short trace to differ in
completion tokens from the longest, because two equal token counts cannot
identify both parameters. Evidence: `test_runtime_fit_uses_fixed_plus_per_token_cost`,
`test_verification_receipt_drives_production_sizing`,
`test_shards_obey_runtime_bound_and_keep_atomic_traces`.

### m2 — verification walltime and drain margin

Bootstrap planning requires `--verification-seconds >= 18000` (5 h; the previous
7200 s cannot cover a 131k trace plus the GLM HF reference and two model loads)
and the measured path uses `max(5 h, 1.25 x fitted cost)`. The capture job's
drain guard now reserves `max(15 min, 1.25 x (per_attempt + rate x tokens)) + 300 s`
for the attempt that would start next (1800 s while unmeasured). Evidence:
`test_bootstrap_requires_explicit_bounded_walltime`,
`test_deadline_drains_before_new_attempt`.

### m3 — frozen bundle closure

`freeze_bundle` now also copies `src/moe_exp/routing_study/*.json`
(`checkpoint_files.json`, `reducer_provenance.json`), the `moe_margin_guiding`
and `moe_identity_guiding` packages, `pyproject.toml` and
`sbatch/routing_study_smoke.sbatch` — every path returned by
`src/moe_exp/routing_study/common.py:code_hashes()` — and fails closed when any
of them is missing. `test_real_worktree_bundle_satisfies_runtime_code_hashes_closure`
freezes the real worktree and asserts the frozen bundle reproduces the live
`code_hashes()` (29 files) exactly.

### m4 — A/B duplicate capture cost

Rows are deduplicated by attempt id inside any plan, and `--cohort all` (planned
after the label gate) captures each attempt once with both memberships, so both
cohort analyses read the same packet. The approved DAG plans B before labels and
A after them; an attempt selected by both is then captured twice with different
cohort variants, which cannot be shared without moving A's label-gated capture
behind B's. The measured duplicate cost is therefore stated in every
`ready.json` under `dedup`
(`shared_attempts`, `duplicated_completion_tokens`, plus the
`single_capture_set_plan` recommendation). For the real test cohort the
intersection is 0 attempts / 0 tokens. Evidence:
`test_duplicate_report_states_separate_plan_cost`,
`test_all_cohort_ready_receipt_serves_one_capture_set_for_both_cohorts`.

## Files changed

Code: `measurement_study/campaign_v3_plan.py`, `campaign_v3_capture.py`,
`capture_gate.py`, `capture_buffers.py`, `production_plan.py`,
`production_capture.py`, `production_analyze.py`, `bundle.py`;
`sbatch/production_analyze.sbatch` (v3 arguments: bundle, manifest dir, storage
root, cohort, output root, explicit outcomes).

Tests: `tests/measurement_study/test_campaign_v3_plan.py` (26 tests, including
the real cohort + real outcome contract), `test_bundle_v3.py` (8, including the
real worktree closure), `test_capture_v3_runner.py`,
`test_analysis_v3.py`, `test_vllm_capture_buffers.py`.

Docs: this file, `report/measurement-study/IMPLEMENTATION_codex.md` (the claim
that "no interface schema additions are required" was wrong and is corrected),
`report/measurement-study/CODEX_CHANGESET_v3.json`.

## Test results

| Job | State | Result |
|---|---|---|
| 58621376 | COMPLETED 1:0 | 1 failed, 146 measurement + 74 guiding passed; failure was a fixture still asserting the old set-equality differential; acceptance also withheld because source was edited mid-run |
| 58622122 | COMPLETED 1:0 | same fixture expectation, corrected again (native selection kept internally consistent) |
| **58622773** | **COMPLETED 0:0 (1:50, billing=2)** | **147 measurement + 74 guiding passed, source/test hashes unchanged before/after** |

Login-node light runs (correlation-client-3.11, no torch, under the 600 s CPU
limit): 112 passed in 57 s (measurement subset without torch modules), plus
focused plan (26), analysis (30), capture (8) and bundle (8) runs. Earlier
receipts for the pre-fix implementation are jobs 58612258, 58613554, 58614435,
58614723.

## Remaining risks and follow-ups

- **Monitoring gap:** `scripts/watch_slurm.py 58621376 --max-seconds 3600` could
  not arm in this Codex session — `binding()` raised
  `KeyError: 'TMUX_PANE'` because these tool calls execute outside the Codex TUI
  pane. The three CPU jobs were monitored with bounded foreground
  `squeue`/`sacct` polls and by reading the receipts/logs; no watcher was armed
  and no TUI delivery occurred.
- The GLM near-tie flip-rate differential and the fitted per-attempt runtime are
  validated on CPU fixtures only; real tensors and real timings need the seven
  verify-only GPU jobs (Claude's step, after re-review and freeze).
- The capture submitter must pass each manifest's `requested_seconds` as the
  job `--time` (>= 5 h for the verifier); the static sbatch directive is 16 h
  and does not read the manifest.
- A single `--cohort all` capture set (m4 dedupe) is covered by CPU
  planner/analysis tests; it has not been exercised on GPU.
- `production_analyze --outcomes` now changes the CLI; any live submitter that
  still calls the old argument set must be updated before use.

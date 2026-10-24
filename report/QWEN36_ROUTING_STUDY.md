# Qwen3.6 routing study: current status

The implementation is isolated in `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study`,
branch `study/qwen36-routing`, based on `9f16c55`. The original pilot checkout and its
queued jobs remain unchanged. No installed vLLM files were modified; no commit or push
was made.

**Latest GPU smoke 58436697 failed after 14m12s. The Marlin/V1 model loaded and
hooks were configured; the first forward hit a request-ID mismatch. The adapter
fix is implemented and passes **64 CPU tests**, with zero skips. Dispatch acceptance and
generation throughput remain pending.**
No benchmark accuracy result, empirical frontier, or full-study resource estimate
has been produced.

Current code-bound artifacts:

- [Package guide](../src/moe_exp/routing_study/README.md)
- [Frozen Marlin/V1 protocol](../experiments/qwen36-routing/marlin-v1/protocol.json)
- [CPU test report](../verification/routing-study/marlin-v1/cpu-tests.json)
- [Current verification record](../verification/routing-study/marlin-v1/verification.json)
- [Reviewed next-smoke proposal](../experiments/qwen36-routing/marlin-v1/smoke-resource-proposal.json)

## Observed smoke outcomes

The authorized smoke envelope was one GPU-hour, separate from the pilot's 32 GPU-hours.
All three allocations used two A100 GPUs, 16 CPUs and 120 GiB RAM on
`boost_usr_prod` / `boost_qos_dbg`.

| Job | Scheduler state | Allocated elapsed time | Outcome |
|---|---|---:|---|
| 58389406 | Failed | 9 seconds | Source adapter compared attempt-suffixed IDs as common problem metadata; corrected. |
| 58389615 | Cancelled | 7m37s | Stopped after startup logs and installed source exposed offline model-name rewriting; exact cached snapshot now accepted. Cleanup completed by 7m41s. |
| 58389922 | Failed | 9m12s | Checkpoint contents verified; forced Triton rejected the checkpoint's block-FP8 quantization scheme during model construction. |

Actual allocated GPU time totals **0.5656 GPU-hours**, or **0.5678 GPU-hours** when
conservatively including cancellation cleanup. These figures are GPU allocation
time, distinct from CINECA's local-hour billing units. Per-job receipts and completion
records are under [verification](../verification/routing-study/).

The third run passed all 60 tests then present on the compute node. Full checkpoint
content verification took **93.45 seconds**; its
[saved proof](../results/routing-study/smoke-58389922/smoke/checkpoint-validation.json)
binds the exact cached revision. The run resolved Qwen3_5MoeForConditionalGeneration,
FP8, BF16 activations, TP2, eager execution, eight-request capacity and no speculative
or prefix caching. Neither worker reached instrumented generation.

## Corrections now implemented and CPU verified

Problem metadata now uses source problem IDs while original attempt IDs remain
available for matching the 64 pilot problems and four engineering fixtures. Exact
question, gold and rendered-token consistency is still required.

The engine now pins the inspected **V1 GPUModelRunner** and **modular Marlin** backend.
Installed Marlin source supports this block-FP8 scheme with FP8 weights and BF16
activations (W8A16), accepting selected IDs and weights at the modular boundary.
The execution arithmetic, runner and expanded installed-source hashes are recorded
in backend fingerprints. Both calibration and generation must use this same path.
GPU acceptance is required before claiming it works; other runners/backends are rejected.

The failed run's rendezvous log showed `/tmp` despite a shell staging export. The
corrected launcher passes private `TMPDIR` explicitly after `srun`, pins the private
RPC directory, checks Python's actual temporary directory, and uses a short path
under `/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/` to respect Unix socket limits.
Checkpoint verification and each smoke stage now produce explicit progress evidence.

## Population preparation verified on compute

The source population contains 1,547 problems. The frozen split preserves all 64 pilot
problems and four fixtures in calibration. All attempts and detected duplicate groups
stay together. No holdout outcomes were used.

| Benchmark | Calibration problems | Tuning problems | Test problems |
|---|---:|---:|---:|
| MATH-500 | 300 | 100 | 100 |
| AIME24 | 18 | 6 | 6 |
| AIME25 | 18 | 6 | 6 |
| AMC23 | 24 | 8 | 8 |
| Minerva | 163 | 54 | 55 |
| Olympiad | 405 | 135 | 135 |

There are 2,788 saved calibration attempts, 1,236 planned tuning attempts per condition,
and 930 planned final-test attempts per condition. Existing population manifests are
historical artifacts bound to their trial code/backend; the next smoke prepares a fresh
manifest for Marlin/V1 with the same deterministic split rules.

## Next smoke: concrete proposal, not submitted

The corrected [batch script](../sbatch/routing_study_smoke.sbatch) passed shell syntax
and Slurm test-only validation. No corrected Marlin/V1 job is queued.

The proposal requests **one additional run of two A100 GPUs for up to 30 minutes**,
16 CPUs, 120 GiB RAM, a 2 GiB artifact check and an 8 GiB private staging allowance.
This is the same four-fixture/eight-request engineering workload, with one model load.
It would bring cumulative smoke allocation to at most **1.568 GPU-hours** including
previous cleanup. Approval requested: raise the total smoke ceiling from **1.0 to
1.6 GPU-hours**. A full 30-minute retry exceeds the remaining original envelope.
The pilot's 32 GPU-hours stay reserved.

On 2026-09-22, the last quota check showed WORK usage of 926.7G of 1T. The third run
produced about 14M of study artifacts and 2.5M of private staging data. These short-run
sizes do not establish long-run storage requirements. A full-study budget will only
be proposed after measured successful startup, throughput and dispatch validation.

The earlier synthetic PNG/PDF/SVG exports remain plotting verification fixtures,
not empirical Qwen3.6 results. Earlier frozen protocols and CPU reports remain intact
as historical provenance.

## Authorized corrected launch — 2026-09-23

The user approved the 1.6 GPU-hour cumulative smoke ceiling. Corrected Marlin/V1
job **58436697** was submitted and verified running on `lrdn0632`, with two A100s
and a 30-minute maximum. No smoke acceptance is available at submission time.
[Submission receipt](../verification/routing-study/submission-58436697.json).

## Separate metrics pilot: observed 2026-09-23

Read-only inspection confirms complete labeling (30,187 sentences, 64 traces, zero
unknown labels) and accepted NF4 routing capture (640,800 windows, all 40 layers,
554,738 retained tokens). Labeling job 58377369 completed. Job 58377404 completed
replay and capture acceptance, then timed out during downstream analysis; its
analysis manifest remains stale at `running` and no completed evaluation/report
exists. These are collection results, not validated predictive or intervention
findings. No pilot artifacts or jobs were changed.
[Observed pilot status](../verification/routing-study/pilot-status-2026-09-23.json).

## Latest smoke result — 58436697

The 2026-09-23 run completed population preparation, passed all 62 then-current
CPU tests, verified checkpoint contents, and loaded the model with modular Marlin
and the V1 runner. Engine startup took **590.8 seconds**. The first forward failed
because vLLM rewrote external request ID `0` to internal `0-afd64386`, which the
registered request-state map correctly rejected. No measured accuracy/token result
or accepted dispatch capture was produced.

The isolated adapter now uses the supported request-ID randomization switch,
globally unique IDs across waves/conditions/retries, and a returned-ID check.
The installed assignment method is covered by a new CPU regression. Current code
is frozen under [request-id-fix](../experiments/qwen36-routing/request-id-fix/).
No installed vLLM files or pilot artifacts were modified.

Cumulative smoke allocation including conservative cleanup is **1.0411 GPU-hours**
of the approved **1.6 GPU-hours**, leaving **0.5589 GPU-hours**. No replacement
job has been submitted. [Completion record](../verification/routing-study/completion-58436697.json).

Request-ID fix verification completed: **64 tests passed**, zero skips; current
code matches the frozen protocol. [CPU report](../verification/routing-study/request-id-fix/cpu-tests.json)
and [verification record](../verification/routing-study/request-id-fix/verification.json).

## Bounded Slurm wake-up notifications (2026-09-23)

`scripts/watch_slurm.py` observes one existing job every 30 seconds and sends a
terminal-state or monitoring-expiry event into the existing Codex tmux session.
It uses no new service and allocates no GPUs. Private HOME receipts distinguish
terminal delivery from agent acknowledgement. It checks job ownership, ignores
step records, prefers live requeued jobs over historical failures, and does not
treat missing accounting records as success. Composer drafts postpone delivery;
process replacement or a changed terminal prevents delivery. Each watcher exits
on delivery or its explicit deadline. This depends on the original Codex/tmux
session and login node staying alive; it is not an email delivery guarantee.

After future authorized submissions, arm the watcher immediately and retain its
foreground tool session ID. See `AGENTS.md` for the follow-up procedure. No new
study run is submitted as part of notification setup.

Verification: 10 targeted CPU tests passed. A live accounting query for job
58436697 returned FAILED, exit 1:0, elapsed 00:14:12. The watcher queued that
event in this Codex terminal and exited; a second invocation suppressed the
duplicate. The queued follow-up subsequently started a new agent turn automatically;
receipt was acknowledged, and final accounting and failure logs were rechecked.
The full event-to-agent wake-up path is therefore verified in this session.

## Completed pilot correlation analysis

The saved-data CPU recovery completed in jobs 58456386 and 58457518. Original
routing metrics and new reasoning-class dynamics are analyzed in
[PILOT_METRIC_CORRELATIONS.md](PILOT_METRIC_CORRELATIONS.md). Class-linked routing
associations were observed; no routing-correctness family survived BH 5%.
This is a 64-question NF4 replay pilot, not an FP8 intervention result.

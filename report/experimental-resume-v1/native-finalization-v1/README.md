# Native finalization diagnostic

This is a separate discovery experiment: the first eight families in the frozen
48-family discovery order, seeds 0 and 1, and two prompt arms (32 assignments).
Both arms use native routing, the Qwen3.6-35B-A3B-FP8 snapshot, the existing card
sampler, and a 16,384-token output cap. The treatment appends the exact instruction
in `MANIFEST.json` inside the original user message. Utility outcomes and the
frozen four-operator panel do not enter selection.

`MANIFEST.json` binds every assignment to its prompt IDs, model/tokenizer,
sampler, source hashes, family, seed and arm. `MODIFIED_PROMPTS.json` preserves the
modified text and IDs. The generator qualifies an engineering fixture against
uninstrumented native execution and checks a staggered four-request batch before
scientific requests. It captures native expert IDs and executed weights on both
TP ranks, checks their agreement, and commits each finished request separately.
Shared experts remain a separate branch. Unknown native selections are never
reconstructed from affinities.

`SAVED_AUDIT.json` reuses the reviewed 42 continuations, their prefix contexts,
existing classifications and saved per-rank target-mass/displacement telemetry.
It separates intermediate results, complete candidate expressions and already
checked answers. The offline reference annotations do not replace answer grading.
These 1,024-token continuations and the earlier 256-token analyses remain separate
from this experiment's original-prompt generations.

The analysis reuses frozen reasoning-boundary and strict/J1 contracts. Caps and
committed errors are operationally wrong; missing execution and unfinished grades
remain unknown. The verified closing marker is excluded from reasoning length.
Effects average the two paired seed differences within family, then weight each
family equally. Three endpoints share a simultaneous 95% family-bootstrap region
(50,000 draws, seed 20261005); missing-data bounds, small samples and zero-variance
components are reported. No equivalence or accuracy-retention claim is licensed.

The CPU verification receipt records **36 passing tests** (17 new contract tests
and 19 existing panel/controller/dose tests), run in a project-owned test-only
environment inheriting scientific dependencies. The inference environment was not
modified. Frozen and working-package tests run in separate processes within the
same CPU allocation to preserve their distinct import contracts.

## Operations and continuation

Use the installed CINECA procedure from the actual `lmolfett` login shell. CPU
jobs create sealed readiness; `saldo` is unavailable on compute nodes and the
existing compute-to-login accounting probe failed. Each dispatch is a finite
native call with a fresh balance, queue and association check:

```bash
bash scripts/experimental_resume/launch_native_finalization_v3.sh generation
# After the generation-dependent CPU job creates J1_READY.json:
bash scripts/experimental_resume/launch_native_finalization_v3.sh j1
# Read actual accounting, including unsuccessful preparation allocations:
bash scripts/experimental_resume/launch_native_finalization_v3.sh account
```

The generation request is two GPUs, 16 CPUs, 120 GiB and eight hours on
`boost_usr_prod/normal`. CPU preparation, measurement and finalization each use
two CPUs, 32 GiB and at most one hour on `lrd_all_viz/normal`. Frozen J1 requests
are exactly priced after blind strict grading, with at most one existing two-GPU,
six-hour J1 allocation for this 32-assignment cycle. These are request ceilings,
not measured consumption or a new quota subdivision.

`submissions/` contains attempted, committed and verified job receipts. Repeating
a dispatch reuses a matching receipt; an unresolved submission attempt stops for
reconciliation. Completed generation receipts are never rerun. An interrupted
generation attempt without a receipt remains unknown and requires explicit
reconciliation before recovery. There are no persistent dispatchers or automatic
downstream experiments.

Three unsuccessful preparation jobs preceded scientific sealing: `59416713`
(missing exported `_module_raw`, two seconds), `59417057` (`saldo` unavailable on
compute, five seconds), and `59417570` (test namespace collision, 18 seconds).
The last also exposed Slurm replacing `TMPDIR` inside `srun`; v2 wrappers explicitly
set user-owned staging in the step and the entrypoints check it. The versioned
correction receipts preserve these observations. No scientific results were
affected by those setup failures.

The run directory is
`/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/native-finalization-v1`.
Final tables and accuracy-versus-length PNG/PDF figures are written to `analysis/`
after grading. `STUDY_INDEX-*.json` snapshots link prior study indices, artifact
seals, observed allocation costs and unresolved uncertainty. Costs recorded while
a CPU allocation is still running are explicitly provisional.

No external API or private-source transfer is part of this cycle. The legacy
credential must be rotated before any future external API use.

## Qualification correction and health events

Job `59418248` failed during the captured engineering fixture, before any
scientific attempt. Importing the frozen validator module installed a class
forward hook that expected a steering controller on an already loaded native
engine. Its dependent measurement job `59418250` preserved all 32 outcomes as
unknown. The observed parent costs are 3.257222 billing core-hours and 0.392222
GPU-hours, including earlier preparation allocations.

`correction-v2/` preserves a separately sealed correction, manifest and design
reconciliation. The v2 worker pins the native bound instance forward before
importing the same validators. Every original sealed source and result remains
unchanged. CPU job `59421210` passed three regression tests plus the original 36
tests and verified the same prompts, families, seeds, order and sampler. The
corrected manifest seal is
`6cd588eebb3edb4096613fbd49cf25ca6a51f5d6301724e0475784703ac2e7ba`.

Corrected generation job `59421662` and dependent measurement job `59421664`
retain the authorized resource ceilings. Continue this version with
`bash scripts/experimental_resume/launch_native_finalization_v4.sh j1` after its
measurement readiness succeeds. The corrected accounting adapter includes the
preserved parent allocations and prevents double counting.

`native_finalization_health_v1.py` observes local kernel file events on the
writing compute node within the existing allocation. A finite native scheduler
readiness wait precedes attachment. Qualification, errors, generation completion,
grading readiness and the observer deadline produce sealed one-shot health
receipts; no repeated Slurm queries, new GPU allocations or persistent service
are required. A bounded tool callback sends the event to this existing Codex
task. Initial and termination-triggered snapshots cover events that finish before
the observer attaches. The qualification observer was confirmed armed on
`lrdn0608.leonardo.local` for job `59421662`.

That observer subsequently delivered `ENGINEERING_ENDED_WITHOUT_QUALIFICATION`.
Job `59421662` passed the native-forward guard but failed the exact replay
comparison, again before scientific attempts. Measurement `59421664` preserved
32 unknown assignments. Final observed costs through this failure are 6.133889
billing core-hours and 0.744444 GPU-hours, including the preserved parent jobs.
No answer-accuracy or length effect has been observed.

`correction-v3/` adds owned snapshots and sealed replay evidence for two native
repeats and an instrumented repeat, with separate token, ordered-route and
route-membership comparisons. It retains both exact replay gates. This is an
engineering diagnostic to distinguish native replay instability from an
instrumentation effect; it does not alter scientific prompts, seeds, sampler,
cap or grading. Use `launch_native_finalization_v5.sh` for this version, and pass
`correction-v3` as the third argument of the finite health wait wrapper.

CPU preparation `59427818` passed 42 tests and reconciled the unchanged design.
Its manifest seal is
`02400561137e8ae86087c54f5ee685eb4cb71a40a953efb6c1827be7d93ef925`.
GPU replay/generation `59428024` and measurement `59428026` are submitted. A
queued job cannot accept the node-local observer: `scontrol wait_job` waits for
usable nodes of an existing allocation, and `srun --jobid` rejects pending jobs.
This limitation does not indicate an execution failure.

LEONARDO configures `SlurmUser=slurm`, so documented `strigger` event registration
is unavailable to this user. Start signal `59428759` uses the authorized CPU
partition with two CPUs, 32 GiB and a one-minute ceiling, dependent on the GPU
job starting. A single native `sbatch --wait` client waits for that short signal;
the assistant does not repeatedly query the queue. It then attaches the
node-local qualification observer. Qualification has a 30-minute observer
deadline, measurement a bounded CPU deadline, and generation its allocation
deadline. Observers never automatically retry or cancel scientific jobs.
`health-start.receipt.json` includes the extra CPU allocation in accounting.

`saved-cases.csv` summarizes the seven already reviewed cases (six completions
each); its categories remain case-level observations. `saved-review.csv` retains
the 42 rows and offline candidate character offsets. These tables do not replace
final-answer grading or establish population frequencies.

## Current sealed execution: correction v4

The v3 engineering job `59428024` ended before any scientific attempt. Its
sealed replay evidence showed identical 64-token outputs across two native
runs and the instrumented run. The native runs themselves differed in 2,829
expert slots; requiring identical routes across executions therefore falsely
rejected a native control. Both instrumented rank captures exactly matched the
routes returned for that execution, with no rank disagreement in IDs or weights.
The cause of native route variation remains unresolved.

`correction-v4/` changes that demonstrated qualification defect only: exact
token replay, faithful aligned capture within each execution, normalized
executed weights, rank agreement and staggered batch checks remain required.
It preserves all three replay arrays, the prior seals and the unchanged
32-assignment design. No affected scientific results existed.

Preparation job `59429841` passed all **50 tests** and sealed the corrected
manifest as
`b5f8f361cbe7ddffbdfa1ced12c4a3fba52a73fe958a52b73caaeb367eb0e008`.
Generation `59430300` and dependent measurement `59430302` are verified.
The current GPU request is capped at 7h20; together with the 2,006 seconds
spent in earlier engineering allocations, generation allocation ceilings
remain below the original eight-hour allowance. The CPU start signal
`59430380` is included in the operational ledger and excluded from outcomes.

Use the current finite dispatch after its measurement readiness succeeds:

```bash
bash scripts/experimental_resume/launch_native_finalization_v6.sh j1
bash scripts/experimental_resume/launch_native_finalization_v6.sh account
```

For event-driven attachment use
`wait_native_finalization_health_v2.sh JOB PHASE correction-v4`. The v2 wrapper
fits the shortened allocation. Queued start notifications use a single blocking
`sbatch --wait` client; active jobs use local kernel file events. There is no
periodic queue or file polling. Earlier version commands above document history
and should not dispatch additional scientific executions.

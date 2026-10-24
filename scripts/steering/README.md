# scripts/steering

Launchers for the routing interventions: edit the routers while a model reasons, then measure what changed. Python decides what runs (`src/steering`); these scripts decide where and in which order.

| Script | What it does |
| --- | --- |
| `pulse.sh SHARDS HOURS MODEL...` | GPU jobs of the pulsed design, one per shard of question families. |
| `pulse_analyze.sh OUT` | CPU jobs: the outcome tables, then the effects of the main and strong designs into `OUT`. |
| `whole.sh KIND NAME TESTS [OPTION...]` | Seals a complement, relink or outcome release (`python -m steering.release`; `--cutoff` is required for complement and outcome, a relink takes its parent's cutoff), then runs `launch.sh`. `AFTER="JOB ..."` passes running jobs on to `launch.sh`, e.g. a relink's parent combine. |
| `launch.sh NAME [JOB...]` | Submits the jobs of a sealed release: the unit tests (sealed in `CPU_VALIDATION.json`), then the manifests, the GPU shards, one extraction per model, the final analysis and the staged paper export. Refuses to submit anything when the jobs' time limits exceed the release's CPU or GPU ceilings. A failed launch cancels what it queued and keeps their IDs in `JOBS.cancelled-TIME.txt`. `DRY_RUN=1` prints the sbatch lines only. |
| `outcome_usage.sh` | CPU job: how often each expert is used in every cohort-B attempt (input of an outcome release). |
| `outcome_next.sh EFFECTS NAME TESTS [OPTION...]` | Applies the declared extension rule to an outcome analysis and, if it is met, seals and launches the next release. |
| `gpu.sh MODULE ARGS...` | The job script of a GPU stage: one Python module on 2 GPUs. |

## Pulsed steering

This gives the main and strong doses of the paper.

```bash
sbatch -J pulse-plan scripts/slurm/cpu.sh steering.pulse plan   # CPU: writes driver/MODEL/ASSIGNMENTS.json
scripts/steering/pulse.sh 2 12 gpt                            # 2 shards of 12 hours for GPT-OSS
sbatch -t 12:00:00 -J fr-gpt-0 scripts/steering/gpu.sh steering.pulse run --model gpt --shard 0/2 --design strong   # strong dose, after plan-strong
scripts/steering/pulse_analyze.sh /new/folder
```

## LFM2.5

LFM2.5 has no server script. Its stages run in this order, and `scripts/routing/lfm.sh` follows them:

1. `sbatch scripts/steering/gpu.sh steering.pulse native --shard I/N` generates its traces.
2. `sbatch scripts/slurm/cpu.sh routing.lfm materialize` picks and prepares the sentences to label (CPU).
3. `sbatch scripts/steering/gpu.sh steering.pulse annotate --shard I/N` labels them.
4. `sbatch scripts/steering/gpu.sh steering.pulse lfm-replay --shard I/N` records its routers.

## Whole-reasoning steering and the dose sweep

```bash
scripts/steering/whole.sh complement NAME TESTS --cutoff 2026-10-24T06:00   # seal report/NAME, then submit its jobs
DRY_RUN=1 scripts/steering/launch.sh NAME                                   # look at the jobs first
```

A release is the folder `report/NAME`: it fixes the models, strengths, shards, cut-off time and efficiency link of one run, and it holds a frozen copy of the code. Its jobs run from that copy in `mfa-moe/releases/<commit>`, not from the working tree. `whole.sh` needs a clean `src`, `tests` and `scripts` tree. `scripts/slurm/ledger.sh whole-` adds up what the releases cost.

## Outcome-selected nudge

This is not in the paper.

```bash
scripts/steering/outcome_usage.sh
scripts/steering/whole.sh outcome outcome-steering-20261022-r1 TESTS --shards 3 --cutoff 2026-10-23T23:00
sbatch scripts/steering/outcome_next.sh report/RELEASE/analysis/EFFECTS_OUTCOME.json NEXT_NAME TESTS --cutoff TIME --models gemma,glm
```

Paper use: pulsed steering gives figure `routing` and its tables, whole-reasoning steering gives `whole_states`, `whole_dose` and their tables. See [`src/steering`](../../src/steering/README.md).

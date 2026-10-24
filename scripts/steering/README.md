# scripts/steering

Launchers for the routing interventions: edit the routers while a model reasons, then measure what changed. Python decides what runs (`src/steering`); these scripts decide where and in which order.

| Script | What it does |
| --- | --- |
| `pulse.sh SHARDS HOURS MODEL...` | Pulsed steering: submits `SHARDS` GPU jobs per model, each a disjoint set of question families. A rerun resumes. |
| `pulse_analyze.sh OUT` | CPU jobs: the outcomes of every continuation (main design for 8 models, strong for 4), then the effects of both designs into `OUT/main` and `OUT/strong`. `OUT` must not exist. |
| `whole.sh KIND NAME TESTS [OPTION...]` | Seals a whole-reasoning release (`steering.release`), then runs `launch.sh`. `KIND` is `complement`, `relink` or `outcome`; `TESTS` is the number of tests that passed locally. |
| `launch.sh NAME [JOB...]` | Submits the jobs of a sealed release: tests, manifests, GPU shards, one extraction per model, the final analysis and the staged paper export. A failed launch cancels what it queued. `DRY_RUN=1` only prints the `sbatch` lines. |
| `outcome_usage.sh` | CPU job: how often each expert is used in every cohort-B attempt (the input of an outcome release). |
| `outcome_next.sh EFFECTS NAME TESTS [OPTION...]` | Applies the rule for extending the outcome experiment to a finished analysis and, if it holds, seals and launches the next release. |
| `gpu.sh MODULE ARGS...` | The GPU job the launchers submit: one Python module, 2 GPUs, 4 hours unless you pass `-t`. |
| `tests.sh` | The job every release runs first: the steering and routing unit tests. |

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

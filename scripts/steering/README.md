# scripts/steering

Launchers for the routing interventions: edit the routers while a model reasons, then measure what changed. Python decides what runs (`src/steering`); these scripts decide where and in which order.

| Script | What it does |
| --- | --- |
| `pulse.sh SHARDS HOURS MODEL...` | GPU jobs of the pulsed design, one per shard of question families. |
| `pulse_analyze.sh OUT` | CPU jobs: the outcome tables, then the effects of the main and strong designs into `OUT`. |
| `whole.sh KIND NAME TESTS [OPTION...]` | Seals a complement, relink, outcome, capture, sweep or continuation release (`python -m steering.release`; `--cutoff` is required for every kind but a relink, which takes its parent's cutoff; a continuation also needs `--parent`, a sweep `--half`, `--capture` and `--models`), then runs `launch.sh`. `AFTER="JOB ..."` passes running jobs on to `launch.sh`, e.g. a relink's parent combine or a continuation's parent extractions. |
| `launch.sh NAME [JOB...]` | Submits the jobs of a sealed release: the unit tests (sealed in `CPU_VALIDATION.json`), then the manifests, the GPU shards, one extraction per model, the final analysis and the staged paper export. Job names start with the release's `job_prefix`: `whole` normally, `oc` for a capture, `os` for an outcome sweep (a capture release has no final analysis). Refuses to submit anything when the jobs' time limits exceed the release's CPU or GPU ceilings. A failed launch cancels what it queued and keeps their IDs in `JOBS.cancelled-TIME.txt`. `DRY_RUN=1` prints the sbatch lines only. |
| `outcome_usage.sh` | CPU job: how often each expert is used in every cohort-B attempt (input of an outcome release). |
| `outcome_derive.sh NAME MODEL...` | Queues, for a launched capture release, one CPU job per model after that model's extraction: class usage, expert sets, signal table and attempt selection (`steering.outcome derive`), run from the release snapshot. `DRY_RUN=1` prints the sbatch lines. |
| `outcome_pick.sh HALF_A MODEL NAME TESTS [OPTION...]` | CPU job, submitted behind half A's analysis: applies the frozen pick rule (`steering.outcome pick`) and, if it picks cells, seals and launches half B with `whole.sh sweep`. |
| `outcome_next.sh EFFECTS NAME TESTS [OPTION...]` | Applies the declared extension rule to an outcome analysis and, if it is met, seals and launches the next release. |

## Pulsed steering

This gives the main and strong doses of the paper.

The GPU stages run through `scripts/slurm/run_module_gpu.sbatch MODULE ARGS...` (one Python module on 2 GPUs).

```bash
sbatch -J pulse-plan scripts/slurm/run_module_cpu.sbatch steering.pulse plan   # CPU: writes driver/MODEL/ASSIGNMENTS.json
scripts/steering/pulse.sh 2 12 gpt                            # 2 shards of 12 hours for GPT-OSS
sbatch -t 12:00:00 -J fr-gpt-0 scripts/slurm/run_module_gpu.sbatch steering.pulse run --model gpt --shard 0/2 --design strong   # strong dose, after plan-strong
scripts/steering/pulse_analyze.sh /new/folder
```

## LFM2.5

LFM2.5 has no server script. Its stages run in this order, and `scripts/routing/lfm.sh` follows them:

1. `sbatch scripts/slurm/run_module_gpu.sbatch steering.pulse native --shard I/N` generates its traces.
2. `sbatch scripts/slurm/run_module_cpu.sbatch routing.lfm materialize` picks and prepares the sentences to label (CPU).
3. `sbatch scripts/slurm/run_module_gpu.sbatch steering.pulse annotate --shard I/N` labels them.
4. `sbatch scripts/slurm/run_module_gpu.sbatch steering.pulse lfm-replay --shard I/N` records its routers.

## Whole-reasoning steering and the dose sweep

```bash
scripts/steering/whole.sh complement NAME TESTS --cutoff 2026-10-24T06:00   # seal report/NAME, then submit its jobs
DRY_RUN=1 scripts/steering/launch.sh NAME                                   # look at the jobs first
```

A complement whose shards could not all start before its cutoff gets a continuation that runs what is left. Read its plan first; `AFTER` names the parent's extraction jobs, which the plan lists:

```bash
python -m steering.followup report/whole-reasoning-20261021-r18          # plan of the follow-up of a complement: rows by state, shards, GPU ceiling
AFTER="JOB ..." scripts/steering/whole.sh continuation whole-reasoning-20261021-r21 TESTS --parent whole-reasoning-20261021-r18 --cutoff 2026-10-27T06:00
```

A release is the folder `report/NAME`: it fixes the models, strengths, shards, cut-off time and efficiency link of one run, and it holds a frozen copy of the code. Its jobs run from that copy in `mfa-moe/releases/<commit>`, not from the working tree. `whole.sh` needs a clean `src`, `tests` and `scripts` tree. `scripts/slurm/billing_hours_used.sh whole-` adds up what the releases cost.

## Outcome-selected nudge

This is not in the paper.

```bash
scripts/steering/outcome_usage.sh
scripts/steering/whole.sh outcome outcome-steering-20261022-r1 TESTS --shards 3 --cutoff 2026-10-23T23:00
sbatch scripts/steering/outcome_next.sh report/RELEASE/analysis/EFFECTS_OUTCOME.json NEXT_NAME TESTS --cutoff TIME --models gemma,glm
```

## Outcome steering r2 (class-conditioned)

Not in the paper. Ledger F-0013, F-0014, D-0006; code in [`src/steering/outcome`](../../src/steering/outcome/README.md).

```bash
scripts/steering/whole.sh capture NAME TESTS --shards gpt:3,glm:15 --cutoff 2026-10-30T23:00   # native attempts with routing, jobs oc-*
scripts/steering/outcome_derive.sh NAME gpt glm                                              # CPU chain per model after its extraction
scripts/steering/whole.sh sweep NAME_A TESTS --half A --capture NAME --models gpt --shards 15 --cutoff 2026-10-30T23:00   # jobs os-*
sbatch --dependency=afterok:ANALYSIS_JOB scripts/steering/outcome_pick.sh NAME_A gpt NAME_B TESTS --capture NAME --shards 2 --cutoff 2026-10-30T23:00
```

The capture and sweep stores go to FAST (`report/NAME/stores` is a link). `ANALYSIS_JOB` is the `os-analyze` line of `report/NAME_A/JOBS.txt`.

Paper use: pulsed steering gives figure `routing` and its tables, whole-reasoning steering gives `whole_states`, `whole_dose` and their tables. See [`src/steering`](../../src/steering/README.md).

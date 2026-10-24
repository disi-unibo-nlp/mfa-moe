# scripts/steering

Launchers for the routing interventions: edit the routers while a model reasons, then measure what changed. Python decides what runs (`src/steering`); these scripts decide where and in which order. A model is passed as its Hugging Face id; the study folder, the release folder and every output folder are arguments, never a default.

| Script | What it does |
| --- | --- |
| `run_pulsed_steering.sh STUDY_DIR HF_MODEL SHARDS HOURS [main\|strong]` | GPU jobs of the pulsed design for one model, one per shard of question families. It first checks that the study holds the model's manifest of the design. |
| `analyze_pulsed_steering.sh STUDY_DIR OUT_DIR` | CPU jobs: the outcome tables of every model that has a manifest in the study (`driver/MODEL/ASSIGNMENTS.json` for the main design, `STRONG.json` for the strong one), then the effects of each design into `OUT_DIR/main` and `OUT_DIR/strong`. |
| `seal_release.sh KIND NAME TESTS [OPTION...]` | Seals a complement, relink, outcome, capture, sweep or continuation release (`python -m steering.release`; `--cutoff` is required for every kind but a relink, which takes its parent's cutoff; a continuation also needs `--parent`, a sweep `--half`, `--capture` and `--models`). It submits nothing. |
| `submit_release_jobs.sh RELEASE_DIR [JOB...]` | Submits the jobs of a sealed release: the unit tests (sealed in `CPU_VALIDATION.json`), then the manifests, the GPU shards, one extraction per model, the final analysis and the staged paper export. It reads what to submit from `python -m steering.whole_reasoning plan` (JSON). Job names start with the release's `job_prefix`: `whole` normally, `oc` for a capture, `os` for an outcome sweep (a capture release has no final analysis). JOB... are running jobs the final analysis also waits for, e.g. a relink's parent combine or a continuation's parent extractions. Refuses to submit anything when the jobs' time limits exceed the release's CPU or GPU ceilings. A failed launch cancels what it queued and keeps their IDs in `JOBS.cancelled-TIME.txt`. `DRY_RUN=1` prints the sbatch lines only. |
| `measure_expert_usage.sh REPORTS_DIR OUT_DIR HF_MODEL...` | CPU job: how often each expert is used in every cohort-B attempt (input of an outcome release). `REPORTS_DIR` holds the base release `whole-reasoning-20261021-r7`. |
| `derive_outcome_sets.sh RELEASE_DIR HF_MODEL...` | Queues, for a launched capture release, one CPU job per model after that model's extraction: class usage, expert sets, signal table and attempt selection (`steering.outcome derive`), run from the release snapshot. `DRY_RUN=1` prints the sbatch lines. |
| `pick_and_launch_half_b.sbatch HALF_A_DIR HF_MODEL NAME TESTS [OPTION...]` | CPU job, submitted behind half A's analysis: applies the frozen pick rule (`steering.outcome pick`) and, if it picks cells, seals half B with `seal_release.sh sweep` and submits it. |
| `decide_and_launch_next.sbatch EFFECTS NAME TESTS [OPTION...]` | CPU job: applies the declared extension rule to an outcome analysis and, if it is met, seals the next release with `seal_release.sh outcome` and submits it. |

## Pulsed steering

This gives the main and strong doses of the paper. The GPU stages run through `scripts/slurm/run_module_gpu.sbatch MODULE ARGS...` (one Python module on 2 GPUs). `STUDY_DIR` is the routing study, `report/final-routing-20261015-r1`.

```bash
sbatch -J pulsed-plan scripts/slurm/run_module_cpu.sbatch steering.pulsed_steering plan        # CPU: writes driver/MODEL/ASSIGNMENTS.json
scripts/steering/run_pulsed_steering.sh STUDY_DIR openai/gpt-oss-20b 2 12                      # 2 shards of 12 hours for GPT-OSS
for m in $(python -m models.roster); do scripts/steering/run_pulsed_steering.sh STUDY_DIR $m 2 12; done   # every model
scripts/steering/run_pulsed_steering.sh STUDY_DIR openai/gpt-oss-20b 2 12 strong               # strong dose, after plan-strong
scripts/steering/analyze_pulsed_steering.sh STUDY_DIR /new/folder
```

`scripts/slurm/billing_hours_used.sh` adds up what these jobs cost (they are named `fr-MODEL-I-of-N`). The study folder reaches the jobs as `STUDY_DATA_ROOT`, which `store.STUDY` reads when a module starts, so the Python modules of this folder have no `--study` option.

## LFM2.5

LFM2.5 has no server script. Its GPU stages are `steering.lfm25_generation`, and they run in this order together with the CPU stages of `scripts/routing/`:

1. `sbatch scripts/slurm/run_module_gpu.sbatch steering.lfm25_generation native --shard I/N` generates its traces.
2. `sbatch scripts/slurm/run_module_cpu.sbatch routing.lfm materialize` picks and prepares the sentences to label (CPU).
3. `sbatch scripts/slurm/run_module_gpu.sbatch steering.lfm25_generation annotate --shard I/N` labels them.
4. `sbatch scripts/slurm/run_module_gpu.sbatch steering.lfm25_generation replay --shard I/N` records its routers.

## Whole-reasoning steering and the dose sweep

```bash
scripts/steering/seal_release.sh complement NAME TESTS --cutoff 2026-10-24T06:00   # seal report/NAME
DRY_RUN=1 scripts/steering/submit_release_jobs.sh report/NAME                      # look at the jobs first
scripts/steering/submit_release_jobs.sh report/NAME                                # submit them
```

A complement whose shards could not all start before its cutoff gets a continuation that runs what is left. Read its plan first; the parent's extraction jobs, which the plan lists, go to `submit_release_jobs.sh` so the combine waits for them:

```bash
python -m steering.plan_continuation report/whole-reasoning-20261021-r18          # plan of the follow-up of a complement: rows by state, shards, GPU ceiling
scripts/steering/seal_release.sh continuation whole-reasoning-20261021-r21 TESTS --parent whole-reasoning-20261021-r18 --cutoff 2026-10-27T06:00
scripts/steering/submit_release_jobs.sh report/whole-reasoning-20261021-r21 JOB...
```

A release is the folder `report/NAME`: it fixes the models, strengths, shards, cut-off time and efficiency link of one run, and it holds a frozen copy of the code. Its jobs run from that copy in `mfa-moe/releases/<commit>`, not from the working tree. `seal_release.sh` needs a clean `src`, `tests` and `scripts` tree. `scripts/slurm/billing_hours_used.sh whole-` adds up what the releases cost.

## Outcome-selected nudge

This is not in the paper. Its usage tables are read from `steering.outcome.select.BULK`, so `measure_expert_usage.sh` with a new `OUT_DIR` checks a measurement but does not feed a release.

```bash
scripts/steering/measure_expert_usage.sh report OUT_DIR openai/gpt-oss-20b Qwen/Qwen3-30B-A3B
scripts/steering/seal_release.sh outcome outcome-steering-20261022-r1 TESTS --shards 3 --cutoff 2026-10-23T23:00
scripts/steering/submit_release_jobs.sh report/outcome-steering-20261022-r1
sbatch scripts/steering/decide_and_launch_next.sbatch report/RELEASE/analysis/EFFECTS_OUTCOME.json NEXT_NAME TESTS --cutoff TIME --models nvidia/Gemma-4-26B-A4B-NVFP4,zai-org/GLM-4.7-Flash
```

## Outcome steering r2 (class-conditioned)

Not in the paper. Code in [`src/steering/outcome`](../../src/steering/outcome/README.md).

```bash
scripts/steering/seal_release.sh capture NAME TESTS --shards gpt:3,glm:15 --cutoff 2026-10-30T23:00   # native attempts with routing, jobs oc-*
scripts/steering/submit_release_jobs.sh report/NAME
scripts/steering/derive_outcome_sets.sh report/NAME openai/gpt-oss-20b zai-org/GLM-4.7-Flash       # CPU chain per model after its extraction
scripts/steering/seal_release.sh sweep NAME_A TESTS --half A --capture NAME --models gpt --shards 15 --cutoff 2026-10-30T23:00
scripts/steering/submit_release_jobs.sh report/NAME_A                                               # jobs os-*
sbatch --dependency=afterok:ANALYSIS_JOB scripts/steering/pick_and_launch_half_b.sbatch report/NAME_A openai/gpt-oss-20b NAME_B TESTS --capture NAME --shards 2 --cutoff 2026-10-30T23:00
```

The capture and sweep stores go to FAST (`report/NAME/stores` is a link). `ANALYSIS_JOB` is the `os-analyze` line of `report/NAME_A/JOBS.txt`. `--models` and `--shards` take Hugging Face ids or short keys (`gpt`, `glm`); the release records the short key.

Paper use: pulsed steering gives figure `routing` and its tables, whole-reasoning steering gives `whole_states`, `whole_dose` and their tables. See [`src/steering`](../../src/steering/README.md).

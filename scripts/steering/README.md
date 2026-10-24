# Steering launchers

| Script | What it does |
| --- | --- |
| `pulse.sh SHARDS HOURS MODEL...` | GPU jobs of the pulsed design, one per shard of question families. |
| `pulse_analyze.sh OUT` | CPU jobs: the outcome tables, then the effects of the main and strong designs into `OUT`. |
| `whole.sh KIND NAME TESTS [OPTION...]` | Seals a complement, relink or outcome release (`python -m steering.release`; `--cutoff` is required for complement and outcome, a relink takes its parent's cutoff), then runs `launch.sh`. `AFTER="JOB ..."` passes running jobs on to `launch.sh`, e.g. a relink's parent combine. |
| `launch.sh NAME [JOB...]` | Submits the jobs of a sealed release: the unit tests (sealed in `CPU_VALIDATION.json`), then the manifests, the GPU shards, one extraction per model, the final analysis and the staged paper export. Refuses to submit anything when the jobs' time limits exceed the release's CPU or GPU ceilings. A failed launch cancels what it queued and keeps their IDs in `JOBS.cancelled-TIME.txt`. `DRY_RUN=1` prints the sbatch lines only. |
| `outcome_usage.sh` | CPU job: how often each expert is used in every cohort-B attempt (input of an outcome release). |
| `outcome_next.sh EFFECTS NAME TESTS [OPTION...]` | Applies the declared extension rule to an outcome analysis and, if it is met, seals and launches the next release. |
| `gpu.sh MODULE ARGS...` | The job script of a GPU stage: one Python module on 2 GPUs. |

The jobs of a release run from its frozen code copy in `mfa-moe/releases/<commit>`, not from the
working tree. Python decides what runs; these scripts decide where and in which order.

# Steering launchers

| Script | What it does |
| --- | --- |
| `pulse.sh SHARDS HOURS MODEL...` | GPU jobs of the pulsed design, one per shard of question families. |
| `pulse_analyze.sh OUT` | CPU jobs: the outcome tables, then the effects of the main and strong designs into `OUT`. |
| `whole.sh KIND NAME TESTS [OPTION...]` | Seals a complement, relink or outcome release (`python -m steering.release`; `--cutoff` is required for the first two kinds), then runs `launch.sh`. |
| `launch.sh NAME [JOB...]` | Submits the jobs of a sealed release: tests, then the manifests, the GPU shards, one extraction per model, the final analysis and the staged paper export. A failed launch cancels what it queued. `DRY_RUN=1` prints the sbatch lines only. |
| `outcome_usage.sh` | CPU job: how often each expert is used in every cohort-B attempt (input of an outcome release). |
| `outcome_next.sh EFFECTS NAME TESTS [OPTION...]` | Applies the declared extension rule to an outcome analysis and, if it is met, seals and launches the next release. |
| `gpu.sh MODULE ARGS...`, `tests.sh` | The job scripts the launchers submit: one Python module, and the unit tests. |

The jobs of a release run from its frozen code copy in `mfa-moe/releases/<commit>`, not from the
working tree. Python decides what runs; these scripts decide where and in which order.

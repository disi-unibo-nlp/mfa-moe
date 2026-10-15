# Study commands

From the repository:

```bash
scripts/study/run.sh plan
scripts/study/run.sh submit
scripts/study/run.sh status
scripts/study/run.sh resume
scripts/study/run.sh export
```

`plan` is read-only. `submit` and `resume` reconcile the same submission journal.
Add `--json` for machine-readable output. `job.sh` is the allocated worker;
the Python scheduler supplies its account, resources, dependencies and log paths.

Read `report/final-routing-20261015-r1/RECAP.md` for the latest observed results.
Python lives in `src/moe_exp/study/`. Older Python and `.sbatch` launchers stay
in place until their accepted jobs and frozen dependencies can be archived.

After an explicitly authorized code correction, `run.sh submit --replace-terminal` requires all prior allocations to be terminal and a source-matched release receipt in `study-reviews/`. It preserves the previous binding and journals. Ordinary `submit` and `resume` reuse the active release.

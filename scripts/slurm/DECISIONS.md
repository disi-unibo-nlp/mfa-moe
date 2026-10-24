# slurm: decisions

## bash-holds-slurm: Bash holds Slurm, Python holds the science
- status: active
- decided by: user (2026-10-24)
- applies to: Every launcher and job graph of the repository: model servers, generation, labeling, capture, routing analyses, steering releases, paper exports.
- not to: The computations themselves, sealed receipts and their formats, frozen pre-registered files.
- chosen: `scripts/<topic>/` holds only bash: one short launcher per experiment with positional arguments, `scripts/slurm/env.sh` sourced by every job, one generic `serve_model.sh HF_MODEL` whose `vllm serve` flags come from one table. Job graphs (dependencies, resources, walltimes, budget checks, cancellation) live in bash (`scripts/steering/launch.sh`). `src/<topic>/` holds only Python, which seals releases (`python -m steering.release`), reads its configuration from the release files, and computes; it never submits, accounts or monitors Slurm jobs.
- rejected: Python-built sbatch lines, journaled Python schedulers and monitors, accounting and GPU-hour projection in Python (the pre-2026-10-23 design, at tag archive/pre-cleanup-20261023); configuration passed to Python through WHOLE_* environment variables; long launchers with option parsing and if-else chains.
- evidence: User 2026-10-22/23: "only bash in scripts, and py in src [...] to launch a vllm serve, no need to do infinite if-else, args etc. Just a simple script that run vllm serve model-name gen-configs its fine. If the file can be 6 rows then we can do that." Implemented in repo-cleanup (steps 5-7); the steering launch graph was checked with DRY_RUN against r18, r20 and outcome-r1 releases.
- keywords: slurm, sbatch, launcher, bash, scripts, release, job graph, vllm serve, env.sh

# scripts/slurm

The parts every other launcher shares.

| File | What it does |
| --- | --- |
| `env.sh` | Setup that every job sources. It needs `SLURM_JOB_ID`, loads Python 3.11.7 and CUDA 12.6, puts the `vllm-cu129` environment on `PATH`, sets `PYTHONPATH=src`, switches Hugging Face to offline mode, and points every cache at a per-job folder `mfa-moe/r/<job id>` (short, because vLLM sockets fail when the path is longer than 107 characters). It derives the server port from the job id, so two jobs on one node do not collide, exports `$DSPY` (the judge client environment, found through `scripts/setup/judge_client_env_path.sh`) and `$JUDGE_MODEL` (the judge's Hugging Face id, from `python -m models.roster --judge`), and defines `start_server HF_MODEL`, which starts `scripts/serve/serve_model.sh HF_MODEL` in the background and waits until it answers (a cold start takes about 16 minutes, the wait gives up after 45). |
| `run_module_cpu.sbatch MODULE ARGS...` | Runs one Python module as a CPU job on the serial partition (4 CPUs, 30 GB, 4 hours by default). |
| `run_module_gpu.sbatch MODULE ARGS...` | Runs one Python module as a GPU job on the Booster partition (2 GPUs, 16 CPUs, 120 GB, 4 hours by default), with up to 4 BLAS threads next to the engine. Used by the steering stages. |
| `billing_hours_used.sh [PREFIX]` | Adds up the billing hours of the jobs whose name starts with a prefix (default `fr-`), against the 6,000-hour cap. Login node only. |
| `gpu_hours_left.sh` | Prints the billing hours and the GPU hours left on the account quota, read from CINECA's `saldo -b` (one GPU bills 8 hours per hour on the Booster), and the hours consumed this month. Login node only. |

```bash
sbatch -J my-analysis scripts/slurm/run_module_cpu.sbatch routing.expert_selection export CAMPAIGN_DIR /path/to/new/folder   # python -m routing.expert_selection ...
scripts/slurm/billing_hours_used.sh              # jobs named fr-*, the pulsed-steering jobs
scripts/slurm/billing_hours_used.sh whole-       # jobs named whole-*, the whole-reasoning releases
scripts/slurm/gpu_hours_left.sh                  # what is left of the whole quota, in billing hours and GPU hours
sbatch -t 12:00:00 -J fr-gpt-0 scripts/slurm/run_module_gpu.sbatch steering.pulse run --model gpt --shard 0/2
```

The two job scripts take the module name and its arguments; override resources with `sbatch` options before the script path (`-t`, `-c`, `--mem`, `--export=ALL,THREADS=4`). One BLAS thread is used unless `THREADS` is set, because fork pools run far slower with more.

Why the port depends on the job: Leonardo nodes hold four A100s, so two 2-GPU jobs can share a node. With one fixed port, a probe once passed the health check against its neighbour's server and then every request failed with an error that never named the port. Do not hard-code a port.

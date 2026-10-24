# scripts/slurm

The parts every other launcher shares.

| File | What it does |
| --- | --- |
| `env.sh` | Setup that every job sources. It needs `SLURM_JOB_ID`, loads Python 3.11.7 and CUDA 12.6, puts the `vllm-cu129` environment on `PATH`, sets `PYTHONPATH=src`, switches Hugging Face to offline mode, and points every cache at a per-job folder `mfa-moe/r/<job id>` (short, because vLLM sockets fail when the path is longer than 107 characters). It derives the server port from the job id, so two jobs on one node do not collide, and defines `start_server NAME`, which starts `scripts/serve/NAME.sh` in the background and waits until it answers (a cold start takes about 16 minutes, the wait gives up after 45). |
| `cpu.sh` | Runs one Python module as a CPU job on the serial partition (4 CPUs, 30 GB, 4 hours by default). |
| `ledger.sh` | Adds up the billing hours of the jobs whose name starts with a prefix, against the 6,000-hour cap. Login node only. |

```bash
sbatch -J my-analysis scripts/slurm/cpu.sh routing.selection export /path/to/new/folder   # python -m routing.selection ...
scripts/slurm/ledger.sh              # jobs named fr-*, the pulsed-steering jobs
scripts/slurm/ledger.sh whole-       # jobs named whole-*, the whole-reasoning releases
```

`cpu.sh` takes the module name and its arguments; override resources with `sbatch` options before the script path (`-t`, `-c`, `--mem`, `--export=ALL,THREADS=4`). One BLAS thread is used unless `THREADS` is set, because fork pools run far slower with more.

Why the port depends on the job: Leonardo nodes hold four A100s, so two 2-GPU jobs can share a node. With one fixed port, a probe once passed the health check against its neighbour's server and then every request failed with an error that never named the port. Do not hard-code a port.

# scripts

Bash only. Each folder matches a topic in `src/`, and each launcher runs one experiment or one step of it.

| Folder | What it does |
| --- | --- |
| [`slurm/`](slurm/README.md) | The shared job setup (`env.sh`), the generic CPU and GPU jobs (`run_module_cpu.sbatch`, `run_module_gpu.sbatch`) and the billing and quota tallies. |
| [`serve/`](serve/README.md) | The one script that starts the vLLM server of any model, the judge included. |
| [`generation/`](generation/README.md) | Generate benchmark traces with a model; check answers with the judge. |
| [`labeling/`](labeling/README.md) | Optimise the judge, pick the sentences to label, label them. |
| [`capture/`](capture/README.md) | Record the routers of each model on the saved traces, and accept a finished capture. |
| [`routing/`](routing/README.md) | The CPU analyses of native routing. |
| [`steering/`](steering/README.md) | The routing interventions and their analyses. |
| [`paper/`](paper/README.md) | Export tables and figure data, build the PDF, check the writing rules. |
| [`setup/`](setup/README.md) | Build the judge-client environment. |

Rules that hold for every script:

- Run from the repository root.
- Files named `*.sbatch` are Slurm jobs: `sbatch scripts/DIR/NAME.sbatch ARGS`. The `.sh` files with a `#SBATCH` header (`slurm/run_module_cpu.sbatch`, `slurm/run_module_gpu.sbatch`, `steering/outcome_next.sh`) are submitted the same way.
- The other `.sh` files run on the login node and only submit jobs, or print, or build the paper. They never start a model.
- A job refuses to start outside Slurm. Its setup is `slurm/env.sh`, so all jobs look the same: same modules, offline mode, own cache folders.
- An `OUT_DIR` argument must name a folder that does not exist yet.

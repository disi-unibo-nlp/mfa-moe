# scripts/setup

`build_judge_client_env.sh` builds the environment of the judge client, once. DSPy and GEPA are installed from `requirements/qwen-fullprecision-smoke.lock` (with hashes) into a virtual environment whose name ends in the first 16 hex digits of the lock file's hash. It stops if that folder already exists.

```bash
scripts/setup/build_judge_client_env.sh
```

The environment lives in `/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/`. It has DSPy and GEPA but not torch or scipy. The launchers that label sentences or optimise the judge add the `vllm-cu129` packages to `PYTHONPATH` for the rest (`scripts/slurm/env.sh` exports the folder as `$DSPY`).

`judge_client_env_path.sh` is the one place that says where the environment is: it computes the folder from the lock file's hash and sets `DSPY`. The build script and `scripts/slurm/env.sh` source it (from the repository root), and so does `scripts/labeling/optimize_judge.sh`. A new lock file therefore moves every user to the new folder at once.

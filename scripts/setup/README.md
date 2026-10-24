# scripts/setup

`dspy_env.sh` builds the environment of the judge client, once. DSPy and GEPA are installed from `requirements/qwen-fullprecision-smoke.lock` (with hashes) into a virtual environment whose name ends in the first 16 hex digits of the lock file's hash. It stops if that folder already exists.

```bash
scripts/setup/dspy_env.sh
```

The environment lives in `/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/`. It has DSPy and GEPA but not torch or scipy. The launchers that label sentences or optimise the judge add the `vllm-cu129` packages to `PYTHONPATH` for the rest (`scripts/slurm/env.sh` exports the folder as `$DSPY`).

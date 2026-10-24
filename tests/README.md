# tests

Unit tests, one folder per topic in `src/`. They run on tiny synthetic data or tiny real architectures. No test downloads weights, loads a real model, calls a live judge or submits a job. `test_imports.py` checks that every module and every test imports only names that exist, and that every module is reachable from a launcher, so dead code does not linger.

| Folder | Tests of |
| --- | --- |
| [`capture/`](capture/README.md) | `src/capture` |
| [`generation/`](generation/README.md) | `src/generation` and the server scripts |
| [`labeling/`](labeling/README.md) | `src/labeling` (including GEPA) |
| [`models/`](models/README.md) | `src/models` |
| [`paper/`](paper/README.md) | `src/paper` |
| [`routing/`](routing/README.md) | `src/routing`, `src/store.py`, `src/stats.py` |
| [`steering/`](steering/README.md) | `src/steering`, and [`steering/outcome/`](steering/outcome/README.md) for `src/steering/outcome` |

There are three ways to run them, depending on what a topic imports:

```bash
W=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs
export PYTHONPATH=src TMPDIR=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/me      # a folder you own: tests refuse /tmp

# 1. Light (correlation-client-3.11: pytest, numpy, pandas; no torch, no scipy). Fine on the login node.
$W/correlation-client-3.11/bin/python -m pytest -q -p no:cacheprovider tests/capture

# 2. vLLM environment (torch and scipy; unittest only, no pytest). Use a CPU job for anything large.
$W/vllm-cu129/bin/python -B -m unittest discover -s tests/steering -t tests/steering

# 3. pytest with torch and DSPy (about 2 minutes of CPU: run it as a CPU job). The light interpreter, plus the packages of the other two environments on the path.
SP=lib/python3.11/site-packages
DSPY=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-88189c22f214ed65
PYTHONPATH=src:$DSPY/$SP:$W/vllm-cu129/$SP $W/correlation-client-3.11/bin/python -m pytest -q -p no:cacheprovider tests/models
```

Each topic README says which of the three applies. A test that needs a package the environment lacks is skipped or fails at import, so check the environment before reading a failure as a bug. The login node allows 600 s of CPU, so run the large folders (`steering`, and the torch tests of option 3) in Slurm. `scripts/steering/tests.sh` does that for `tests/steering`, `tests/steering/outcome` and `tests/routing`; every steering release runs it first.

Tests that read sealed files in `report/` (three in `tests/steering`) fail in a fresh git worktree, because those files are not in git.

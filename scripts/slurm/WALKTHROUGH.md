# slurm: walkthroughs

## slurm-environment-trace: Slurm and serve, the environment of one job
- commit: 610ed2a6b1f5089b828de755c0eb951df4df7e08
- source: a dry run on the login node of scripts/slurm/env.sh and scripts/serve/gpt.sh and judge.sh with a made-up job id (the values are real output of those scripts; the id is demo:), a stub `vllm` that prints its arguments in place of the server, and requirements/qwen-fullprecision-smoke.lock; tests/generation/test_serve_scripts.py
- code: scripts/slurm/env.sh, scripts/slurm/cpu.sh, scripts/serve/args.sh, scripts/serve/gpt.sh, scripts/serve/judge.sh, scripts/setup/dspy_env.sh, tests/generation/test_serve_scripts.py

**Input:** A Slurm job that needs a model server, as every generation, labelling and steering job is. The
trace gives it the demo job id `SLURM_JOB_ID=61940001`, sources `scripts/slurm/env.sh` from the repository
root, and then runs `scripts/serve/gpt.sh` and `scripts/serve/judge.sh` with a stub in place of `vllm`.
Nothing was started and no GPU was used. The folder `mfa-moe/r/61940001` that the script made was gone
afterwards.

### 1. Guard and modules (`env.sh`, lines 1 to 5)

The script stops with "run this inside a Slurm job" when `SLURM_JOB_ID` is unset, then sets `set -euo
pipefail` and `umask 077`, and loads `python/3.11.7` and `cuda/12.6`. The module list afterwards holds 24
entries including `python/3.11.7` and `cuda/12.6`.

### 2. One short folder per job (`env.sh`: `RUN`, `trap`)

| Variable | Value |
|---|---|
| `RUN` | `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/61940001` (54 characters) |
| `TMPDIR`, `VLLM_RPC_BASE_PATH`, `XDG_CACHE_HOME`, `DSPY_CACHEDIR` | `RUN`, or a folder inside it |
| a vLLM engine socket in `RUN` | 80 characters, under the 107-byte limit of a Unix socket |

An `EXIT` trap stops the job's background processes and removes `RUN`. Every cache lives in `RUN`, so jobs do not
share state, and the path stays short because a deeper `TMPDIR` breaks tensor-parallel engines.

### 3. Software and offline mode (`env.sh`)

| Variable | Value |
|---|---|
| `PATH` starts with | `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin` |
| `PYTHONPATH` | `$PWD/src` (the checkout the job is submitted from) |
| `HF_HOME`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `mfa-moe/cache/hf`, 1, 1 (compute nodes are offline) |
| `TIKTOKEN_ENCODINGS_BASE`, `TIKTOKEN_RS_CACHE_DIR` | `mfa-moe/cache/tiktoken`, `RUN/tiktoken` (gpt-oss tokenizer file) |
| `OMP_NUM_THREADS` and the other BLAS counts | 1, unless the launcher passes `THREADS` |
| `DSPY` | the judge-client environment `.../qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-88189c22f214ed65` |

The last folder name is the first 16 hex digits of the sha256 of `requirements/qwen-fullprecision-smoke.lock`
(88189c22f214ed65, recomputed). `scripts/setup/dspy_env.sh` builds an environment with that name and stops at
`test ! -e $ENV` when it exists, so it was not run.

### 4. One port per job (`env.sh`: `PORT`, `URL`, `start_server`)

`PORT = 20000 + SLURM_JOB_ID % 20000` = 20000 + 61940001 % 20000 = 20001, so `URL` is
`http://127.0.0.1:20001/v1`. Two 2-GPU jobs can share a node of four GPUs, and a fixed port once let a probe pass the
health check against its neighbour's server. `start_server NAME` first calls `curl -sf -m 5
http://127.0.0.1:$PORT/health`; here it fails, so the port is free. If it answered, the job would exit with
"port is taken by another server". It then waits for `/health` for at most 45 minutes (`SECONDS < 2700`) and
stops when the server process dies.

### 5. The server command (`serve/args.sh`, `serve/gpt.sh`, `serve/judge.sh`)

`args.sh` sets `SERVE_ARGS`: host 127.0.0.1, the job's port, key `local-vllm-key`, tensor parallel 2, bfloat16
weights and KV cache, vLLM generation config, prefix caching, 8,192 batched tokens and 90 percent of GPU
memory. Each model script adds only its own flags. The stub printed these commands (the shared flags in the
middle are identical):

| Script | What it adds after the shared flags |
|---|---|
| `gpt.sh`: `vllm serve openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee …` | `--reasoning-parser openai_gptoss`, attention `TRITON_ATTN`, `--max-model-len 131072`, `--max-num-seqs 16` |
| `judge.sh`: `vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 …` | `--reasoning-parser qwen3`, attention `FLASH_ATTN`, `--language-model-only --enforce-eager`, `--max-model-len 49152`, `--max-num-seqs 64`, MTP with 3 speculative tokens |

### 6. How a launcher uses it (`cpu.sh`, `generate.sbatch`, `label.sbatch`)

`scripts/slurm/cpu.sh MODULE ARGS…` is a 4-CPU, 30 GB job on the serial partition that sources `env.sh` and runs
`python -m MODULE ARGS`. A GPU launcher such as `generate.sbatch` sources `env.sh`, calls `start_server $1` and
passes `--base-url $URL` to its module. `tests/generation/test_serve_scripts.py` (9 tests, 0.07 s) checks that
revision, parser, context, concurrency, attention, speculation, language-model-only and extra flags of every
profile in `model_profiles.py` equal what its serve script starts, and that the shared flags match the contract.

### Worth noticing

- The port and the folder both come from the job id, so nothing in the repository names a fixed port or a
  fixed scratch folder; the example job gets 20001 and 54 characters of path.
- The judge differs from a model server in three ways that matter to a job: a short 49,152-token context,
  64 concurrent sequences, and eager mode with MTP speculation.
- The scripts only decide settings. The recorded contract of the traces (`model_profiles.py`) is the other half,
  and the test above is what keeps the two together.
- A cold start takes about 16 minutes (`scripts/serve/README.md`), so the wait inside `start_server` is 45.

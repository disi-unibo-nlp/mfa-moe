# slurm: walkthroughs

## slurm-environment-trace: Slurm and serve, the environment of one job
- commit: e63f4d8dc28cc751351c09cdb329d323de567c6b
- source: a dry run on 2026-10-25, on the login node, of the code at this commit: scripts/slurm/env.sh sourced from the repository root with a made-up job id, scripts/serve/serve_model.sh run with a stub `vllm` that prints its arguments in place of the server, and requirements/qwen-fullprecision-smoke.lock (the values are real output of those scripts; the job id and the stub are demo:); tests/generation/test_serve_flags.py
- code: scripts/slurm/env.sh, scripts/slurm/run_module_cpu.sbatch, scripts/slurm/run_module_gpu.sbatch, scripts/serve/serve_model.sh, src/generation/serve_flags.py, scripts/setup/build_judge_client_env.sh, scripts/setup/judge_client_env_path.sh, tests/generation/test_serve_flags.py

**Input:** A Slurm job that needs a model server, as every generation, labelling and steering job is. The
trace gives it the demo job id `SLURM_JOB_ID=61940001`, sources `scripts/slurm/env.sh` from the repository
root, and then runs `scripts/serve/serve_model.sh` for `openai/gpt-oss-20b` and for the judge with a stub in place
of `vllm`. Nothing was started and no GPU was used. The folder `mfa-moe/r/61940001` that the script made was gone
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
| a vLLM engine socket in `RUN` (`RUN/<uuid4>`, the name vLLM 0.29 gives it) | 91 characters, under the 107-byte limit of a Unix socket (97 with the `ipc://` prefix, which does not count) |

An `EXIT` trap stops the job's background processes and removes `RUN`. Every cache lives in `RUN`, so jobs do not
share state, and the path stays short because a deeper `TMPDIR` breaks tensor-parallel engines.

### 3. Software, offline mode and the judge (`env.sh`, `judge_client_env_path.sh`)

| Variable | Value |
|---|---|
| `PATH` starts with | `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin` |
| `PYTHONPATH` | `$PWD/src` (the checkout the job is submitted from) |
| `HF_HOME`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `mfa-moe/cache/hf`, 1, 1 (compute nodes are offline) |
| `TIKTOKEN_ENCODINGS_BASE`, `TIKTOKEN_RS_CACHE_DIR` | `mfa-moe/cache/tiktoken`, `RUN/tiktoken` (gpt-oss tokenizer file) |
| `OMP_NUM_THREADS` and the other BLAS counts | 1, unless the launcher passes `THREADS` |
| `DSPY` | the judge-client environment `.../qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-88189c22f214ed65` |
| `JUDGE_MODEL` | `Qwen/Qwen3.8-27B`, printed by `python -m models.roster --judge` |

`env.sh` sources `scripts/setup/judge_client_env_path.sh`, which sets `DSPY_LOCK=requirements/qwen-fullprecision-smoke.lock`
and names the environment after the first 16 hex digits of the lock's sha256 (88189c22f214ed65, recomputed from
88189c22f214ed65ef506797cdccca50588f60f20073f75d7969ec2ae4efa256). `scripts/setup/build_judge_client_env.sh` sources
the same file, builds an environment with that name and stops at `test ! -e $DSPY` when it exists. It exists, so the
builder was not run.

### 4. One port per job (`env.sh`: `PORT`, `URL`, `start_server`)

`PORT = 20000 + SLURM_JOB_ID % 20000` = 20000 + 61940001 % 20000 = 20001, so `URL` is
`http://127.0.0.1:20001/v1`. Two 2-GPU jobs can share a node of four GPUs, and a fixed port once let a probe pass the
health check against its neighbour's server. `start_server HF_MODEL` first calls `curl -sf -m 5
http://127.0.0.1:$PORT/health`; here it fails (curl exit 7), so the port is free. If it answered, the job would exit
with "port is taken by another server". It then starts `bash scripts/serve/serve_model.sh HF_MODEL` in the
background with its output in `mfa-moe/logs/<job id>-<model name>-server.log`, and waits for `/health` for at most
45 minutes (`SECONDS < 2700`), stopping early when the server process dies and printing the last 50 log lines.
Two runs with the stub:

| Run | Result |
|---|---|
| demo: a stub `vllm` that also answers `/health` on its `--port` | `start_server openai/gpt-oss-20b` returned after 5 s (one poll); the log `61940001-gpt-oss-20b-server.log` holds the stub's arguments |
| the same call while that stub still answers | exits 1 with `port 20001 is taken by another server` |

### 5. The server command (`serve_model.sh`, `serve_flags.serve_argv`)

`serve_model.sh HF_MODEL` has no model logic: `python -m generation.serve_flags HF_MODEL` prints the arguments, one
per line, `mapfile` reads them into an array, and the script runs `exec vllm serve "${args[@]}"`. `serve_flags.py`
builds them from one table: the server settings of the model's card profile in `model_profiles.py` (or the judge's
own settings) plus `shared_flags(port)`. The shared flags are host 127.0.0.1, the job's port, key `local-vllm-key`,
tensor parallel 2, bfloat16 weights and KV cache, vLLM generation config, prefix caching, 8,192 batched tokens and
90 percent of GPU memory. The port comes from `$PORT`; without it `serve_flags` stops with "PORT is not set". The
stub printed these commands (the shared flags in the middle are identical):

| Model | What the stub received | What it adds after the shared flags |
|---|---|---|
| `openai/gpt-oss-20b` | 31 arguments (`serve` and 30 more), starting `serve openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee` | `--reasoning-parser openai_gptoss`, attention `TRITON_ATTN`, `--max-model-len 131072`, `--max-num-seqs 16` |
| judge `Qwen/Qwen3.8-27B` | 35 arguments (`serve` and 34 more), starting `serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `--reasoning-parser qwen3`, attention `FLASH_ATTN`, `--language-model-only --enforce-eager`, `--max-model-len 49152`, `--max-num-seqs 64`, MTP with 3 speculative tokens |

The short key `gpt` gives the same command as `openai/gpt-oss-20b`. LFM2.5 is refused with `ValueError:
LiquidAI/LFM2.5-24B-A2B has no vLLM server: its engine is part of src/steering`.

### 6. How a launcher uses it (`run_module_cpu.sbatch`, `run_module_gpu.sbatch`, `generate_answers.sbatch`, `label_sentences.sbatch`)

`scripts/slurm/run_module_cpu.sbatch MODULE ARGS…` is a 4-CPU, 30 GB, 4-hour job on the serial partition that
sources `env.sh` and runs `python -m MODULE ARGS`. `run_module_gpu.sbatch` does the same on 2 GPUs of the Booster
partition (16 CPUs, 120 GB) with up to 4 BLAS threads. A server launcher such as `generate_answers.sbatch` sources
`env.sh`, calls `start_server "$1"` and passes `--base-url $URL` to its module; `label_sentences.sbatch` calls
`start_server $JUDGE_MODEL` and runs the module with `$DSPY/bin/python`. `tests/generation/test_serve_flags.py` (20
tests, 0.21 s) pins the command line of each of the seven card models and the judge flag by flag, checks that
revision or local weights, parser, context, concurrency, attention, speculation and language-model-only of every
card profile equal what the command starts, that the short key, the LFM2.5 refusal and the missing `PORT` behave as
above, and that the command prints one argument per line.

### Worth noticing

- The port and the folder both come from the job id, so nothing in the repository names a fixed port or a
  fixed scratch folder; the example job gets 20001 and 54 characters of path.
- The judge differs from a model server in three ways that matter to a job: a short 49,152-token context,
  64 concurrent sequences, and eager mode with MTP speculation.
- The server flags and the recorded contract of the traces come from the same table (`model_profiles.py`), so a
  change to a setting changes both; the test above pins the resulting command lines.
- A cold start takes about 16 minutes (`scripts/slurm/README.md`; `scripts/serve/README.md` measures about 750 s
  for Qwen3-30B-A3B), so the wait inside `start_server` is 45.

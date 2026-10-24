#!/bin/bash
# Build the judge client env (dspy and gepa on top of the vLLM env) from the hashed lock file, once:
#   scripts/setup/dspy_env.sh     -> .../qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-<first 16 hex of the lock's sha256>
set -euo pipefail
module load python/3.11.7 cuda/12.6
LOCK=requirements/qwen-fullprecision-smoke.lock
ENV=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-$(sha256sum $LOCK | cut -c1-16)
test ! -e $ENV
UV=/leonardo/home/userexternal/lmolfett/.local/bin/uv
$UV venv --python /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python --system-site-packages $ENV
$UV pip install --python $ENV/bin/python --require-hashes --requirement $LOCK

#!/bin/bash
# Build the judge client env (dspy and gepa on top of the vLLM env) from the hashed lock file, once, from the repo root:
#   scripts/setup/build_judge_client_env.sh     -> .../qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-<first 16 hex of the lock's sha256>
set -euo pipefail
module load python/3.11.7 cuda/12.6
source scripts/setup/judge_client_env_path.sh  # sets DSPY_LOCK and DSPY
test ! -e $DSPY
UV=/leonardo/home/userexternal/lmolfett/.local/bin/uv
$UV venv --python /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python --system-site-packages $DSPY
$UV pip install --python $DSPY/bin/python --require-hashes --requirement $DSPY_LOCK

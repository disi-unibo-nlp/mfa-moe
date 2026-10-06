#!/bin/bash
set -euo pipefail
[[ "$(id -un)" == lmolfett && "$(hostname -f)" == *.leonardo.local ]] || exit 2
module load python/3.11.7
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/scripts/experimental_resume:$REPO/src:/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/code/s1-9a61e32f48c04c24"
exec /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B \
    "$REPO/scripts/experimental_resume/native_finalization_native_dispatch_v3.py" "${1:?prepare, generation, j1 or account}"

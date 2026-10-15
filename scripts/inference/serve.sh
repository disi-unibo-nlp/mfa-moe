#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm GPU allocation}"
: "${SLURM_JOB_GPUS:?A GPU allocation is required}"
model="${1:?usage: serve.sh MODEL [vllm options]}"
shift
module load python/3.11.7 cuda/12.6
export PATH="/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve "$model" "$@" --host 127.0.0.1

#!/bin/bash
# Execute one admitted cohort stage in its exact native allocation.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through the cohort submission journal}"
: "${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
test "$(id -un)" = lmolfett
case "$(hostname -f)" in *leonardo*) ;; *) exit 71 ;; esac
test "$#" = 3
cohort="$1"
stage="$2"
model="$3"
cd /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
module load python/3.11.7
module load cuda/12.6
env_root=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129
tmp="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/causal-rescue/${stage}-${SLURM_JOB_ID}"
mkdir -p "$tmp"
export TMPDIR="$tmp" XDG_CACHE_HOME="$tmp/xdg" TORCH_HOME="$tmp/torch"
export TRITON_CACHE_DIR="$tmp/triton" TORCHINDUCTOR_CACHE_DIR="$tmp/inductor"
export CUDA_CACHE_PATH="$tmp/cuda" VLLM_CACHE_ROOT="$tmp/vllm" MPLCONFIGDIR="$tmp/matplotlib"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export TIKTOKEN_ENCODINGS_BASE=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/tiktoken
export TIKTOKEN_RS_CACHE_DIR="$tmp/tiktoken"
export PATH="$env_root/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/scripts/experimental_resume${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
exec srun --kill-on-bad-exit=1 "$env_root/bin/python" -B -m moe_exp.study.rescue \
    worker --cohort "$cohort" --stage "$stage" --model "$model"

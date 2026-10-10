#!/bin/bash
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Run through the authorized Slurm launcher}"
test "$(id -un)" = lmolfett
case "$(hostname -f)" in *leonardo*) ;; *) exit 71 ;; esac
module load python/3.11.7
module load cuda/12.6
campaign_job_tmp="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/research-campaign-v3-${SLURM_JOB_ID}"
mkdir -p "$campaign_job_tmp"
export TMPDIR="$campaign_job_tmp"
export XDG_CACHE_HOME="$campaign_job_tmp/xdg" TORCH_HOME="$campaign_job_tmp/torch"
export TRITON_CACHE_DIR="$campaign_job_tmp/triton" TORCHINDUCTOR_CACHE_DIR="$campaign_job_tmp/inductor"
export CUDA_CACHE_PATH="$campaign_job_tmp/cuda" VLLM_CACHE_ROOT="$campaign_job_tmp/vllm"
export DSPY_CACHEDIR="$campaign_job_tmp/dspy" MPLCONFIGDIR="$campaign_job_tmp/matplotlib"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PATH="/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH"
export PYTHONPATH="/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/src:/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume"
trap '/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B scripts/research_campaign_v3/cleanup_tmp.py' EXIT
srun --kill-on-bad-exit=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B \
    scripts/research_campaign_v3/run.py worker --card "$1"

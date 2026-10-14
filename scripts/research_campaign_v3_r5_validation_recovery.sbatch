#!/bin/bash
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Only run in the authorized Slurm recovery allocation}"
test "$(id -un)" = lmolfett
case "$(hostname -f)" in *leonardo*) ;; *) exit 71 ;; esac
module load python/3.11.7
module load cuda/12.6
campaign_recovery_tmp="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/research-r5-validation-${SLURM_JOB_ID}"
mkdir -p "$campaign_recovery_tmp"
export TMPDIR="$campaign_recovery_tmp"
export XDG_CACHE_HOME="$campaign_recovery_tmp/xdg" DSPY_CACHEDIR="$campaign_recovery_tmp/dspy"
export MPLCONFIGDIR="$campaign_recovery_tmp/matplotlib"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/src:/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume
srun --kill-on-bad-exit=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B \
    scripts/research_campaign_v3_r5_validation_recovery.py worker --card "$1"

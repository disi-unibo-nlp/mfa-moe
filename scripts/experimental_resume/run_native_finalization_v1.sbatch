#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --job-name=native-finalization-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* &&
   "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == boost_usr_prod ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
STAGE=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
module load python/3.11.7 cuda/12.6
umask 077
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/nf-$SLURM_JOB_ID
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/vllm" "$RUNTIME/mpl"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg" MPLCONFIGDIR="$RUNTIME/mpl"
export TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" VLLM_CACHE_ROOT="$RUNTIME/vllm"
export VLLM_RPC_BASE_PATH="$RUNTIME"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_BATCH_INVARIANT=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
export PYTHONPATH="$REPO/scripts/experimental_resume:$REPO/src:$STAGE/code/s1-9a61e32f48c04c24"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$END_TEXT" && "$END_TEXT" != N/A && "$END_TEXT" != *$'\n'* ]] || exit 2
END_EPOCH="$(date -d "$END_TEXT" +%s)"
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/run_native_finalization_v1.py" --deadline-epoch "$((END_EPOCH - 600))"

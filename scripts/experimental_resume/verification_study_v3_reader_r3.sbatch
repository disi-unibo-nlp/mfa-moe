#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" == lrdn* ]] || exit 2
[[ "${SLURM_CLUSTER_NAME:-}" == leonardo && "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == boost_usr_prod ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:-}" == 16 && $# == 1 ]] || exit 2
JOB_CARD=$1
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
module load python/3.11.7
module load cuda/12.6
umask 077
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/t3/r$SLURM_JOB_ID
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch"
export TMPDIR="$RUNTIME/tmp" TMP="$RUNTIME/tmp" TEMP="$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch"
export VLLM_RPC_BASE_PATH="$RUNTIME"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_BATCH_INVARIANT=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
export SLURM_EXPORT_ENV=ALL
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$END_TEXT" && "$END_TEXT" != N/A && "$END_TEXT" != *$'\n'* ]] || exit 2
END_EPOCH="$(date -d "$END_TEXT" +%s)"
[[ "$END_EPOCH" -gt "$(date +%s)" ]] || exit 2
srun --ntasks=1 --cpus-per-task=16 --cpu-bind=none --export=ALL "$PY" -B -u "$REPO/scripts/experimental_resume/verification_study_v3_transport_r3.py" execute \
  --kind reader --job-card "$JOB_CARD" --deadline-epoch "$((END_EPOCH-300))"

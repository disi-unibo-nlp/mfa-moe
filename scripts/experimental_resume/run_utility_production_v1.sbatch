#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --job-name=utility-production-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%A_%a.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%A_%a.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* &&
   "${SLURM_JOB_PARTITION:-}" == boost_usr_prod ]] || exit 2
: "${UTILITY_PRODUCTION_CONFIG:?sealed production configuration required}"
: "${UTILITY_PRODUCTION_MANIFEST:?sealed production manifest required}"
: "${UTILITY_PRODUCTION_OUT:?digest-bound production output required}"
: "${SLURM_ARRAY_TASK_ID:?exact sealed shard index required}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
module load python/3.11.7
module load cuda/12.6
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-utility-production-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg"
export TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_BATCH_INVARIANT=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
export PYTHONPATH="$REPO/scripts/experimental_resume:$REPO/src:$ROOT/code/s1-9a61e32f48c04c24"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$END_TEXT" && "$END_TEXT" != N/A && "$END_TEXT" != *$'\n'* ]] || exit 2
END_EPOCH="$(date -d "$END_TEXT" +%s)"
[[ "$END_EPOCH" -gt "$(date +%s)" ]] || exit 2
RECOVERY=()
if [[ -n "${UTILITY_PRODUCTION_RECOVERY:-}" ]]; then
  RECOVERY=(--recovery-manifest "$UTILITY_PRODUCTION_RECOVERY")
fi
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/run_utility_production_v1.py" \
  --config "$UTILITY_PRODUCTION_CONFIG" --manifest "$UTILITY_PRODUCTION_MANIFEST" \
  --out "$UTILITY_PRODUCTION_OUT" --shard-index "$SLURM_ARRAY_TASK_ID" \
  --deadline-epoch "$((END_EPOCH - 600))" "${RECOVERY[@]}"

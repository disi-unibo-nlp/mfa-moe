#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=mechanism-cap-recovery-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%A_%a.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%A_%a.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
: "${CAP_RECOVERY_MODE:?qualification or generation required}"
[[ "$CAP_RECOVERY_MODE" == qualification || "$CAP_RECOVERY_MODE" == generation ]] || exit 2
SHARD=0
EXACT_JOB="$SLURM_JOB_ID"
if [[ "$CAP_RECOVERY_MODE" == generation ]]; then
  SHARD="${SLURM_ARRAY_TASK_ID:?priced generation shard required}"
  EXACT_JOB="${SLURM_ARRAY_JOB_ID:?}_$SHARD"
fi
END_TEXT="$(squeue -h -j "$EXACT_JOB" -o '%e')"
[[ -n "$END_TEXT" && "$END_TEXT" != N/A && "$END_TEXT" != *$'\n'* ]] || exit 2
END_EPOCH="$(date -d "$END_TEXT" +%s)"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
DRIVER="$REPO/scripts/experimental_resume/mechanism_semantic_cap_recovery_v1.py"
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/mechanism-cap-recovery-runtime-$SLURM_JOB_ID
module load python/3.11.7
umask 077
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
EXTRA=()
if [[ "${CAP_RECOVERY_RESUME_UNCOMMITTED:-0}" == 1 ]]; then EXTRA+=(--resume-uncommitted); fi
srun --ntasks=1 "$PY" -B -u "$DRIVER" --cpu-preflight
srun --ntasks=1 "$PY" -B -u "$DRIVER" --mode "$CAP_RECOVERY_MODE" --shard-index "$SHARD" \
  --deadline-epoch "$((END_EPOCH - 180))" "${EXTRA[@]}"

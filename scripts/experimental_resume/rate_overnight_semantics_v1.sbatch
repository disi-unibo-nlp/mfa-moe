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
#SBATCH --job-name=overnight-blind-reader-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%A_%a.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%A_%a.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
: "${OVERNIGHT_BLIND_FRAME:?sealed blind frame required}"
: "${OVERNIGHT_READER_PRICE:?sealed complete reader price required}"
: "${OVERNIGHT_READER_OUT:?reader output root required}"
: "${SLURM_ARRAY_TASK_ID:?priced shard array index required}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
TASK_SPEC="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SLURM_END_TEXT="$(squeue -h -j "$TASK_SPEC" -o '%e')"
[[ -n "$SLURM_END_TEXT" && "$SLURM_END_TEXT" != N/A && "$SLURM_END_TEXT" != *$'\n'* ]] || exit 2
SLURM_END_EPOCH="$(date -d "$SLURM_END_TEXT" +%s)"
[[ "$SLURM_END_EPOCH" -gt "$(date +%s)" ]] || exit 2
module load python/3.11.7
umask 077
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/overnight-semantic-runtime-$TASK_SPEC
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
EXTRA=()
if [[ "${OVERNIGHT_RESUME_UNCOMMITTED:-0}" == 1 ]]; then EXTRA+=(--resume-uncommitted); fi
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/rate_overnight_semantics_v1.py" \
  --frame "$OVERNIGHT_BLIND_FRAME" --price "$OVERNIGHT_READER_PRICE" --out "$OVERNIGHT_READER_OUT" \
  --shard-index "$SLURM_ARRAY_TASK_ID" --deadline-epoch "$((SLURM_END_EPOCH - 180))" "${EXTRA[@]}"
SHARD_DIR="$(printf 'shard-%03d' "$SLURM_ARRAY_TASK_ID")"
[[ -f "$OVERNIGHT_READER_OUT/$SHARD_DIR/SUMMARY.json" ]] || exit 2

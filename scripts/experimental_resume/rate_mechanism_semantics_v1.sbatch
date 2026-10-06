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
#SBATCH --job-name=mechanism-1024-reader-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%A_%a.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%A_%a.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "$(hostname -s)" != login* ]] || exit 2
: "${MECHANISM_BLIND_FRAME:?sealed BLIND_FRAME.json path required}"
: "${MECHANISM_READER_PRICE:?sealed reader price path required}"
: "${MECHANISM_READER_OUT:?reader output directory required}"
: "${SLURM_ARRAY_TASK_ID:?priced shard array index required}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
DRIVER="$REPO/scripts/experimental_resume/rate_mechanism_semantics_v1.py"
SLURM_END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$SLURM_END_TEXT" && "$SLURM_END_TEXT" != N/A ]] || exit 2
SLURM_END_EPOCH="$(date -d "$SLURM_END_TEXT" +%s)"
[[ "$SLURM_END_EPOCH" -gt "$(date +%s)" ]] || exit 2
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/mechanism-semantic-runtime-$SLURM_JOB_ID
module load python/3.11.7
umask 077
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
EXTRA=()
if [[ "${MECHANISM_RESUME_UNCOMMITTED:-0}" == 1 ]]; then
  EXTRA+=(--resume-uncommitted)
fi
srun --ntasks=1 "$PY" -B -u "$DRIVER" \
  --frame "$MECHANISM_BLIND_FRAME" \
  --price "$MECHANISM_READER_PRICE" \
  --out "$MECHANISM_READER_OUT" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --deadline-epoch "$((SLURM_END_EPOCH - 180))" \
  "${EXTRA[@]}"

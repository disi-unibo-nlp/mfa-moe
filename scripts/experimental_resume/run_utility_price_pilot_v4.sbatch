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
#SBATCH --job-name=utility-16k-price-pilot-v4
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
: "${UTILITY_PILOT_PROPOSAL:?sealed bounded runtime proposal required}"
: "${UTILITY_PILOT_MANIFEST:?exact first-two-family 8-cell manifest required}"
: "${UTILITY_PILOT_OUT:?bound pilot output directory required}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
module load python/3.11.7
module load cuda/12.6
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-utility-price-pilot-$SLURM_JOB_ID"
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
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/attach_utility_price_pilot_v4.py" \
  --proposal "$UTILITY_PILOT_PROPOSAL" --phase validate-pilot
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/run_utility_price_pilot_v2.py" run \
  --manifest "$UTILITY_PILOT_MANIFEST" --out "$UTILITY_PILOT_OUT" --deadline-epoch "$((END_EPOCH - 600))"
[[ -f "$UTILITY_PILOT_OUT/SUMMARY-$SLURM_JOB_ID.json" ]] || exit 2

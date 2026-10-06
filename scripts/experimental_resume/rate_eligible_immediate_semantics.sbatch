#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --job-name=st-eligible-rate
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
KIND="${KIND:-${1:-v3}}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/rate_eligible_immediate_semantics_v1.py"
case "$KIND" in
  v3)
    FRAME="$ROOT/runs/routing-control-v1/blind-eligible-micro-v3-320ecca116c7ab63/BLIND_FRAME.json"
    PRICE="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_MICRO_SERIAL_BLIND_RATING_PRICE_v1.json"
    OUT_STEM="ratings-eligible-micro-v3"
    ;;
  v4)
    FRAME="$ROOT/runs/routing-control-v1/blind-eligible-deactivation-v4-85b7f90ae7295d85/BLIND_FRAME.json"
    PRICE="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_BLIND_RATING_PRICE_v1.json"
    OUT_STEM="ratings-eligible-deactivation-v4"
    ;;
  *) echo "KIND must be v3 or v4" >&2; exit 2 ;;
esac
[[ -f "$FRAME" && -f "$PRICE" && -f "$DRIVER" ]] || { echo "missing frame, price or rating driver" >&2; exit 2; }
SLURM_END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$SLURM_END_TEXT" && "$SLURM_END_TEXT" != N/A ]] || exit 2
SLURM_END_EPOCH="$(date -d "$SLURM_END_TEXT" +%s)"
export STEER_DEADLINE_EPOCH="$((SLURM_END_EPOCH - 180))"
[[ "$STEER_DEADLINE_EPOCH" -gt "$(date +%s)" ]] || exit 2
module load python/3.11.7
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-eligible-rate-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin:$PATH
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume:$ROOT/code/s1-9a61e32f48c04c24"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
FRAME_SHA="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"][:16])' "$FRAME")"
OUT="$ROOT/runs/routing-control-v1/$OUT_STEM-$FRAME_SHA"
srun --ntasks=1 "$PY" -B -u "$DRIVER" --frame "$FRAME" --price "$PRICE" --out "$OUT" --cpu-preflight
srun --ntasks=1 "$PY" -B -u "$DRIVER" --frame "$FRAME" --price "$PRICE" --out "$OUT" --deadline-epoch "$STEER_DEADLINE_EPOCH"
[[ -f "$OUT/SUMMARY.json" ]] || { echo "rating summary missing: $OUT" >&2; exit 2; }

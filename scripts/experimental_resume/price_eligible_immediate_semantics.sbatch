#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --job-name=st-eligible-price
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
KIND="${KIND:-${1:-v3}}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/price_eligible_immediate_semantics_v1.py"
case "$KIND" in
  v3)
    FRAME="$ROOT/runs/routing-control-v1/blind-eligible-micro-v3-320ecca116c7ab63/BLIND_FRAME.json"
    OUT="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_MICRO_SERIAL_BLIND_RATING_PRICE_v1.json"
    ;;
  v4)
    FRAME="$ROOT/runs/routing-control-v1/blind-eligible-deactivation-v4-85b7f90ae7295d85/BLIND_FRAME.json"
    OUT="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_BLIND_RATING_PRICE_v1.json"
    ;;
  *) echo "KIND must be v3 or v4" >&2; exit 2 ;;
esac
[[ -f "$FRAME" && -f "$DRIVER" ]] || { echo "missing frame or price driver" >&2; exit 2; }
umask 077
export TMPDIR="$ROOT/runs/routing-control-v1/runtime-eligible-price-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume:$ROOT/code/s1-9a61e32f48c04c24"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --frame "$FRAME" --out "$OUT"
STATUS="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUT")"
[[ "$STATUS" == PASS_COMPLETE_STAGE ]] || { echo "price status: $STATUS" >&2; exit 2; }

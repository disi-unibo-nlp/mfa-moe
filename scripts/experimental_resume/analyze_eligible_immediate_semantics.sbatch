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
#SBATCH --job-name=st-eligible-analyze
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
KIND="${KIND:-${1:-v3}}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/analyze_eligible_immediate_semantics_v1.py"
case "$KIND" in
  v3)
    MANIFEST="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_MICRO_SERIAL_MANIFEST_v3.json"
    BLIND="$ROOT/runs/routing-control-v1/blind-eligible-micro-v3-320ecca116c7ab63"
    RATINGS_STEM="ratings-eligible-micro-v3"
    OUT="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_MICRO_SERIAL_SEMANTIC_ANALYSIS_v1.json"
    ;;
  v4)
    MANIFEST="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v4.json"
    BLIND="$ROOT/runs/routing-control-v1/blind-eligible-deactivation-v4-85b7f90ae7295d85"
    RATINGS_STEM="ratings-eligible-deactivation-v4"
    OUT="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_SEMANTIC_ANALYSIS_v2.json"
    ;;
  *) echo "KIND must be v3 or v4" >&2; exit 2 ;;
esac
[[ -f "$MANIFEST" && -f "$BLIND/ARM_MAP.json" && -f "$BLIND/BLIND_FRAME.json" && -f "$DRIVER" ]] || {
  echo "missing manifest, blind frame or analysis driver" >&2; exit 2; }
FRAME="$BLIND/BLIND_FRAME.json"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
FRAME_SHA="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"][:16])' "$FRAME")"
RATINGS="$ROOT/runs/routing-control-v1/$RATINGS_STEM-$FRAME_SHA"
[[ -f "$RATINGS/SUMMARY.json" ]] || { echo "ratings summary is missing" >&2; exit 2; }
umask 077
export TMPDIR="$ROOT/runs/routing-control-v1/runtime-eligible-analyze-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume:$ROOT/code/s1-9a61e32f48c04c24"
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --arm-map "$BLIND/ARM_MAP.json" \
  --frame "$FRAME" --ratings "$RATINGS" --out "$OUT"

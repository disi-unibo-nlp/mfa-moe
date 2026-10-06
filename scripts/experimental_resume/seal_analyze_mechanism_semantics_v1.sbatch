#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=mechanism-semantic-itt-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett &&
   "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1
SUFFIX=40ea298e178d0863
SEMANTIC="$ROOT/mechanism-1024-semantics-v1-$SUFFIX"
RATINGS="$ROOT/mechanism-1024-readers-v1-$SUFFIX"
MANIFEST="$REPO/report/experimental-resume-v1/MECHANISM_VALIDATION_MANIFEST_v1.json"
OUT="$REPO/report/experimental-resume-v1/MECHANISM_1024_SEMANTIC_ITT_v1.json"
module load python/3.11.7
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
umask 077
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/rate_mechanism_semantics_v1.py" \
  --frame "$SEMANTIC/BLIND_FRAME.json" --price "$SEMANTIC/READER_PRICE.json" \
  --out "$RATINGS" --seal-stage
[[ -f "$RATINGS/STAGE_SUMMARY.json" ]] || exit 2
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/analyze_mechanism_semantics_v1.py" \
  --manifest "$MANIFEST" --arm-map "$SEMANTIC/ARM_MAP.json" \
  --frame "$SEMANTIC/BLIND_FRAME.json" --price "$SEMANTIC/READER_PRICE.json" \
  --ratings-out "$RATINGS" --out "$OUT"
"$PY" -B -c 'import sys; from run_boundary_micro_screen import sealed; d=sealed(sys.argv[1]); assert d["schema"]=="mechanism-1024-semantic-itt-v1" and d["assigned"]==760 and d["accepted_families"]==78; print("sealed ITT:",d["sha256"],d["assigned"],"assignments")' "$OUT"

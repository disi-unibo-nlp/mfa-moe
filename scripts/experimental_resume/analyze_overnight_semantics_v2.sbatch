#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --job-name=overnight-itt-analysis-v2
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
: "${OVERNIGHT_MANIFEST:?sealed generation manifest required}"
: "${OVERNIGHT_MEASUREMENT_OUT:?blind frame and price directory required}"
: "${OVERNIGHT_READER_OUT:?complete reader output root required}"
: "${OVERNIGHT_ANALYSIS_OUT:?analysis output directory required}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
module load python/3.11.7
umask 077
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/rate_overnight_semantics_v2.py" \
  --frame "$OVERNIGHT_MEASUREMENT_OUT/BLIND_FRAME.json" --price "$OVERNIGHT_MEASUREMENT_OUT/READER_PRICE.json" \
  --out "$OVERNIGHT_READER_OUT" --seal-stage
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/analyze_overnight_semantics_v2.py" \
  --manifest "$OVERNIGHT_MANIFEST" --arm-map "$OVERNIGHT_MEASUREMENT_OUT/ARM_MAP.json" \
  --frame "$OVERNIGHT_MEASUREMENT_OUT/BLIND_FRAME.json" --price "$OVERNIGHT_MEASUREMENT_OUT/READER_PRICE.json" \
  --ratings-out "$OVERNIGHT_READER_OUT" --out "$OVERNIGHT_ANALYSIS_OUT"
[[ -f "$OVERNIGHT_ANALYSIS_OUT/ANALYSIS.json" && -f "$OVERNIGHT_ANALYSIS_OUT/ASSIGNED_RESULTS.json" ]] || exit 2

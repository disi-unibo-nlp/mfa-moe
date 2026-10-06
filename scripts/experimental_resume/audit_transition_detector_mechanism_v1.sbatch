#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=st-mechanism-detect
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/audit_transition_detector_mechanism_v1.py"
UNITS="$ROOT/runs/routing-control-v1/dense-mechanism/MECHANISM_UNITS_v1.json"
OUT="$REPO/report/experimental-resume-v1/MECHANISM_DETECTOR_AUDIT_v1.json"
[[ -f "$DRIVER" && -f "$UNITS" ]] || { echo "missing driver or mechanism units" >&2; exit 2; }
umask 077
export TMPDIR="$ROOT/runs/routing-control-v1/runtime-mechanism-detect-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --units "$UNITS" --out "$OUT"

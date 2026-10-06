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
#SBATCH --job-name=st-prepare-units
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
POOL="${POOL:-${1:-mechanism}}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/prepare_dense_pool_units.py"
case "$POOL" in
  mechanism) OUT="$ROOT/runs/routing-control-v1/dense-mechanism/MECHANISM_UNITS_v1.json" ;;
  utility)   OUT="$ROOT/runs/routing-control-v1/dense-utility/UTILITY_UNITS_v1.json" ;;
  discovery) OUT="$ROOT/runs/routing-control-v1/dense-discovery/DISCOVERY_UNITS_v1_rebuilt.json" ;;
  *) echo "POOL must be discovery, mechanism or utility" >&2; exit 2 ;;
esac
[[ -f "$DRIVER" ]] || { echo "missing pool-unit driver" >&2; exit 2; }
umask 077
export TMPDIR="$ROOT/runs/routing-control-v1/runtime-prepare-units-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --pool "$POOL" --out "$OUT"

#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --job-name=st-extension-generation-price-v2
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett &&
   "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DOC="$REPO/report/experimental-resume-v1"
RUNNER="$REPO/scripts/experimental_resume/overnight_routing_runner_v2.py"
DESIGN="$DOC/OVERNIGHT_FRESH_COMPARISON_DESIGN_v2.json"
MANIFEST="$DOC/OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json"
PRICE="$DOC/OVERNIGHT_FRESH_COMPARISON_PRICE_v2.json"
[[ -f "$DOC/MECHANISM_EXTENSION_OVERNIGHT_ENROLLMENT_v1.json" && -f "$DESIGN" ]] || exit 2
module load python/3.11.7
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-extension-generation-price-v2-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume:$ROOT/code/s1-9a61e32f48c04c24"
export VLLM_BATCH_INVARIANT=0
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$RUNNER" --prepare --design "$DESIGN" --manifest "$MANIFEST" --price "$PRICE" --max-wall-seconds 8100
srun --ntasks=1 "$PY" -B -u "$RUNNER" --manifest "$MANIFEST" --cpu-preflight

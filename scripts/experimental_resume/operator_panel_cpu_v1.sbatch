#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=operator-panel-cpu-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
: "${OPERATOR_PANEL_PHASE:?}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
module load python/3.11.7
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-operator-panel-cpu-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$REPO/scripts/experimental_resume:$REPO/src:$ROOT/code/s1-9a61e32f48c04c24"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/operator_panel_orchestration_v1.py" cpu --phase "$OPERATOR_PANEL_PHASE"

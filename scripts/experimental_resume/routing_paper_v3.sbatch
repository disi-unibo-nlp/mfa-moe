#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --job-name=st-routing-paper-v3
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
[[ "$#" == 2 ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runtime/paper-${SLURM_JOB_ID}
mkdir -p "$RUNTIME/tmp" "$RUNTIME/matplotlib"
module load python/3.11.7
export TMPDIR="$RUNTIME/tmp" MPLCONFIGDIR="$RUNTIME/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/routing_paper_v3.py" "$1" "$2"

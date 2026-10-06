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
#SBATCH --job-name=mechanism-cap-recovery-analysis-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python
module load python/3.11.7
umask 077
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/mechanism_semantic_cap_recovery_v1.py" \
  --analyze "$REPO/report/experimental-resume-v1/MECHANISM_SEMANTIC_CAP_RECOVERY_SENSITIVITY_v1.json"

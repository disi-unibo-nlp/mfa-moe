#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --job-name=r3d-merged-audit-v1
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/r3d-merged-audit-v1-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/r3d-merged-audit-v1-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
module load python/3.11.7
SCRIPT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume/audit_r3d_merged_inventory_v1.py
EXPECTED=dc3405dfa74fb9d971f0c30dd43fa760f206681281236ef22565ed2b35f7f06f
ACTUAL=$(sha256sum "$SCRIPT")
[[ "${ACTUAL%% *}" == "$EXPECTED" ]] || { echo 'R3-D merged audit source changed' >&2; exit 3; }
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
srun --ntasks=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -u "$SCRIPT"

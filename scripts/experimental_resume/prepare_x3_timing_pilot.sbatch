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
#SBATCH --job-name=x3-timing-pilot-prep-v1
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/x3-timing-pilot-prep-v1-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/x3-timing-pilot-prep-v1-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
module load python/3.11.7
SCRIPT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume/prepare_x3_timing_pilot.py
EXPECTED=1649d21d36841d7d888a228525584a08cf750ede927706c0b3baf9585fc5881b
ACTUAL=$(sha256sum "$SCRIPT")
[[ "${ACTUAL%% *}" == "$EXPECTED" ]] || { echo 'X3 timing pilot source changed' >&2; exit 3; }
export STEER_CODE=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/code/s1-9a61e32f48c04c24
export PYTHONPATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume:$STEER_CODE:$STEER_CODE/moe_exp_src
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export TMPDIR=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/x3-timing-pilot-prep-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
srun --ntasks=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -u "$SCRIPT"

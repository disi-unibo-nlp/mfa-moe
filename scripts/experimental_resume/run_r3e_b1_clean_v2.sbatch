#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --job-name=r3e-b1-clean-v2
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/r3e-b1-clean-v2-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/r3e-b1-clean-v2-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
module load python/3.11.7
SCRIPT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume/run_r3e_b1_clean_v2.py
EXPECTED=7e7c40902e751eb52bf8873d86997bd038e621b72b61eef448942fafbaa0e066
ACTUAL=$(sha256sum "$SCRIPT")
[[ "${ACTUAL%% *}" == "$EXPECTED" ]] || { echo 'clean B1 driver changed' >&2; exit 3; }
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
srun --ntasks=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -u "$SCRIPT"

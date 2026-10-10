#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_CLUSTER_NAME:-} == leonardo && ${SLURM_JOB_ACCOUNT:-} == iscrc_miosr && ${SLURM_JOB_PARTITION:-} == lrd_all_serial && $# == 1 ]] || exit 2
[[ $(id -un) == lmolfett ]] || exit 2
umask 077
module load python/3.11.7
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/em1-handoff-${SLURM_JOB_ID}
mkdir -p "$RUNTIME"
export TMPDIR="$RUNTIME" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --ntasks=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -u "$REPO/scripts/experimental_resume/expert_memory_study_v1_handoff_r1.py" run --card "$1"

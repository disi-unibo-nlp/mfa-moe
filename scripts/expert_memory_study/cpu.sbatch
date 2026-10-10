#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_CLUSTER_NAME:-} == leonardo && ${SLURM_JOB_ACCOUNT:-} == iscrc_miosr && ${SLURM_JOB_PARTITION:-} == lrd_all_serial ]] || exit 2
[[ $(id -un) == lmolfett && $# == 3 ]] || exit 2
umask 077
module load python/3.11.7
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/em1-${SLURM_JOB_ID}
mkdir -p "$RUNTIME"
export TMPDIR="$RUNTIME" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
END_TEXT=$(squeue -h -j "$SLURM_JOB_ID" -o '%e')
[[ -n $END_TEXT && $END_TEXT != N/A && $END_TEXT != *$'\n'* ]] || exit 2
END_EPOCH=$(date -d "$END_TEXT" +%s)
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/expert_memory_study/run.py" run-card --campaign "$1" --card "$2" --authorization "$3" --deadline-epoch "$((END_EPOCH-60))"

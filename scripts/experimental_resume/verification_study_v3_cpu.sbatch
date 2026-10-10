#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "${SLURM_CLUSTER_NAME:-}" == leonardo && "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == lrd_all_serial ]] || exit 2
[[ $# == 5 ]] || exit 2
CAMPAIGN=$1
STAGE=$2
MODEL_KEY=$3
RESULT_ROOT=$4
JOB_CARD=$5
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
umask 077
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/cv3-cpu-$SLURM_JOB_ID
mkdir -p "$RUNTIME"
export TMPDIR="$RUNTIME" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
if [[ "$STAGE" == evidence ]]; then
  srun --ntasks=1 "$PY" -B "$REPO/scripts/experimental_resume/verification_study_v3.py" evidence \
    --campaign "$CAMPAIGN" --root "$RESULT_ROOT"
else
  srun --ntasks=1 "$PY" -B "$REPO/scripts/experimental_resume/verification_study_v3_task.py" "$JOB_CARD"
fi

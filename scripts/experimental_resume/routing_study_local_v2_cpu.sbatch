#!/bin/bash
# Planned CPU analysis wrapper; submit only with stage resource authorization.
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=routing-local-context-v2
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/routing-study-v2/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/routing-study-v2/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
[[ "${SLURM_JOB_PARTITION:-}" == lrd_all_viz && $# -gt 0 ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
module load python/3.11.7
umask 077
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/rcv2-$SLURM_JOB_ID
mkdir -p "$RUNTIME/tmp"
export TMPDIR="$RUNTIME/tmp"
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
case "$1" in
  context|anticipation|dense-ledger) srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/routing_study_local_v2.py" "$@" ;;
  *) exit 2 ;;
esac

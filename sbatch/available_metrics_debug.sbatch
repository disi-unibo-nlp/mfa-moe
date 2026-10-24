#!/bin/bash
#SBATCH --job-name=mfa-all-metrics
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lorenzo.molfetta@unibo.it
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/available-metrics/v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/available-metrics/v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr ]] || exit 2
case "${1:-}" in labels|routing) ;; *) exit 2;; esac
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
STUDY="$P/routing-study"
RUN="$STUDY/results/available-metrics/v1/runtime/${SLURM_JOB_ID}"
umask 077
mkdir -p "$RUN/tmp" "$RUN/cache"
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache" MPLCONFIGDIR="$RUN/cache/mpl"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$P/repo/src:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
cd "$STUDY"
srun "$P/envs/vllm-cu129/bin/python" -u scripts/analyze_available_metrics.py "$1" --workers 8

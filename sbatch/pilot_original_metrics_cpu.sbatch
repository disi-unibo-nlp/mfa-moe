#!/bin/bash
#SBATCH --job-name=pilot-original-metrics
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/pilot-metrics/logs/%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/pilot-metrics/logs/%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_PARTITION:-}" == lrd_all_serial && "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr ]] || exit 2
[[ "$(id -un)" == lmolfett ]] || exit 2
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
STUDY="$P/routing-study"
OUT="$STUDY/results/pilot-metrics/original-v1"
umask 077
mkdir -p "$OUT/runtime/tmp" "$OUT/runtime/matplotlib"
export TMPDIR="$OUT/runtime/tmp" MPLCONFIGDIR="$OUT/runtime/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$P/repo/src:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
srun "$P/envs/vllm-cu129/bin/python" -B "$STUDY/scripts/analyze_pilot_original_metrics.py" \
 --root "$P/repo/results/correlation_pipeline/routing-dynamics-v1/pilot" --base "$STUDY/results/pilot-metrics/correlations-v1" --output "$OUT"

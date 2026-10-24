#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lorenzo.molfetta@unibo.it
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.err
#SBATCH --job-name=mfa-gemma-source-recovery
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=00:30:00
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr ]] || exit 2
case "${1:-}" in gpt|glm|qwen330b|qwen36|nemotron|gemma) ;; *) exit 2;; esac
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
ROOT="$P/routing-study"
BUNDLE="${PRODUCTION_BUNDLE:-$ROOT/results/measurement-study/production-code-v2}"
RUN="$ROOT/results/measurement-study/runtime/$SLURM_JOB_ID"
umask 077
mkdir -p "$RUN/tmp" "$RUN/cache"
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache" MPLCONFIGDIR="$RUN/cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$P/repo/src:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
cd "$BUNDLE"
export CUDA_VISIBLE_DEVICES=''
srun "$P/envs/vllm-cu129/bin/python" -B -u -m measurement_study.gemma_source_recovery --bundle "$BUNDLE"

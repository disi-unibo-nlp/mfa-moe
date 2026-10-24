#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
ROOT="$P/routing-study"
BUNDLE="$ROOT/results/measurement-study/label-recovery-code-v2"
RUN="$ROOT/results/measurement-study/runtime/$SLURM_JOB_ID"
umask 077
mkdir -p "$RUN/tmp" "$RUN/cache"
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$ROOT:$BUNDLE/src:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
cd "$ROOT"
srun "$P/envs/vllm-cu129/bin/python" -B -u -m measurement_study.recovery_label_preflight

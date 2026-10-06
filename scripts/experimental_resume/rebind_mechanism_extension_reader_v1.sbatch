#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --job-name=st-extension-reader-price-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett &&
   "${SLURM_JOB_PARTITION:-}" == lrd_all_serial ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/seal_mechanism_extension_reader_v1.py"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == 72dfb6098e8da24118781f0f97f080f45247e2b1302dbe572c0b8a05e0a6ad49 ]] || exit 2
module load python/3.11.7
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-extension-reader-price-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
srun --ntasks=1 "$PY" -B -u "$DRIVER" rebind

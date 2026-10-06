#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=native-finalization-cpu-v3
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* &&
   "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
STAGE=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
module load python/3.11.7
umask 077
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/nf-cpu-$SLURM_JOB_ID
mkdir -p "$RUNTIME/tmp" "$RUNTIME/mpl" "$RUNTIME/xdg"
export TMPDIR="$RUNTIME/tmp" MPLCONFIGDIR="$RUNTIME/mpl" XDG_CACHE_HOME="$RUNTIME/xdg"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/scripts/experimental_resume:$REPO/src:$STAGE/code/s1-9a61e32f48c04c24"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --overlap --ntasks=1 env TMPDIR="$RUNTIME/tmp" "$PY" -B -u "$REPO/scripts/experimental_resume/native_finalization_correction_v3.py" "${1:?prepare, measure or finalize}"

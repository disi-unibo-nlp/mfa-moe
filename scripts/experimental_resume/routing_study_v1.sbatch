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
#SBATCH --job-name=routing-study-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/routing-study-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/routing-study-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
module load python/3.11.7
umask 077
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/routing-study-v1/job-$SLURM_JOB_ID
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch"
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
export VLLM_BATCH_INVARIANT=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
[[ $# -ge 1 ]] || exit 2
MODE="$1"
shift
case "$MODE" in
  inventory|discovery|report)
    [[ "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
    srun --ntasks=1 env TMPDIR="$RUNTIME/tmp" "$PY" -B -u "$REPO/scripts/experimental_resume/routing_study_v1.py" "$MODE" "$@" ;;
  prepare|analyze)
    [[ "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
    srun --ntasks=1 env TMPDIR="$RUNTIME/tmp" "$PY" -B -u "$REPO/scripts/experimental_resume/dense_recovery_v2.py" "$MODE" "$@" ;;
  qualify|run)
    [[ "${SLURM_JOB_PARTITION:-}" == boost_usr_prod && -n "${SLURM_JOB_GPUS:-}" ]] || exit 2
    export OMP_NUM_THREADS=8
    srun --ntasks=1 env TMPDIR="$RUNTIME/tmp" "$PY" -B -u "$REPO/scripts/experimental_resume/dense_recovery_v2.py" "$MODE" "$@" ;;
  *) exit 2 ;;
esac

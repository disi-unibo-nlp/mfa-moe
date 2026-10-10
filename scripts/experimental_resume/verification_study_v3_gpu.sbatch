#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" == lrdn* ]] || exit 2
[[ "${SLURM_CLUSTER_NAME:-}" == leonardo && "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == boost_usr_prod ]] || exit 2
[[ $# == 5 ]] || exit 2
CAMPAIGN=$1
STAGE=$2
MODEL_KEY=$3
RESULT_ROOT=$4
JOB_CARD=$5
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
module load python/3.11.7
module load cuda/12.6
umask 077
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/cv3-runtime-$SLURM_JOB_ID
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch"
export VLLM_RPC_BASE_PATH="$RUNTIME"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_BATCH_INVARIANT=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$END_TEXT" && "$END_TEXT" != N/A && "$END_TEXT" != *$'\n'* ]] || exit 2
END_EPOCH="$(date -d "$END_TEXT" +%s)"
[[ "$END_EPOCH" -gt "$(date +%s)" ]] || exit 2
# This first wave performs engineering qualification only. Production tasks
# require separately measured, bound job cards and scientific prerequisites.
if [[ "$STAGE" == qualify ]]; then
  srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/verification_study_v3.py" qualify \
    --campaign "$CAMPAIGN" --model "$MODEL_KEY" --root "$RESULT_ROOT" \
    --output "$RESULT_ROOT/$MODEL_KEY/qualification" --deadline-epoch "$((END_EPOCH-300))"
else
  srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/verification_study_v3_task.py" \
    "$JOB_CARD" --deadline-epoch "$((END_EPOCH-300))"
fi

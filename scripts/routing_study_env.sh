#!/bin/bash
# Source only inside an authorized Slurm allocation. Existing environments stay read-only.
set -euo pipefail
[[ "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname)" == lrdn* ]] || exit 2
[[ "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == boost_usr_prod ]] || exit 2
STUDY_REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study
STUDY_PROJECT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
STUDY_RUN="$STUDY_REPO/results/routing-study/smoke-${SLURM_JOB_ID}"
STUDY_STAGE=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/rs-${SLURM_JOB_ID}
module load python/3.11.7 cuda/12.6
umask 077
mkdir -p "$STUDY_RUN" "$STUDY_STAGE"
export TMPDIR="$STUDY_STAGE"
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
export VLLM_RPC_BASE_PATH="$STUDY_STAGE" VLLM_USE_V2_MODEL_RUNNER=0
export HF_HUB_CACHE="$STUDY_PROJECT/cache/hf/hub"
export HF_HOME="$STUDY_STAGE/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export XDG_CACHE_HOME="$STUDY_STAGE/cache" TORCH_HOME="$STUDY_STAGE/torch"
export TRITON_CACHE_DIR="$STUDY_STAGE/triton" TORCHINDUCTOR_CACHE_DIR="$STUDY_STAGE/inductor"
export MPLCONFIGDIR="$STUDY_STAGE/matplotlib" VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$STUDY_PROJECT/envs/vllm-cu129/bin:$PATH"
export PYTHONPATH="$STUDY_REPO/src:$STUDY_PROJECT/envs/vllm-cu129/lib/python3.11/site-packages:$STUDY_PROJECT/envs/correlation-client-3.11/lib/python3.11/site-packages"
STUDY_PYTHON="$STUDY_PROJECT/envs/vllm-cu129/bin/python"
cd "$STUDY_REPO"

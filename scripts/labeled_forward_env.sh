#!/bin/bash
# Sourced only by the native labeled-corpus Slurm wrappers.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr ]] || exit 2
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
REPO="$P/repo"
ROOT="$REPO/results/correlation_pipeline/labeled-forward-v1"
CODE="$ROOT/code"
PHASE="${1:?phase}"
SOURCE="${2:?source}"
case "$SOURCE" in gpt|gemma|qwen36|nemotron) ;; *) exit 2 ;; esac
if [[ "$PHASE" == forward ]]; then
    [[ "${SLURM_JOB_PARTITION:-}" == boost_usr_prod && "$(hostname)" == lrdn* ]] || exit 2
    module load python/3.11.7 cuda/12.6
else
    [[ "${SLURM_JOB_PARTITION:-}" == lrd_all_serial ]] || exit 2
    module load python/3.11.7
    export CUDA_VISIBLE_DEVICES=
fi
umask 077
RUN="$ROOT/runtime/$SOURCE-$PHASE-${SLURM_JOB_ID}"
mkdir -p "$RUN/tmp" "$RUN/cache"
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$RUN/cache/hf" HF_HUB_CACHE="$P/cache/hf/hub"
export TORCH_HOME="$RUN/cache/torch" TRITON_CACHE_DIR="$RUN/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor" DSPY_CACHEDIR="$RUN/cache/dspy"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "$PHASE" == analyze ]]; then
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
fi
export PYTHONPATH="$CODE/src:$P/envs/vllm-cu129/lib/python3.11/site-packages:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
if [[ "$SOURCE" == gpt && "$PHASE" == forward ]]; then
    export PYTHONPATH="$P/envs/forward-kernels-0.16:$PYTHONPATH"
fi
export PATH="$P/envs/vllm-cu129/bin:$PATH"
cd "$REPO"
srun --ntasks=1 --nodes=1 "$P/envs/vllm-cu129/bin/python" "$CODE/scripts/run_labeled_forward.py" "$PHASE" "$SOURCE"

#!/bin/bash
# Shared native pilot setup; every new output/cache remains in private WORK.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname)" == lrdn* ]] || exit 2
[[ "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == boost_usr_prod ]] || exit 2
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
REPO="$P/repo"
ROOT="$REPO/results/correlation_pipeline/routing-dynamics-v1/pilot"
module load python/3.11.7 cuda/12.6
umask 077
RUNTIME="$ROOT/runtime/${SLURM_JOB_ID}"
mkdir -p "$RUNTIME/tmp" "$RUNTIME/cache"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/cache"
export HF_HOME="$RUNTIME/cache/hf" HF_HUB_CACHE="$P/cache/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="$RUNTIME/cache/torch" TRITON_CACHE_DIR="$RUNTIME/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME/cache/inductor" DSPY_CACHEDIR="$RUNTIME/cache/dspy"
export MPLCONFIGDIR="$RUNTIME/cache/matplotlib" VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$P/envs/vllm-cu129/bin:$PATH"
# Reuse the existing judge dependencies read-only. Bytecode and caches are redirected above.
JUDGE_PACKAGES=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-88189c22f214ed65/lib/python3.11/site-packages
export PYTHONPATH="$REPO/src:$P/envs/vllm-cu129/lib/python3.11/site-packages:$P/envs/correlation-client-3.11/lib/python3.11/site-packages:$JUDGE_PACKAGES"
PYTHON="$P/envs/vllm-cu129/bin/python"
cd "$REPO"
"$PYTHON" scripts/run_routing_dynamics.py preflight --root "$ROOT"

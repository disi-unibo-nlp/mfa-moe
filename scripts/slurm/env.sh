# Shared setup of every Slurm job. Source it from the repo root: source scripts/slurm/env.sh
: "${SLURM_JOB_ID:?run this inside a Slurm job}"
set -euo pipefail
umask 077
module load python/3.11.7 cuda/12.6
export MFA=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
# Scratch files go in a short per-job folder: vLLM sockets fail when their path passes 107 characters.
RUN=$MFA/r/$SLURM_JOB_ID
mkdir -p "$RUN" "$MFA/logs"
trap 'test -z "$(jobs -p)" || { kill $(jobs -p) 2>/dev/null || true; sleep 5; }; rm -rf "$RUN"' EXIT
export PATH=$MFA/envs/vllm-cu129/bin:$PATH PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1
export HF_HOME=$MFA/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TIKTOKEN_ENCODINGS_BASE=$MFA/cache/tiktoken TIKTOKEN_RS_CACHE_DIR=$RUN/tiktoken  # gpt-oss tokenizer file; compute nodes are offline
export TMPDIR=$RUN VLLM_RPC_BASE_PATH=$RUN XDG_CACHE_HOME=$RUN MPLCONFIGDIR=$RUN/mpl DSPY_CACHEDIR=$RUN/dspy
export TORCH_HOME=$RUN/torch TRITON_CACHE_DIR=$RUN/triton TORCHINDUCTOR_CACHE_DIR=$RUN/inductor CUDA_CACHE_PATH=$RUN/cuda VLLM_CACHE_ROOT=$RUN/vllm
export VLLM_USE_FLASHINFER_SAMPLER=0 TORCH_CUDA_ARCH_LIST=8.0
# One BLAS thread unless the launcher asks for more (sbatch --export=ALL,THREADS=4): fork pools oversubscribe otherwise.
export OMP_NUM_THREADS=${THREADS:-1} OPENBLAS_NUM_THREADS=${THREADS:-1} MKL_NUM_THREADS=${THREADS:-1}
# One port per job: two 2-GPU jobs can land on the same node.
export PORT=$((20000 + SLURM_JOB_ID % 20000))
export URL=http://127.0.0.1:$PORT/v1
# The judge client (DSPy) lives in its own environment.
export DSPY=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-88189c22f214ed65

# Start scripts/serve/MODEL.sh in the background and wait until it answers (cold start takes ~16 min; give up at 45).
start_server() {
    curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null && { echo "port $PORT is taken by another server"; exit 1; }
    bash "scripts/serve/$1.sh" > "$MFA/logs/$SLURM_JOB_ID-$1-server.log" 2>&1 &
    until curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; do
        kill -0 $! && ((SECONDS < 2700)) || { tail -50 "$MFA/logs/$SLURM_JOB_ID-$1-server.log"; exit 1; }
        sleep 5
    done
}

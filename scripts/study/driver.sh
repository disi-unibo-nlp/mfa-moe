#!/bin/bash
# Reduced final-routing driver job. Submit from the login node, e.g.
#   sbatch -A iscrc_miosr -p boost_usr_prod --qos=normal -N1 --ntasks=1 --gres=gpu:2 \
#          --cpus-per-task=16 --mem=120G -t 12:00:00 -J fr-gpt-0 -o LOG \
#          scripts/study/driver.sh run --model gpt --shard 0/2
# All arguments are passed to `python -m moe_exp.study.driver`.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm; direct execution on login nodes is forbidden}"
: "${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
test "$(id -un)" = lmolfett
cd "${RELEASE_ROOT:-/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo}"
module load python/3.11.7
module load cuda/12.6
ENV=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129
RUN="/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/runtime/drv-${SLURM_JOB_ID}"
RPC="/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/${SLURM_JOB_ID}"
mkdir -p "$RUN" "$RPC"
trap 'rm -rf "$RPC" "$RUN"' EXIT
# Unix sockets (ZMQ IPC, torch symmetric-memory IpcChannel) live under TMPDIR: keep it short (107-char limit).
export TMPDIR="$RPC" VLLM_RPC_BASE_PATH="$RPC"
export XDG_CACHE_HOME="$RUN/xdg" TORCH_HOME="$RUN/torch" TRITON_CACHE_DIR="$RUN/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN/inductor" CUDA_CACHE_PATH="$RUN/cuda" VLLM_CACHE_ROOT="$RUN/vllm"
export DSPY_CACHEDIR="$RUN/dspy" MPLCONFIGDIR="$RUN/matplotlib"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export TIKTOKEN_ENCODINGS_BASE=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/tiktoken TIKTOKEN_RS_CACHE_DIR="$RUN/tiktoken"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/scripts/experimental_resume"
THREADS=$(( SLURM_CPUS_PER_TASK < 4 ? SLURM_CPUS_PER_TASK : 4 ))
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi
echo "job ${SLURM_JOB_ID} on $(hostname) args: $*"
"$ENV/bin/python" -B -m "${STUDY_MODULE:-moe_exp.study.driver}" "$@"

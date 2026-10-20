#!/bin/bash
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm; direct login execution is forbidden}"
: "${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
test "$(id -un)" = lmolfett
case "$(hostname -f)" in *leonardo*) ;; *) exit 71 ;; esac
if [[ $# -ne 3 ]]; then
    echo 'usage: rescue_reader.sbatch ENV MODEL COHORT' >&2
    exit 2
fi
cd /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
env_root="$1"
module load python/3.11.7
module load cuda/12.6
tmp="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/final-routing-rescue/reader-${SLURM_JOB_ID}"
mkdir -p "$tmp"
export TMPDIR="$tmp" XDG_CACHE_HOME="$tmp/xdg" TORCH_HOME="$tmp/torch"
export TRITON_CACHE_DIR="$tmp/triton" VLLM_CACHE_ROOT="$tmp/vllm" MPLCONFIGDIR="$tmp/matplotlib"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export TIKTOKEN_ENCODINGS_BASE=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/tiktoken
export TIKTOKEN_RS_CACHE_DIR="$tmp/tiktoken"
export PYTHONDONTWRITEBYTECODE=1 PATH="$env_root/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/scripts/experimental_resume${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
exec srun --kill-on-bad-exit=1 env TMPDIR="$tmp" "$env_root/bin/python" -B -m moe_exp.study.rescue_reader run --model "$2" --cohort "$3"

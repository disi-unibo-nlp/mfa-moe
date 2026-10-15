#!/bin/bash
# Resources, account, paths and time are explicitly supplied by final_routing.py.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm; direct execution on login nodes is forbidden}"
: "${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
test "$(id -un)" = lmolfett
case "$(hostname -f)" in *leonardo*) ;; *) exit 71 ;; esac
if [[ $# -ne 3 ]]; then
    echo 'usage: job.sh ENV BINDING CARD' >&2
    exit 2
fi
module load python/3.11.7
module load cuda/12.6
final_routing_tmp="/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs/final-routing-20261015-r1/runtime/fr-${SLURM_JOB_ID}"
mkdir -p "$final_routing_tmp"
export TMPDIR="$final_routing_tmp"
export XDG_CACHE_HOME="$final_routing_tmp/xdg" TORCH_HOME="$final_routing_tmp/torch"
export TRITON_CACHE_DIR="$final_routing_tmp/triton" TORCHINDUCTOR_CACHE_DIR="$final_routing_tmp/inductor"
export CUDA_CACHE_PATH="$final_routing_tmp/cuda" VLLM_CACHE_ROOT="$final_routing_tmp/vllm"
export DSPY_CACHEDIR="$final_routing_tmp/dspy" MPLCONFIGDIR="$final_routing_tmp/matplotlib"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export PYTHONDONTWRITEBYTECODE=1
export PATH="$1/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/scripts/experimental_resume${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
final_routing_python="$1/bin/python"
final_routing_binding="$2"
trap '"$final_routing_python" -B -m moe_exp.study.maintenance "$final_routing_binding"' EXIT
srun --kill-on-bad-exit=1 "$final_routing_python" -B -m moe_exp.study worker --binding "$2" --card "$3"

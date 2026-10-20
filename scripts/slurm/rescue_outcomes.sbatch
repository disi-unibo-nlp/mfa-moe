#!/bin/bash
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm; direct login execution is forbidden}"
: "${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
test "$(id -un)" = lmolfett
case "$(hostname -f)" in *leonardo*) ;; *) exit 71 ;; esac
if [[ $# -ne 3 ]]; then
    echo 'usage: rescue_outcomes.sbatch ENV MODEL COHORT' >&2
    exit 2
fi
cd /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
env_root="$1"
model="$2"
module load python/3.11.7
export PYTHONDONTWRITEBYTECODE=1
export PATH="$env_root/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/scripts/experimental_resume${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export TIKTOKEN_ENCODINGS_BASE=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/tiktoken
tmp="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/final-routing-rescue/extract-${SLURM_JOB_ID}"
mkdir -p "$tmp"
export TMPDIR="$tmp" TIKTOKEN_RS_CACHE_DIR="$tmp/tiktoken"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec srun --kill-on-bad-exit=1 env TMPDIR="$tmp" "$env_root/bin/python" -B -m moe_exp.study.outcomes \
    --model "$model" --shard 0/1 --design rescue --cohort "$3"

#!/bin/bash
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm; direct login execution is forbidden}"
: "${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
test "$(id -un)" = lmolfett
case "$(hostname -f)" in *leonardo*) ;; *) exit 71 ;; esac
if [[ $# -ne 2 ]]; then
    echo 'usage: rescue_analyze.sbatch ENV COHORT' >&2
    exit 2
fi
cd /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
env_root="$1"
module load python/3.11.7
tmp="/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/r/${SLURM_JOB_ID}"
mkdir -p "$tmp"
export TMPDIR="$tmp" MPLCONFIGDIR="$tmp/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$env_root/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD/scripts/experimental_resume${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
exec srun --kill-on-bad-exit=1 env TMPDIR="$tmp" VLLM_RPC_BASE_PATH="$tmp" "$env_root/bin/python" -B -m moe_exp.study.rescue analyze --cohort "$2"

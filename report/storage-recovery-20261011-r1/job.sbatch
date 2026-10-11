#!/bin/bash
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through the authorized bounded cleanup card}"
test "$(id -un)" = lmolfett
case "$(hostname -f)" in *leonardo*) ;; *) exit 71 ;; esac
module load python/3.11.7
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp
srun --kill-on-bad-exit=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B \
    /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/storage-recovery-20261011-r1/worker.py

#!/bin/bash
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Run qualification through Slurm}"
test "$(id -un)" = lmolfett
case "$(hostname -f)" in *leonardo*) ;; *) exit 71 ;; esac
module load python/3.11.7
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp
srun --kill-on-bad-exit=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B \
    scripts/research_campaign_v3_r3/smoke.py --output report/research-campaign-v3-20261011-r3/READINESS.json

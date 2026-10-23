#!/bin/bash
# Apply the frozen pick rule to a half-A receipt and launch half B (Slurm CPU job queued by launch.sh).
# Runs from this checkout, because sealing half B takes the commit of a git checkout.
#   pick.sh EFFECTS MODEL HALF_B_NAME HALF_B_SHARDS LOCAL_TESTS
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm; direct execution on login nodes is forbidden}"
cd "$(dirname "$0")/../.."
module load python/3.11.7
RUN="/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/${SLURM_JOB_ID}"
mkdir -p "$RUN"
trap 'rm -rf "$RUN"' EXIT
export TMPDIR="$RUN" HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf HF_HUB_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD/scripts/experimental_resume" \
       WHOLE_ROOT="$(dirname "$(dirname "$(dirname "$1")")")/$3" OMP_NUM_THREADS=1
/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -m moe_exp.outcome_steering pick \
  --effects "$1" --model "$2" --name "$3" --shards "$4" --local-tests "$5"

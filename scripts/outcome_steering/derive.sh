#!/bin/bash
# CPU chain for one model of a capture release (labels, class usage, r2 selection, signal table,
# attempt selection), run from a frozen code snapshot. Receipts go to <release>/<model>/.
#   sbatch -A iscrc_miosr -p boost_usr_prod --qos=normal -N1 --ntasks=1 --cpus-per-task=4 --mem=30G \
#          -t 04:00:00 -J oc-derive-gpt -o LOG scripts/outcome_steering/derive.sh SNAPSHOT RELEASE gpt
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm; direct execution on login nodes is forbidden}"
snapshot="${1:?code snapshot}"; release="${2:?capture release directory}"; model="${3:?model}"
module load python/3.11.7
RUN="/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/${SLURM_JOB_ID}"
mkdir -p "$RUN"
trap 'rm -rf "$RUN"' EXIT
export TMPDIR="$RUN" HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf HF_HUB_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$snapshot/src:$snapshot/scripts/experimental_resume" \
       STUDY_DATA_ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/final-routing-20261015-r1 \
       WHOLE_ROOT="$release" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$snapshot"
echo "job ${SLURM_JOB_ID} on $(hostname): derive $model from $snapshot"
/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -m moe_exp.outcome_steering derive \
  --release "$release" --models "$model" --workers "${SLURM_CPUS_PER_TASK:-4}"

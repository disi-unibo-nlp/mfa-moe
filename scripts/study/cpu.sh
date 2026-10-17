#!/bin/bash
# CPU job for the reduced final-routing analysis, e.g.
#   sbatch -A iscrc_miosr -p lrd_all_serial --qos=normal -N1 --ntasks=1 --cpus-per-task=4 --mem=30G \
#          -t 04:00:00 -J fr-cpu-lexical -o LOG scripts/study/cpu.sh moe_exp.study.lexical fit
# First argument: python module; the rest is passed through.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm; direct execution on login nodes is forbidden}"
cd /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
module load python/3.11.7
ENV=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129
RUN="/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/${SLURM_JOB_ID}"
mkdir -p "$RUN"
trap 'rm -rf "$RUN"' EXIT
export TMPDIR="$RUN" MPLCONFIGDIR="$RUN/mpl" HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TIKTOKEN_ENCODINGS_BASE=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/tiktoken TIKTOKEN_RS_CACHE_DIR="$RUN/tiktoken"
export PYTHONPATH="$PWD/src:$PWD/scripts/experimental_resume"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
module="$1"; shift
echo "job ${SLURM_JOB_ID} on $(hostname): $module $*"
"$ENV/bin/python" -B -m "$module" "$@"

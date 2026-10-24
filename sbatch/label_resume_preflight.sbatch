#!/bin/bash
#SBATCH --job-name=mfa-label-resume-preflight
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lorenzo.molfetta@unibo.it
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
BUNDLE="$P/routing-study/results/measurement-study/label-resume-v1/code"
ENV_ROOT=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-88189c22f214ed65
RUN="$P/routing-study/results/measurement-study/runtime/$SLURM_JOB_ID"
umask 077
mkdir -p "$RUN/tmp" "$RUN/cache"
module load python/3.11.7 cuda/12.6
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache" DSPY_CACHEDIR="$RUN/cache/dspy"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$BUNDLE/src:$ENV_ROOT/lib/python3.11/site-packages:$P/envs/vllm-cu129/lib/python3.11/site-packages"
cd "$BUNDLE"
srun "$ENV_ROOT/bin/python" -u label_resume_preflight.py

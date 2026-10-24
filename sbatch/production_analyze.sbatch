#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lorenzo.molfetta@unibo.it
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.err
#SBATCH --job-name=mfa-production-analysis
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr ]] || exit 2
case "${1:-}" in gpt|glm|qwen330b|qwen36|nemotron|gemma|qwen35) ;; *) exit 2;; esac
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
ROOT="$P/routing-study"
BUNDLE="${PRODUCTION_BUNDLE:?Explicit frozen v3 bundle required}"
MANIFEST_DIR="${ANALYSIS_MANIFEST_DIR:?Explicit ready.json manifest directory required}"
STORAGE_ROOT="${ANALYSIS_STORAGE_ROOT:?Explicit campaign-v3 storage root required}"
ANALYSIS_COHORT="${ANALYSIS_COHORT:-B}"
OUTPUT_ROOT="${ANALYSIS_OUTPUT_ROOT:?Explicit fresh output root below results/measurement-study/campaign-v3 required}"
OUTCOMES="${ANALYSIS_OUTCOMES:?Explicit corrected-outcome sidecar or outcomes_v3.latest.json pointer required}"
RUN="$ROOT/results/measurement-study/runtime/$SLURM_JOB_ID"
umask 077
mkdir -p "$RUN/tmp" "$RUN/cache"
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache" MPLCONFIGDIR="$RUN/cache/mpl"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$BUNDLE/src:$BUNDLE:$BUNDLE/scripts:$P/envs/vllm-cu129/lib/python3.11/site-packages"
cd "$BUNDLE"
export CUDA_VISIBLE_DEVICES=''
srun "$P/envs/vllm-cu129/bin/python" -B -u -m measurement_study.production_analyze "$1" --workers "${ANALYSIS_WORKERS:-6}" \
  --bundle "$BUNDLE" --manifest-dir "$MANIFEST_DIR" --storage-root "$STORAGE_ROOT" \
  --cohort "$ANALYSIS_COHORT" --output-root "$OUTPUT_ROOT" --outcomes "$OUTCOMES"

#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=st-mechanism-seal
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
PREP="$REPO/scripts/experimental_resume/prepare_mechanism_validation_v1.py"
RUNNER="$REPO/scripts/experimental_resume/run_mechanism_validation_v1.py"
[[ "$(sha256sum "$PREP" | cut -d' ' -f1)" == 9de97d7d9d9e2e169fe057fe63748cf0eca0708212d42469c3e8ed7652e8e287 ]] || exit 2
[[ "$(sha256sum "$RUNNER" | cut -d' ' -f1)" == 72d82b30ce1af567bed17f122651d861241fc5342706799822508951e1a31bbe ]] || exit 2
[[ -f "$REPO/report/experimental-resume-v1/MECHANISM_ENGINE_1024_QUAL_RESULT_v1.json" ]] || exit 2
[[ -f "$ROOT/runs/routing-control-v1/dense-mechanism/ratings-v2-7497e6aafbaf689c-497b2e8f57646826/SUMMARY.json" ]] || exit 2
module load python/3.11.7
umask 077
RUNTIME="$ROOT/runtime-mechanism-validation-seal-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume:$ROOT/code/s1-9a61e32f48c04c24"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$PREP" --max-wall-seconds 86400

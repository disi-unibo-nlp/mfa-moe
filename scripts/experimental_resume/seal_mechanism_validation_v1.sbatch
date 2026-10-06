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
#SBATCH --job-name=st-mechanism-completion
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
MANIFEST="$REPO/report/experimental-resume-v1/MECHANISM_VALIDATION_MANIFEST_v1.json"
PRICE="$REPO/report/experimental-resume-v1/MECHANISM_VALIDATION_PRICE_v1.json"
DRIVER="$REPO/scripts/experimental_resume/run_mechanism_validation_v1.py"
[[ -f "$MANIFEST" && -f "$PRICE" && -f "$DRIVER" ]] || exit 2
module load python/3.11.7
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
MANIFEST_SHA="$("$PY" -B -c 'import json,sys; m=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); assert p["manifest_sha256"]==m["sha256"] and p["status"]=="PASS_COMPLETE_STAGE"; print(m["sha256"])' "$MANIFEST" "$PRICE")"
OUT="$ROOT/runs/routing-control-v1/mechanism-validation-v1-${MANIFEST_SHA:0:16}"
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-mechanism-validation-seal-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume:$ROOT/code/s1-9a61e32f48c04c24"
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --out "$OUT" --seal-stage
[[ -f "$OUT/STAGE_COMPLETION.json" ]] || exit 2

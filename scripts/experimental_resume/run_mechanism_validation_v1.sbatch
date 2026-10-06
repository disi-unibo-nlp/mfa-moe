#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --job-name=st-mechanism-1024
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
OVERLAY="$ROOT/addenda/ordered/9727c10299b71e7a/moe_exp_src"
BASE="$ROOT/code/s1-9a61e32f48c04c24"
MANIFEST="$REPO/report/experimental-resume-v1/MECHANISM_VALIDATION_MANIFEST_v1.json"
PRICE="$REPO/report/experimental-resume-v1/MECHANISM_VALIDATION_PRICE_v1.json"
DRIVER="$REPO/scripts/experimental_resume/run_mechanism_validation_v1.py"
SHARD_INDEX="${SHARD_INDEX:-${1:-}}"
[[ "$SHARD_INDEX" =~ ^[0-9]+$ && -f "$MANIFEST" && -f "$PRICE" && -f "$DRIVER" && -d "$OVERLAY" ]] || exit 2
module load python/3.11.7
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
MANIFEST_SHA="$("$PY" -B -c 'import json,sys; m=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); assert p["manifest_sha256"]==m["sha256"] and p["status"]=="PASS_COMPLETE_STAGE" and 0 <= int(sys.argv[3]) < len(p["shards"]); print(m["sha256"])' "$MANIFEST" "$PRICE" "$SHARD_INDEX")"
OUT="$ROOT/runs/routing-control-v1/mechanism-validation-v1-${MANIFEST_SHA:0:16}"
SLURM_END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$SLURM_END_TEXT" && "$SLURM_END_TEXT" != N/A ]] || exit 2
SLURM_END_EPOCH="$(date -d "$SLURM_END_TEXT" +%s)"
export STEER_DEADLINE_EPOCH="$((SLURM_END_EPOCH - 180))"
[[ "$STEER_DEADLINE_EPOCH" -gt "$(date +%s)" ]] || exit 2
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-mechanism-validation-v1-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin:$PATH
export PYTHONPATH="$OVERLAY:$REPO/src:$REPO/scripts/experimental_resume:$BASE:$BASE/moe_exp_src"
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --out "$OUT" --cpu-preflight
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --out "$OUT" --overlay "$OVERLAY" --shard-index "$SHARD_INDEX" --batch-size 4
[[ -f "$OUT/shard-$(printf '%03d' "$SHARD_INDEX")/MECHANISM_COMPLETION.json" ]] || exit 2

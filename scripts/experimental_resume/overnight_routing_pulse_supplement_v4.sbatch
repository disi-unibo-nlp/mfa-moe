#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=st-pulse-supplement-v4
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
OVERLAY="$ROOT/addenda/ordered/9727c10299b71e7a/moe_exp_src"
BASE="$ROOT/code/s1-9a61e32f48c04c24"
module load python/3.11.7
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
SLURM_END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$SLURM_END_TEXT" && "$SLURM_END_TEXT" != N/A && "$SLURM_END_TEXT" != *$'\n'* ]] || exit 2
SLURM_END_EPOCH="$(date -d "$SLURM_END_TEXT" +%s)"
export STEER_DEADLINE_EPOCH="$((SLURM_END_EPOCH - 180))"
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-overnight-pulse-supplement-v4-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp" "$RUNTIME/entry-receipts"
export OVERNIGHT_ENTRY_RECEIPT_DIR="$RUNTIME/entry-receipts"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin:$PATH
export PYTHONPATH="$OVERLAY:$REPO/src:$REPO/scripts/experimental_resume:$BASE:$BASE/moe_exp_src"
ENTRY="$REPO/scripts/experimental_resume/overnight_routing_pulse_entry_v4.py"
QUAL_SHA="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$REPO/report/experimental-resume-v1/OVERNIGHT_PULSE_SUPPLEMENT_MANIFEST_v4.json")"
QUAL_OUT="$ROOT/runs/routing-control-v1/overnight-pulse-supplement-v4-${QUAL_SHA:0:16}"
srun --ntasks=1 "$PY" -B -u "$ENTRY" overnight_routing_pulse_supplement_v4 --startup-import-check
srun --ntasks=1 "$PY" -B -u "$ENTRY" overnight_routing_pulse_supplement_v4 --cpu-preflight
srun --ntasks=1 "$PY" -B -u "$ENTRY" overnight_routing_pulse_supplement_v4 --out "$QUAL_OUT"
[[ -f "$REPO/report/experimental-resume-v1/OVERNIGHT_PULSE_SUPPLEMENT_RESULT_v4.json" ]] || exit 2

#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --job-name=st-score-api-cal-v3
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
BASE="$ROOT/code/s1-9a61e32f48c04c24"
OVERLAY="$ROOT/addenda/ordered/9727c10299b71e7a/moe_exp_src"
DRIVER="$ROOT/addenda/score-api-calibration/9c10b0aab240f52d/run_score_api_calibration_v3.py"
MANIFEST="$REPO/report/experimental-resume-v1/SCORE_API_CALIBRATION_MANIFEST_v3.json"
OUT="$ROOT/runs/routing-control-v1/score-api-calibration-b4aa1c878c25fe32"
RUNTIME="$ROOT/runs/routing-control-v1/runtime-score-api-$SLURM_JOB_ID"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == 8f0701dc27cd9d2c01e50ad16d4cab1311263126a40f5d8d4acef3edf7c23be1 ]] || exit 2
SLURM_END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$SLURM_END_TEXT" && "$SLURM_END_TEXT" != N/A ]] || exit 2
SLURM_END_EPOCH="$(date -d "$SLURM_END_TEXT" +%s)"
export STEER_DEADLINE_EPOCH="$((SLURM_END_EPOCH - 180))"
module load python/3.11.7
umask 077
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_BATCH_INVARIANT=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
export PYTHONPATH="$OVERLAY:$BASE:$BASE/moe_exp_src"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --out "$OUT"

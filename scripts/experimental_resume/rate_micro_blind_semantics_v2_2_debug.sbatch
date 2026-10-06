#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --job-name=st-micro-blind-v22
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "$(hostname -s)" != login* ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/rate_micro_blind_semantics_v2_2.py"
PRICE="$REPO/report/experimental-resume-v1/CAUSAL_MICROSCREEN_BLIND_RATING_PRICE_v2.2.json"
SLICE_PRICE="$REPO/report/experimental-resume-v1/CAUSAL_MICROSCREEN_BLIND_DEBUG_PRICE_v2.2.json"
RUBRIC="$REPO/report/experimental-resume-v1/CAUSAL_MICROSCREEN_SEMANTIC_RUBRIC_v2.md"
OUT="$ROOT/runs/routing-control-v1/micro-screen-semantic-v22-1e56570b050828e3"
RUNTIME="$ROOT/runs/routing-control-v1/runtime-micro-semantic-v22-$SLURM_JOB_ID"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == ee339bec443efbf79eb4ff0d3a53e06af416d219d472e3a476f2333c7fe923d2 ]] || exit 2
[[ "$(sha256sum "$PRICE" | cut -d' ' -f1)" == 0baa4db6f0b0160e140d11596835703aa3abfa2143410f95f93e36e0ed1fb9f1 ]] || exit 2
[[ "$(sha256sum "$SLICE_PRICE" | cut -d' ' -f1)" == bbeb0f3da51bb1e35b35d9458d5a13d509fc2c83aa0083fd4357ec5321a9dfe9 ]] || exit 2
[[ "$(sha256sum "$RUBRIC" | cut -d' ' -f1)" == a7a559a10e0df63ebc08a7703cc0ab06c764d7e07c6f7f07978bce26af2f359e ]] || exit 2
SLURM_END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$SLURM_END_TEXT" && "$SLURM_END_TEXT" != 'N/A' ]] || exit 2
SLURM_END_EPOCH="$(date -d "$SLURM_END_TEXT" +%s)"
export MICRO_RATING_DEADLINE_EPOCH="$((SLURM_END_EPOCH - 120))"
[[ "$MICRO_RATING_DEADLINE_EPOCH" -gt "$(date +%s)" ]] || exit 2
module load python/3.11.7
umask 077
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --out "$OUT"

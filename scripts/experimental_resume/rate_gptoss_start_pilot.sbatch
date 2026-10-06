#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=st-gptoss-start-pilot
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "$SLURM_JOB_ID" && "$(id -un)" == lmolfett ]] || exit 2
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery
DRIVER=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume/rate_gptoss_start_pilot.py
PRICE=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/GPTOSS_START_PILOT_PRICE_v1.json
[[ -n "$PILOT_OUT" && -n "$PILOT_DRIVER_SHA" && -n "$PILOT_PRICE_FILE_SHA" ]] || exit 2
[[ "$PILOT_OUT" == "$ROOT"/ratings-gptoss-start-pilot-* && "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == "$PILOT_DRIVER_SHA" && "$(sha256sum "$PRICE" | cut -d' ' -f1)" == "$PILOT_PRICE_FILE_SHA" ]] || exit 2
module load python/3.11.7
umask 077
RUNTIME="$ROOT/runtime-gptoss-start-pilot-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
srun --ntasks=1 "$PY" -B -u "$DRIVER" --out "$PILOT_OUT"

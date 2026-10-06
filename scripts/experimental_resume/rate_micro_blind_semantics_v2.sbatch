#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=st-micro-blind-rate-v2
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "$(hostname -s)" != login* ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/rate_micro_blind_semantics_v2.py"
PRICE="$REPO/report/experimental-resume-v1/CAUSAL_MICROSCREEN_BLIND_RATING_PRICE_v2.1.json"
RUBRIC="$REPO/report/experimental-resume-v1/CAUSAL_MICROSCREEN_SEMANTIC_RUBRIC_v2.md"
OUT="$ROOT/runs/routing-control-v1/micro-screen-semantic-v2p1-27cd70792e1ae110"
RUNTIME="$ROOT/runs/routing-control-v1/runtime-micro-semantic-v2-$SLURM_JOB_ID"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == 70da67380e6a7de80097cfb806a721e24042b3eae035560c03b0d83fc467d08b ]] || exit 2
[[ "$(sha256sum "$PRICE" | cut -d' ' -f1)" == 622d14c5ed04804e4c5733b196d453d64a21ee047856e2f7fd11d59d0f7d809d ]] || exit 2
[[ "$(sha256sum "$RUBRIC" | cut -d' ' -f1)" == a7a559a10e0df63ebc08a7703cc0ab06c764d7e07c6f7f07978bce26af2f359e ]] || exit 2
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

#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=00:35:00
#SBATCH --no-requeue
#SBATCH --job-name=st-dense-judge-parity
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
module load python/3.11.7
umask 077
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-judge-parity
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
DRIVER=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume/qualify_dense_judge.py
[[ -n "${JUDGE_OUT:-}" ]] || { echo 'JUDGE_OUT required' >&2; exit 2; }
[[ "$JUDGE_OUT" == "$ROOT"/results-* ]] || { echo 'output must be digest-bound under private study root' >&2; exit 2; }
[[ -f "$ROOT/FIXTURES.json" && ! -e "$JUDGE_OUT" ]] || { echo 'fixture missing or output exists' >&2; exit 2; }
RUNTIME="$ROOT/runtime-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
srun --ntasks=1 "$PY" -B -u "$DRIVER" --out "$JUDGE_OUT"

#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --job-name=st-deactivate-blind-v4
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/build_eligible_micro_blind_frame_v4.py"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == e16c526f8286f671f87cf4de93294466254949eebf453f5d924c7742af1c67ca ]] || exit 2
MANIFEST="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v4.json"
RUN_OUT="$ROOT/runs/routing-control-v1/eligible-deactivation-serial-v4-85b7f90ae7295d85"
BLIND_OUT="$ROOT/runs/routing-control-v1/blind-eligible-deactivation-v4-85b7f90ae7295d85"
RUNTIME="$ROOT/runs/routing-control-v1/runtime-eligible-blind-deactivation-v4-$SLURM_JOB_ID"
umask 077
mkdir -p "$RUNTIME"
export TMPDIR="$RUNTIME" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/addenda/ordered/9727c10299b71e7a/moe_exp_src:$ROOT/code/s1-9a61e32f48c04c24:$ROOT/code/s1-9a61e32f48c04c24/moe_exp_src:$REPO/scripts/experimental_resume"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --run-out "$RUN_OUT" --out "$BLIND_OUT"

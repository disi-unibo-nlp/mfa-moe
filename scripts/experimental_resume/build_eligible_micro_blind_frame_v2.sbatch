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
#SBATCH --job-name=st-deactivate-blind-v2
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/build_eligible_micro_blind_frame_v2.py"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == 71f2b66d764738139a643d36acf5098de1da71c31d13558ff607815590df8da1 ]] || exit 2
MANIFEST="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v3.json"
RUN_OUT="$ROOT/runs/routing-control-v1/eligible-deactivation-serial-v3-fa29f28dcf5c5a84"
BLIND_OUT="$ROOT/runs/routing-control-v1/blind-eligible-deactivation-v3-fa29f28dcf5c5a84"
RUNTIME="$ROOT/runs/routing-control-v1/runtime-eligible-blind-deactivation-v3-$SLURM_JOB_ID"
umask 077
mkdir -p "$RUNTIME"
export TMPDIR="$RUNTIME" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/scripts/experimental_resume:$ROOT/code/s1-9a61e32f48c04c24"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --run-out "$RUN_OUT" --out "$BLIND_OUT"

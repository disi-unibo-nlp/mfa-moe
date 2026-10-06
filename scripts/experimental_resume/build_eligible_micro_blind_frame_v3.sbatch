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
#SBATCH --job-name=st-eligible-blind-v3
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/build_eligible_micro_blind_frame_v3.py"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == d1a84a82e1e3d1e9ee43a3d00c4da70942c55d702d81b9cebd2e168c66795a51 ]] || exit 2
MANIFEST="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_MICRO_SERIAL_MANIFEST_v3.json"
RUN_OUT="$ROOT/runs/routing-control-v1/eligible-micro-serial-v3-320ecca116c7ab63"
BLIND_OUT="$ROOT/runs/routing-control-v1/blind-eligible-micro-v3-320ecca116c7ab63"
RUNTIME="$ROOT/runs/routing-control-v1/runtime-eligible-blind-micro-v3-$SLURM_JOB_ID"
umask 077
mkdir -p "$RUNTIME"
export TMPDIR="$RUNTIME" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/addenda/ordered/9727c10299b71e7a/moe_exp_src:$ROOT/code/s1-9a61e32f48c04c24:$ROOT/code/s1-9a61e32f48c04c24/moe_exp_src:$REPO/scripts/experimental_resume"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --run-out "$RUN_OUT" --out "$BLIND_OUT"

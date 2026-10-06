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
#SBATCH --job-name=st-eligible-blind-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/build_eligible_micro_blind_frame_v1.py"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == b39993cf61241ee3ca4edd425ee72454216293aecb25c31a214210ae275fa84d ]] || exit 2
case "${STUDY:-}" in
  positive)
    MANIFEST="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_MICRO_SERIAL_MANIFEST_v2.json"
    RUN_OUT="$ROOT/runs/routing-control-v1/eligible-micro-serial-v2-0f9dc13adfcae354"
    BLIND_OUT="$ROOT/runs/routing-control-v1/blind-eligible-micro-v2-0f9dc13adfcae354"
    ;;
  deactivation)
    MANIFEST="$REPO/report/experimental-resume-v1/CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v1.json"
    RUN_OUT="$ROOT/runs/routing-control-v1/eligible-deactivation-serial-v1-86582a4de3c16c0c"
    BLIND_OUT="$ROOT/runs/routing-control-v1/blind-eligible-deactivation-v1-86582a4de3c16c0c"
    ;;
  *) exit 2 ;;
esac
RUNTIME="$ROOT/runs/routing-control-v1/runtime-eligible-blind-$STUDY-$SLURM_JOB_ID"
umask 077
mkdir -p "$RUNTIME"
export TMPDIR="$RUNTIME" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/scripts/experimental_resume:$ROOT/code/s1-9a61e32f48c04c24"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --run-out "$RUN_OUT" --out "$BLIND_OUT"

#!/bin/bash
# Run inside an already authorized lrd_all_serial CPU allocation. No sbatch here.
set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || {
  echo 'extension CPU stages require the lmolfett Slurm allocation' >&2
  exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == lrd_all_serial ]] || {
  echo 'extension CPU stages require lrd_all_serial' >&2
  exit 2
}
phase="${1:-all}"
case "$phase" in all|units|rest) ;; *) echo 'phase must be all, units or rest' >&2; exit 2 ;; esac

repo=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
stage_root=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1
driver="$repo/scripts/experimental_resume/prepare_mechanism_extension_220_v1.py"
py=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
[[ -f "$driver" && -x "$py" ]] || { echo 'extension driver or Python missing' >&2; exit 2; }

umask 077
export TMPDIR="$stage_root/runtime-mechanism-extension-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src:$repo/scripts/experimental_resume"

if [[ "$phase" == all || "$phase" == units ]]; then
  srun --ntasks=1 "$py" -B -u "$driver" units
fi
if [[ "$phase" == all || "$phase" == rest ]]; then
  [[ "${EXT_READER_MAX_WALL_SECONDS:-}" =~ ^[0-9]+$ ]] || {
    echo 'set EXT_READER_MAX_WALL_SECONDS from the live GPU partition limit' >&2
    exit 2
  }
  srun --ntasks=1 "$py" -B -u "$driver" detector
  srun --ntasks=1 "$py" -B -u "$driver" selection
  srun --ntasks=1 "$py" -B -u "$driver" frame
  srun --ntasks=1 "$py" -B -u "$driver" price \
    --max-wall-seconds "$EXT_READER_MAX_WALL_SECONDS"
fi

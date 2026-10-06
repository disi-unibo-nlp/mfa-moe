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
#SBATCH --job-name=st-boundary-qual4-v2
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "$(hostname -s)" != login* ]] || exit 2

REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
BASE="$ROOT/code/s1-9a61e32f48c04c24"
OVERLAY="$ROOT/addenda/ordered/9727c10299b71e7a/moe_exp_src"
DRIVER="$REPO/scripts/experimental_resume/run_boundary_micro_screen.py"
MANIFEST="$REPO/report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v2.json"
OUT="$ROOT/runs/routing-control-v1/micro-screen-qual4-6dbc0af8df340880"
RUNTIME="$ROOT/runs/routing-control-v1/runtime-micro-qual4-v2-$SLURM_JOB_ID"

[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == 1c0c3f0768fd853bc7ce8654b257ccf3cc00c9bcd332ca6f55b93fa5a784816d ]] || exit 2
[[ -d "$BASE" && -d "$OVERLAY" && -f "$MANIFEST" ]] || exit 2
SLURM_END_TEXT="$(squeue -h -j "$SLURM_JOB_ID" -o '%e')"
[[ -n "$SLURM_END_TEXT" && "$SLURM_END_TEXT" != 'N/A' ]] || exit 2
SLURM_END_EPOCH="$(date -d "$SLURM_END_TEXT" +%s)"
# Keep 180 seconds for telemetry flush and vLLM process shutdown.
export STEER_DEADLINE_EPOCH="$((SLURM_END_EPOCH - 180))"
[[ "$STEER_DEADLINE_EPOCH" -gt "$(date +%s)" ]] || exit 2

module load python/3.11.7
umask 077
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
export PYTHONPATH="$OVERLAY:$BASE:$BASE/moe_exp_src"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --out "$OUT" --overlay "$OVERLAY" --batch-size 48

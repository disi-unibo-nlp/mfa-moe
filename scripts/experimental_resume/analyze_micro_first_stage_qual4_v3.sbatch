#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --job-name=st-micro-first-qual4-v3
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1
DRIVER="$REPO/scripts/experimental_resume/analyze_micro_first_stage.py"
MANIFEST="$REPO/report/experimental-resume-v1/CAUSAL_MICROSCREEN_QUAL4_MANIFEST_v3.json"
RUN_OUT="$ROOT/micro-screen-qual4-ac4c9651e71fe067"
OUT="$ROOT/micro-screen-first-stage-qual4-ac4c9651e71fe067"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == 93c53297b6d12ffbaab7b4b6613365b1f4d7c14becdea6c5f2e81e34caddd321 ]] || exit 2
[[ -s "$RUN_OUT/SUMMARY.json" && -f "$MANIFEST" ]] || exit 2
module load python/3.11.7
umask 077
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER" --manifest "$MANIFEST" --run-out "$RUN_OUT" --out "$OUT" --batch-size 48

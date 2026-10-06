#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=overnight-adjudicated-price-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* &&
   "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
: "${OVERNIGHT_MANIFEST:?sealed A/B generation manifest required}"
: "${OVERNIGHT_GENERATION_PRICE:?complete generation price required}"
: "${OVERNIGHT_GENERATION_OUT:?generation output root required}"
: "${OVERNIGHT_MEASUREMENT_OUT:?blind frame and price output root required}"
: "${OVERNIGHT_READER_GPU_HOUR_CEILING:?reviewed complete reader envelope required}"
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
module load python/3.11.7
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-overnight-adjudicated-price-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VLLM_BATCH_INVARIANT=0 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume:$ROOT/code/s1-9a61e32f48c04c24"
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/adjudicate_overnight_ab_seals_v1.py" \
  --manifest "$OVERNIGHT_MANIFEST" --price "$OVERNIGHT_GENERATION_PRICE" --out "$OVERNIGHT_GENERATION_OUT"
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/build_overnight_blind_frame_v1.py" \
  --manifest "$OVERNIGHT_MANIFEST" --generation-price "$OVERNIGHT_GENERATION_PRICE" \
  --run-out "$OVERNIGHT_GENERATION_OUT" --out "$OVERNIGHT_MEASUREMENT_OUT"
srun --ntasks=1 "$PY" -B -u "$REPO/scripts/experimental_resume/price_overnight_semantics_v1.py" \
  --frame "$OVERNIGHT_MEASUREMENT_OUT/BLIND_FRAME.json" --out "$OVERNIGHT_MEASUREMENT_OUT/READER_PRICE.json" \
  --max-wall-seconds 7200 --gpu-hour-ceiling "$OVERNIGHT_READER_GPU_HOUR_CEILING"
"$PY" -B -c 'import sys; import run_boundary_micro_screen as b; import rate_overnight_semantics_v1 as r; f,p=map(b.sealed,sys.argv[1:]); r.validate(f,p); print("PASS_COMPLETE_READER_PRICE",p["ratings"],len(p["shards"]),p["estimated_complete_gpu_hours"])' \
  "$OVERNIGHT_MEASUREMENT_OUT/BLIND_FRAME.json" "$OVERNIGHT_MEASUREMENT_OUT/READER_PRICE.json"

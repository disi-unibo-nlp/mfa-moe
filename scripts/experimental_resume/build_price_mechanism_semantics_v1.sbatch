#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=st-mechanism-semantic-price
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett &&
   "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
MANIFEST="$REPO/report/experimental-resume-v1/MECHANISM_VALIDATION_MANIFEST_v1.json"
GEN_PRICE="$REPO/report/experimental-resume-v1/MECHANISM_VALIDATION_PRICE_v1.json"
FRAME_BUILDER="$REPO/scripts/experimental_resume/build_mechanism_blind_frame_v1.py"
READER_PRICE_DRIVER="$REPO/scripts/experimental_resume/price_mechanism_semantics_v1.py"
[[ -f "$MANIFEST" && -f "$GEN_PRICE" && -f "$FRAME_BUILDER" &&
   -f "$READER_PRICE_DRIVER" ]] || exit 2
module load python/3.11.7
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-mechanism-semantic-price-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/tmp" "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/xdg"
export TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume:$ROOT/code/s1-9a61e32f48c04c24"
MANIFEST_SHA="$("$PY" -B -c 'import sys; from run_boundary_micro_screen import sealed; m=sealed(sys.argv[1]); p=sealed(sys.argv[2]); assert m["schema"]=="mechanism-validation-manifest-v1" and p["status"]=="PASS_COMPLETE_STAGE" and p["manifest_sha256"]==m["sha256"]; print(m["sha256"])' "$MANIFEST" "$GEN_PRICE")"
GEN_OUT="$ROOT/runs/routing-control-v1/mechanism-validation-v1-${MANIFEST_SHA:0:16}"
OUT="$ROOT/runs/routing-control-v1/mechanism-1024-semantics-v1-${MANIFEST_SHA:0:16}"
[[ -f "$GEN_OUT/STAGE_COMPLETION.json" ]] || exit 2
srun --ntasks=1 "$PY" -B -u "$FRAME_BUILDER" \
  --manifest "$MANIFEST" --generation-price "$GEN_PRICE" \
  --run-out "$GEN_OUT" --out "$OUT"
[[ -f "$OUT/ARM_MAP.json" && -f "$OUT/BLIND_FRAME.json" ]] || exit 2
srun --ntasks=1 "$PY" -B -u "$READER_PRICE_DRIVER" \
  --frame "$OUT/BLIND_FRAME.json" --out "$OUT/READER_PRICE.json" \
  --max-wall-seconds 7200 --gpu-hour-ceiling 100
"$PY" -B -c 'import sys; from run_boundary_micro_screen import sealed; f=sealed(sys.argv[1]); p=sealed(sys.argv[2]); assert p["status"]=="PASS_COMPLETE_STAGE" and p["frame_sha256"]==f["sha256"]; print("mechanism reader price:",p["estimated_complete_gpu_hours"],"GPU-h across",len(p["shards"]),"shards")' "$OUT/BLIND_FRAME.json" "$OUT/READER_PRICE.json"

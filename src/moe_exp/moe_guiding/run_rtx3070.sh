#!/usr/bin/env bash
# Run one condition with a small native top-2 model on an 8 GiB CUDA GPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

CONDITION="${1:-selected}"
if [[ $# -gt 0 ]]; then
  shift
fi
case "$CONDITION" in
  baseline|selected|transfer) ;;
  *)
    echo "Usage: bash $0 [baseline|selected|transfer] [generate options...]" >&2
    exit 2
    ;;
esac

exec bash "$SCRIPT_DIR/run_docker.sh" generate \
  --model Isotonic/TinyMixtral-4x248M-MoE \
  --revision 1e3516176a6279ce93923060fcad848dc043af79 \
  --prompts src/moe_exp/moe_guiding/prompts.jsonl \
  --condition "$CONDITION" \
  --threshold 0.4 \
  --layers 6 \
  --chat \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.8 \
  --max-tokens 256 \
  --seed 42 \
  --output-dir "results/moe_guiding/rtx3070/$CONDITION" \
  "$@"

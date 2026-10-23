#!/bin/bash
# Nemotron-3.5-Lightning-30B-A3B (NVFP4) with its DSpark draft model.
source scripts/serve/args.sh
exec vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --revision bee7596271d1495f6992ae224aefde4410e816b8 $SERVE_ARGS \
  --reasoning-parser nemotron_v3 --attention-config '{"backend":"TRITON_ATTN"}' --max-model-len 133120 --max-num-seqs 16 \
  --linear-backend marlin --mamba-backend triton --mamba-cache-mode align \
  --speculative-config '{"method":"dspark","model":"nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark","revision":"8a0177116d138011e63103110f136ec0ca09ebbf","num_speculative_tokens":3}'

#!/bin/bash
# Qwen3.6-35B-A3B (FP8) with 3-token MTP speculation.
source scripts/serve/args.sh
exec vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --revision 95a723d08a9490559dae23d0cff1d9466213d989 $SERVE_ARGS \
  --reasoning-parser qwen3 --attention-config '{"backend":"FLASH_ATTN"}' --language-model-only \
  --max-model-len 133120 --max-num-seqs 16 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

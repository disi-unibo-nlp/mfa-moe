#!/bin/bash
# Qwen3.5-35B-A3B (GPTQ int4, local copy) with 3-token MTP speculation.
source scripts/serve/args.sh
exec vllm serve $MFA/cache/native-replay/Qwen3.5-35B-A3B-GPTQ-Int4/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b \
  --served-model-name Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 $SERVE_ARGS --reasoning-parser qwen3 \
  --attention-config '{"backend":"FLASH_ATTN"}' --language-model-only --max-model-len 133120 --max-num-seqs 16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

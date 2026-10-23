#!/bin/bash
# Qwen3.8-27B, the judge that labels sentences and checks answers (49,152 tokens of context), with 3-token MTP speculation.
source scripts/serve/args.sh
exec vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 $SERVE_ARGS \
  --reasoning-parser qwen3 --attention-config '{"backend":"FLASH_ATTN"}' --language-model-only --enforce-eager \
  --max-model-len 49152 --max-num-seqs 64 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

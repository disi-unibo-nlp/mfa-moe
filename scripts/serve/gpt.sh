#!/bin/bash
# GPT-OSS-20B. Sampling (T 1, top-p 1, medium reasoning) is set by the client.
source scripts/serve/args.sh
exec vllm serve openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee $SERVE_ARGS \
  --reasoning-parser openai_gptoss --attention-config '{"backend":"TRITON_ATTN"}' --max-model-len 131072 --max-num-seqs 16

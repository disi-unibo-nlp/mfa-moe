#!/bin/bash
# Qwen3-30B-A3B. Its native context is 40,960 tokens (no YaRN).
source scripts/serve/args.sh
exec vllm serve Qwen/Qwen3-30B-A3B --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 $SERVE_ARGS \
  --reasoning-parser qwen3 --attention-config '{"backend":"FLASH_ATTN"}' --max-model-len 40960 --max-num-seqs 16

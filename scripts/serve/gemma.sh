#!/bin/bash
# Gemma-4-26B-A4B (NVFP4, local copy). Thinking is switched on by the client.
source scripts/serve/args.sh
exec vllm serve $MFA/cache/native-replay/Gemma-4-26B-A4B-NVFP4/a19cfe00be84568a6867111c9a68c9c44fdcffe6 \
  --served-model-name nvidia/Gemma-4-26B-A4B-NVFP4 $SERVE_ARGS --reasoning-parser gemma4 --language-model-only \
  --max-model-len 133120 --max-num-seqs 8 --linear-backend marlin

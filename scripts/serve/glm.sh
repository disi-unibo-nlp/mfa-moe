#!/bin/bash
# GLM-4.7-Flash.
source scripts/serve/args.sh
exec vllm serve zai-org/GLM-4.7-Flash --revision 7dd20894a642a0aa287e9827cb1a1f7f91386b67 $SERVE_ARGS \
  --reasoning-parser glm45 --attention-config '{"backend":"TRITON_MLA"}' --max-model-len 133120 --max-num-seqs 8

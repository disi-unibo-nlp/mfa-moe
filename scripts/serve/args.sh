# Flags every model server shares: 2 GPUs, bf16 weights and KV cache (A100 has no fp8 KV), local only.
SERVE_ARGS="--host 127.0.0.1 --port $PORT --api-key local-vllm-key --tensor-parallel-size 2
  --dtype bfloat16 --kv-cache-dtype bfloat16 --generation-config vllm --enable-prefix-caching
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90"

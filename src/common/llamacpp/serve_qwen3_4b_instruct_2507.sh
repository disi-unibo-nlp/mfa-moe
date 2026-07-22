#!/usr/bin/env bash
set -euo pipefail

# Non-MTP launch: Qwen3-4B-Instruct-2507 is the plain instruct/non-thinking 4B model.
IMAGE="${IMAGE:-llama.cpp:localcuda}"
HF_MODEL="${HF_MODEL:-bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
GPU_DEVICE="${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
LLAMA_API_KEY="${LLAMA_API_KEY:-local-llamacpp-key}"

CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"

mkdir -p "${HF_CACHE_DIR}"

docker run --rm \
  --gpus "device=${GPU_DEVICE}" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -p "${HOST}:${PORT}:8080" \
  "${IMAGE}" \
  -hf "${HF_MODEL}" \
  --api-key "${LLAMA_API_KEY}" \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers "${N_GPU_LAYERS}" \
  --ctx-size "${CTX_SIZE}" \
  --parallel "${PARALLEL}" \
  --flash-attn "${FLASH_ATTN}" \
  --batch-size "${BATCH_SIZE}"

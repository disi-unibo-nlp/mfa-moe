#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5's MTP weights are embedded in this GGUF. llama.cpp uses them as the
# speculative drafter while the target layers verify every proposed token.
IMAGE="${IMAGE:-llama.cpp:localcuda}"
CONTAINER_NAME="${CONTAINER_NAME:-correlation-qwen35-mtp}"
MODEL_DIR="${MODEL_DIR:-/llms}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf}"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-35B-A3B-MTP-GGUF}"
MODEL_REVISION="${MODEL_REVISION:-main}"
MODEL_ALIASES="${MODEL_ALIASES:-qwen3.5-35b-a3b-mtp-ud-q4-k-xl,local-llamacpp}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
GPU_DEVICE="${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
LLAMA_API_KEY="${LLAMA_API_KEY:-local-llamacpp-key}"

CTX_SIZE="${CTX_SIZE:-49152}"
PARALLEL="${PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
N_GPU_LAYERS="${N_GPU_LAYERS:-all}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-6}"
FLASH_ATTN="${FLASH_ATTN:-on}"
DRY_RUN="${DRY_RUN:-false}"
DOWNLOAD_ONLY="${DOWNLOAD_ONLY:-false}"

download_gguf() {
  local repo="$1"
  local revision="$2"
  local filename="$3"
  local target="${MODEL_DIR}/${filename}"
  local partial="${target}.part"
  local url="https://huggingface.co/${repo}/resolve/${revision}/${filename}?download=true"
  local -a curl_args=(
    --fail
    --location
    --retry 5
    --retry-delay 5
    --continue-at -
    --output "${partial}"
  )

  if [[ -s "${target}" ]]; then
    echo "Model ready: ${target}"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required to download ${filename}." >&2
    return 1
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    curl_args+=(--header "Authorization: Bearer ${HF_TOKEN}")
  fi

  echo "Model missing; downloading into the shared model folder:"
  echo "  Repository: ${repo}"
  echo "  File:       ${filename}"
  echo "  Target:     ${target}"
  if ! curl "${curl_args[@]}" "${url}"; then
    echo "ERROR: download failed; resumable partial retained at ${partial}" >&2
    return 1
  fi
  if [[ ! -s "${partial}" ]]; then
    echo "ERROR: download produced an empty file: ${partial}" >&2
    return 1
  fi
  mv -f -- "${partial}" "${target}"
  echo "Download complete: ${target}"
}

mkdir -p "${MODEL_DIR}" "${HF_CACHE_DIR}"
if [[ ! -w "${MODEL_DIR}" ]]; then
  echo "ERROR: shared model folder is not writable: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ "${PARALLEL}" != "1" ]]; then
  echo "ERROR: llama.cpp native MTP currently requires PARALLEL=1." >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  download_gguf "${MODEL_REPO}" "${MODEL_REVISION}" "${MODEL_NAME}"
  if [[ "${DOWNLOAD_ONLY}" == "true" ]]; then
    exit 0
  fi
fi

command=(
  docker run --rm
  --name "${CONTAINER_NAME}"
  --gpus "device=${GPU_DEVICE}"
  -v "${MODEL_DIR}:/models:ro"
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
  -p "${HOST}:${PORT}:8080"
  "${IMAGE}"
  -m "/models/${MODEL_NAME}"
  --alias "${MODEL_ALIASES}"
  --spec-type draft-mtp
  --spec-draft-n-max "${SPEC_DRAFT_N_MAX}"
  --api-key "${LLAMA_API_KEY}"
  --host 0.0.0.0
  --port 8080
  --n-gpu-layers "${N_GPU_LAYERS}"
  --ctx-size "${CTX_SIZE}"
  --parallel "${PARALLEL}"
  --flash-attn "${FLASH_ATTN}"
  --batch-size "${BATCH_SIZE}"
)

if [[ "${DRY_RUN}" == "true" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

help_output="$(docker run --rm "${IMAGE}" --help 2>&1)"
if [[ "${help_output}" != *draft-mtp* ]]; then
  echo "ERROR: ${IMAGE} does not expose llama.cpp draft-mtp support." >&2
  echo "Rebuild src/common/llamacpp/Dockerfile from a current llama.cpp revision." >&2
  exit 1
fi

exec "${command[@]}"

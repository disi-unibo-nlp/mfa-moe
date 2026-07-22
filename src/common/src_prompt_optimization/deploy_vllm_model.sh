#!/usr/bin/env bash
set -euo pipefail

# Deploy a model through vLLM's OpenAI-compatible API.
# Override any variable from the shell or define it in src_prompt_optimization/.env.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
GPU_ID="${GPU_ID:-0}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-38000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
PID_DIR="${PID_DIR:-$SCRIPT_DIR/pids}"
WAIT_FOR_READY="${WAIT_FOR_READY:-true}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-900}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

mkdir -p "$LOG_DIR" "$PID_DIR"

LOG_FILE="$LOG_DIR/vllm_${SERVER_PORT}.log"
PID_FILE="$PID_DIR/vllm_${SERVER_PORT}.pid"
HEALTH_URL="http://127.0.0.1:${SERVER_PORT}/v1/models"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "A vLLM server is already running for port ${SERVER_PORT} with PID ${OLD_PID}."
    echo "Health URL: ${HEALTH_URL}"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$SERVER_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${SERVER_PORT} is already in use. Stop that process or set SERVER_PORT to a free port."
  exit 1
fi

CMD=(
  vllm serve "$MODEL_NAME"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --dtype "$DTYPE"
)

if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ -n "$DOWNLOAD_DIR" ]]; then
  CMD+=(--download-dir "$DOWNLOAD_DIR")
fi

if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
  # Intentionally split EXTRA_VLLM_ARGS so callers can pass normal CLI flags.
  # Example: EXTRA_VLLM_ARGS="--enable-prefix-caching --max-num-seqs 32"
  read -r -a EXTRA_ARGS <<< "$EXTRA_VLLM_ARGS"
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Starting vLLM server"
echo "  Model: ${MODEL_NAME}"
echo "  Served model name: ${SERVED_MODEL_NAME}"
echo "  GPU(s): ${GPU_ID}"
echo "  URL: ${HEALTH_URL}"
echo "  Log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="$GPU_ID" "${CMD[@]}" >"$LOG_FILE" 2>&1 &
SERVER_PID="$!"
echo "$SERVER_PID" > "$PID_FILE"

echo "Started vLLM PID ${SERVER_PID}"

if [[ "$WAIT_FOR_READY" != "true" ]]; then
  exit 0
fi

echo "Waiting for vLLM readiness..."
START_SECONDS="$(date +%s)"
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Last log lines:"
    tail -80 "$LOG_FILE" || true
    rm -f "$PID_FILE"
    exit 1
  fi

  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "vLLM is ready at ${HEALTH_URL}"
    exit 0
  fi

  NOW_SECONDS="$(date +%s)"
  if (( NOW_SECONDS - START_SECONDS > READY_TIMEOUT_SECONDS )); then
    echo "Timed out after ${READY_TIMEOUT_SECONDS}s waiting for vLLM readiness."
    echo "Last log lines:"
    tail -80 "$LOG_FILE" || true
    exit 1
  fi

  sleep 5
done

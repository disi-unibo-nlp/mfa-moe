#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

SERVER_PORT="${SERVER_PORT:-8000}"
PID_DIR="${PID_DIR:-$SCRIPT_DIR/pids}"
PID_FILE="$PID_DIR/vllm_${SERVER_PORT}.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found for port ${SERVER_PORT}: ${PID_FILE}"
  exit 0
fi

SERVER_PID="$(cat "$PID_FILE")"
if [[ -z "$SERVER_PID" ]]; then
  rm -f "$PID_FILE"
  echo "Removed empty PID file."
  exit 0
fi

if kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "Stopping vLLM PID ${SERVER_PID} on port ${SERVER_PORT}..."
  kill "$SERVER_PID"
  for _ in $(seq 1 30); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Stopped."
      exit 0
    fi
    sleep 1
  done
  echo "PID ${SERVER_PID} did not stop after 30s; sending SIGKILL."
  kill -9 "$SERVER_PID" 2>/dev/null || true
fi

rm -f "$PID_FILE"
echo "Stopped."

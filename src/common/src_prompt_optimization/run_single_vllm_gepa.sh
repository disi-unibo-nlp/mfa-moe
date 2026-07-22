#!/usr/bin/env bash
set -euo pipefail

# Standalone GEPA run using one OpenAI-compatible vLLM server for generation,
# judging, and GEPA reflection. Override any variable from the shell.

MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"

DATASET_NAME="${DATASET_NAME:-disi-unibo-nlp/eurlex_relations}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-200}"
VAL_SAMPLES="${VAL_SAMPLES:-80}"

NUM_THREADS="${NUM_THREADS:-8}"
MAX_FULL_EVALS="${MAX_FULL_EVALS:-25}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gepa_optimization_single_model}"
LOG_DIR="${LOG_DIR:-logs/gepa_optimization}"

TEMPERATURE="${TEMPERATURE:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.3}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-32000}"

WANDB_ARGS=()
if [[ "${USE_WANDB:-false}" == "true" ]]; then
  WANDB_ARGS+=(--use_wandb --wandb_project "${WANDB_PROJECT:-gepa-legal-qa}")
  if [[ -n "${WANDB_NAME:-}" ]]; then
    WANDB_ARGS+=(--wandb_name "$WANDB_NAME")
  fi
fi

python3 "$(dirname "$0")/run_gepa_qa_optimization.py" \
  --dataset_name "$DATASET_NAME" \
  --dataset_split "$DATASET_SPLIT" \
  --train_samples "$TRAIN_SAMPLES" \
  --val_samples "$VAL_SAMPLES" \
  --task_model "$MODEL_NAME" \
  --eval_model "$MODEL_NAME" \
  --reflection_model "$MODEL_NAME" \
  --vllm_url "$SERVER_URL" \
  --task_port "$SERVER_PORT" \
  --eval_port "$SERVER_PORT" \
  --num_threads "$NUM_THREADS" \
  --temperature "$TEMPERATURE" \
  --max_tokens "$MAX_TOKENS" \
  --eval_temperature "$EVAL_TEMPERATURE" \
  --eval_max_tokens "$EVAL_MAX_TOKENS" \
  --max_full_evals "$MAX_FULL_EVALS" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  "${WANDB_ARGS[@]}"

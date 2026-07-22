#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-data/Schoenfeld_Reasoning}"
MODEL="${MODEL:-local-llamacpp}"
API_BASE="${API_BASE:-http://127.0.0.1:8080/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-results/exp0a}"

python -m moe_exp.experiment0a.run \
  --dataset-dir "${DATASET_DIR}" \
  --model "${MODEL}" \
  --api-base "${API_BASE}" \
  --output-dir "${OUTPUT_DIR}" \
  --prompt-variant "${PROMPT_VARIANT:-base}" \
  --few-shot-examples "${FEW_SHOT_EXAMPLES:-3}" \
  --few-shot-units "${FEW_SHOT_UNITS:-8}" \
  --gepa-auto "${GEPA_AUTO:-light}" \
  --num-threads "${NUM_THREADS:-3}"

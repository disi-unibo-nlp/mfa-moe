#!/bin/bash
# Optimise the judge prompt with GEPA against a running OpenAI-compatible server of Qwen3.8-27B, for example
# llama-server --jinja --flash-attn on --ctx-size 8192 --api-key local-llamacpp-key (the key the optimiser sends by default):
#   scripts/labeling/optimize_judge_prompt.sh SERVER_URL DATASET_DIR SEED_PROMPT OUT_DIR
#   e.g.  scripts/labeling/optimize_judge_prompt.sh http://127.0.0.1:8080/v1 data/Schoenfeld_Reasoning \
#           results/exp0a/context-llmjudge-20k-s42/gepa_optimized_prompt_20260811_135905.txt results/gepaLLMAsJudge/new-run
# DATASET_DIR holds the labelled Schoenfeld responses, SEED_PROMPT is the prompt file GEPA starts from. Every other
# setting is the default (seed 42, 21-example few-shot prompt, LLM-judge reward, balanced-accuracy selection, medium
# reasoning effort, split 26/6/6).
: "${1:?SERVER_URL}" "${2:?DATASET_DIR}" "${3:?SEED_PROMPT}" "${4:?OUT_DIR}"
test -d "$2" || { echo "no such folder: $2" >&2; exit 1; }
test -f "$3" || { echo "no such file: $3" >&2; exit 1; }
MFA=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
source scripts/setup/judge_client_env_path.sh  # sets DSPY, the judge client environment
PYTHONPATH=src:$DSPY/lib/python3.11/site-packages:$MFA/envs/vllm-cu129/lib/python3.11/site-packages \
$DSPY/bin/python -m labeling.gepa.optimize_judge --dataset-dir "$2" --api-base "$1" --output-dir "$4" \
  --gepa-auto heavy --enable-thinking --reflection-max-tokens 4096 --seed-prompt-file "$3"

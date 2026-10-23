#!/bin/bash
# Optimise the judge prompt with GEPA against a running OpenAI-compatible server of Qwen3.8-27B (it ran on a
# workstation with llama.cpp: llama-server --jinja --flash-attn on --ctx-size 8192 --api-key local-llamacpp-key):
#   scripts/labeling/optimize_judge.sh API_BASE OUTPUT_DIR     e.g. http://127.0.0.1:8080/v1 results/gepaLLMAsJudge/new-run
M=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
DSPY=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-88189c22f214ed65
PYTHONPATH=src:$DSPY/lib/python3.11/site-packages:$M/envs/vllm-cu129/lib/python3.11/site-packages \
$DSPY/bin/python -B -m moe_exp.labeling.gepa.run --dataset-dir data/Schoenfeld_Reasoning \
  --api-base "$1" --api-key local-llamacpp-key --model local-llamacpp --seed 42 --prompt-variant few-shot \
  --few-shot-examples 21 --gepa-reward llm-judge --selection-metric balanced_accuracy --gepa-auto heavy \
  --seed-prompt-file results/exp0a/context-llmjudge-20k-s42/gepa_optimized_prompt_20260811_135905.txt \
  --enable-thinking --reasoning-effort medium --reflection-max-tokens 4096 --output-dir "$2"

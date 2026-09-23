#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
TEMPLATE_ARGS=()
if [[ -n "${TEMPLATE_DATE:-}" ]]; then
    TEMPLATE_ARGS=(--template-date "$TEMPLATE_DATE")
fi
STAGE="${1:-all}"
ATTEMPTS="${2:-full}"
case "$ATTEMPTS" in single|full) ;; *) echo 'Choose single or full attempts' >&2; exit 2;; esac
case "$STAGE" in prepare|fit|generate|compare|all) ;; *) echo 'Usage: run_global.sh [prepare|fit|generate|compare|all] [single|full]' >&2; exit 2;; esac
MODEL_PROFILE="${MODEL_PROFILE:-qwen}"
case "$MODEL_PROFILE" in
    qwen)
        DEFAULT_MODEL="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
        DEFAULT_GENERATION_ROOT="results/correlation_pipeline/generation/Qwen--Qwen3.5-35B-A3B-GPTQ-Int4"
        DEFAULT_ROUTING_ROOT="results/correlation_pipeline/reasoning-vllm-v1/forward/unsloth--Qwen3.5-35B-A3B"
        ;;
    oss)
        DEFAULT_MODEL="openai/gpt-oss-20b"
        DEFAULT_GENERATION_ROOT="results/correlation_pipeline/gpt-oss-20b/generation/openai--gpt-oss-20b"
        DEFAULT_ROUTING_ROOT="results/correlation_pipeline/gpt-oss-20b/reasoning-vllm-v1/forward/openai--gpt-oss-20b"
        ;;
    gemma)
        DEFAULT_MODEL="nvidia/Gemma-4-26B-A4B-NVFP4"
        DEFAULT_GENERATION_ROOT="results/correlation_pipeline/gemma-nvfp4-nf4/generation/nvidia--Gemma-4-26B-A4B-NVFP4"
        DEFAULT_ROUTING_ROOT="results/correlation_pipeline/gemma-nvfp4-nf4/reasoning-vllm-v1/forward/google--gemma-4-26B-A4B-it"
        ;;
    *) echo 'MODEL_PROFILE must be qwen, oss or gemma' >&2; exit 2 ;;
esac
MODEL="${MODEL:-$DEFAULT_MODEL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/moe_identity_guiding/${MODEL_PROFILE}_global}"
EXPERT_POLARITY="${EXPERT_POLARITY:-positive}"
GUIDING_METHOD="${GUIDING_METHOD:-fixed}"
PAPER_EPSILON="${PAPER_EPSILON:-0.01}"
case "$EXPERT_POLARITY" in positive|negative) ;; *) echo 'Invalid EXPERT_POLARITY' >&2; exit 2;; esac
case "$GUIDING_METHOD" in fixed|paper) ;; *) echo 'Invalid GUIDING_METHOD' >&2; exit 2;; esac
# Isolate new conditions even when OUTPUT_ROOT is explicitly supplied.
if [[ "$EXPERT_POLARITY" != positive || "$GUIDING_METHOD" != fixed ]]; then
    OUTPUT_ROOT="$OUTPUT_ROOT/variants/${EXPERT_POLARITY}_${GUIDING_METHOD}"
    if [[ "$GUIDING_METHOD" == paper ]]; then
        if [[ ! "$PAPER_EPSILON" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
            echo 'PAPER_EPSILON must be a positive decimal number' >&2; exit 2
        fi
        OUTPUT_ROOT="${OUTPUT_ROOT}_eps_${PAPER_EPSILON}"
    fi
    OUTPUT_ROOT="$OUTPUT_ROOT/strength_${STRENGTH:-1}"
fi
GENERATION_ROOT="${GENERATION_ROOT:-$DEFAULT_GENERATION_ROOT}"
ROUTING_ROOT="${ROUTING_ROOT:-$DEFAULT_ROUTING_ROOT}"
RUNNER=src/moe_exp/moe_identity_guiding/run_docker.sh
DATASETS=(aime24 aime25 amc23 math500 minerva olympiad)
GENERATIONS=()
TRACES=()
for dataset in "${DATASETS[@]}"; do
    GENERATIONS+=("$GENERATION_ROOT/$dataset/traces.jsonl")
    TRACES+=("$ROUTING_ROOT/$dataset/traces_with_routing.jsonl")
done
if [[ "$STAGE" == prepare || ( "$STAGE" == all && ! -f "$OUTPUT_ROOT/split/split.json" ) ]]; then
    bash "$RUNNER" prepare-global --generations "${GENERATIONS[@]}" --traces "${TRACES[@]}" \
        --calibration-fraction "${CALIBRATION_FRACTION:-0.7}" --seed 42 \
        --output-dir "$OUTPUT_ROOT/split"
fi
if [[ "$STAGE" == fit || ( "$STAGE" == all && ! -f "$OUTPUT_ROOT/policy.json" ) ]]; then
    bash "$RUNNER" fit --model "$MODEL" --traces "$OUTPUT_ROOT/split/calibration.jsonl" \
        --expert-polarity "$EXPERT_POLARITY" --guiding-method "$GUIDING_METHOD" \
        --paper-epsilon "$PAPER_EPSILON" --max-experts "${MAX_EXPERTS:-8}" \
        --min-support "${MIN_SUPPORT:-4}" --output "$OUTPUT_ROOT/policy.json"
fi
if [[ "$STAGE" == generate || "$STAGE" == all ]]; then
    bash "$RUNNER" prepare-sampling --prompts "$OUTPUT_ROOT/split/prompts.$ATTEMPTS.jsonl" \
        --generations "${GENERATIONS[@]}" --output "$OUTPUT_ROOT/split/prompts.$ATTEMPTS.sampling.jsonl"
    for condition in baseline guided; do
        bash "$RUNNER" generate --model "$MODEL" --policy "$OUTPUT_ROOT/policy.json" \
            --prompts "$OUTPUT_ROOT/split/prompts.$ATTEMPTS.sampling.jsonl" \
            "${TEMPLATE_ARGS[@]}" \
            --resume --diagnostics "${DIAGNOSTICS:-minimal}" \
            --condition "$condition" --strength "${STRENGTH:-1}" \
            --max-num-seqs "${MAX_NUM_SEQS:-16}" \
            --temperature "${TEMPERATURE:-0.6}" --top-p "${TOP_P:-0.95}" --top-k 0 --seed 0 \
            --require-original-sampling \
            --max-tokens 32768 --max-model-len 49152 \
            --output-dir "$OUTPUT_ROOT/sampling/$ATTEMPTS/$condition"
    done
fi
if [[ "$STAGE" == compare || "$STAGE" == all ]]; then
    bash "$RUNNER" compare --baseline "$OUTPUT_ROOT/sampling/$ATTEMPTS/baseline" \
        --guided "$OUTPUT_ROOT/sampling/$ATTEMPTS/guided"
fi

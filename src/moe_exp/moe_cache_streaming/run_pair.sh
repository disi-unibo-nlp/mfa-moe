#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
RUNNER=src/moe_exp/moe_cache_streaming/run_docker.sh
OUTPUT_ROOT="${OUTPUT_ROOT:-results/moe_cache_streaming/oss_$(date -u +%Y%m%dT%H%M%SZ)}"
POLICY="${POLICY:-results/moe_identity_guiding/oss_strength_1/policy.json}"
PROMPTS="${PROMPTS:-results/moe_identity_guiding/oss_strength_1/split/prompts.full.sampling.jsonl}"
if [[ "${DRY_RUN:-false}" != true ]]; then
    mkdir -p "$OUTPUT_ROOT"
fi
for condition in baseline guided; do
    command=(bash "$RUNNER" generate --policy "$POLICY" --prompts "$PROMPTS" \
        --model "${MODEL:-openai/gpt-oss-20b}" --condition "$condition" \
        --strength "${STRENGTH:-1}" --limit "${LIMIT:-24}" \
        --prefill-chunk-size "${PREFILL_CHUNK_SIZE:-2048}" \
        --max-tokens "${MAX_TOKENS:-32768}" --max-model-len "${MAX_MODEL_LEN:-49152}" \
        --template-date "${TEMPLATE_DATE:-2026-09-22}" \
        --output "$OUTPUT_ROOT/$condition")
    if [[ "${DRY_RUN:-false}" == true ]]; then
        "${command[@]}"
    else
        "${command[@]}" 2>&1 | tee "$OUTPUT_ROOT/$condition.log"
    fi
done
read -r -a CACHE_BUDGETS <<< "${BUDGETS:-8 16 32}"
bash "$RUNNER" analyze --baseline "$OUTPUT_ROOT/baseline" \
    --guided "$OUTPUT_ROOT/guided" --budgets "${CACHE_BUDGETS[@]}" \
    --output "$OUTPUT_ROOT/analysis"

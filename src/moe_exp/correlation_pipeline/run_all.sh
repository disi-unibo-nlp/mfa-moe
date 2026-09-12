#!/usr/bin/env bash
set -euo pipefail

# vLLM generation -> frozen GEPA tagging -> Unsloth forward -> all analyses.
PHYS_DIR="${PHYS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/llms}"
STAGE_LAUNCHER="${STAGE_LAUNCHER:-$PHYS_DIR/src/moe_exp/correlation_pipeline/run_docker.sh}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-41800}"
JUDGE_PORT="${JUDGE_PORT:-41800}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
API_KEY="${API_KEY:-local-vllm-key}"
WORKERS="${WORKERS:-8}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
# Preserve the 32768-token completion budget plus room for the prompt.
CTX_SIZE="${CTX_SIZE:-}"
JUDGE_CTX_SIZE="${JUDGE_CTX_SIZE:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
SPECULATIVE_TOKENS="${SPECULATIVE_TOKENS:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
QUANTIZATION="${QUANTIZATION:-}"
GENERATION_QUANTIZATION="${GENERATION_QUANTIZATION:-}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GENERATION_SPECULATION="auto"
ALL_ROUTER_LAYERS=false
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-500}"
DRY_RUN="${DRY_RUN:-false}"
RESULTS_DIR="${RESULTS_DIR:-results/correlation_pipeline}"
GENERATION_DIR="${GENERATION_DIR:-}"
PIPELINE_MODEL=""
GENERATION_MODEL="${GENERATION_MODEL:-}"
JUDGE_PROGRAM="${JUDGE_PROGRAM:-results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json}"
JUDGE_MODEL="${JUDGE_MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
POSITION_BINS=10
DATASETS=(math500 aime24 aime25 olympiad amc23 minerva)
MAX_ITEMS=""
SAMPLES_PER_PROBLEM=""
LIMIT=""
SKIP_GENERATE=false
SKIP_ANNOTATE=false
SKIP_TAGGING=false
SKIP_FORWARD=false
SKIP_ANALYZE=false
SKIP_SAMPLING=false
MAX_SENTENCES=100000

usage() {
    cat <<'HELP'
Usage: bash src/moe_exp/correlation_pipeline/run_all.sh [options]

Runs vLLM generation, frozen GEPA tagging, forward replay, and all analyses.
Each server uses prefix caching, eight sequences and batch-token budget 8192.
Generation uses model-specific reasoning and MTP settings; the judge is unchanged.
New generations are identified by their checkpoint; reasoning-vllm-v1 keeps
vLLM annotations/extraction/analysis separate from existing GGUF results.

  --datasets NAME...         Default: six benchmarks; GPQA-Diamond/MMLU-Pro opt-in
  --workers N                Concurrent generation requests (default: 8)
  --judge-workers N          Concurrent sentence requests (default: 8)
  --model NAME               Generation and forward/analysis checkpoint; leaves judge unchanged
  --generation-model NAME    Override generation only (also accepts saved alias with --skip-generate)
  --max-tokens N             Completion budget (default: 32768)
  --ctx-size N               Generation context (default: 49152; Qwen3: 40960)
  --generation-speculation MODE  auto, mtp, or none (default: model profile)
  --generation-quantization MODE  Optional vLLM generation quantization override
  --cpu-offload-gb N         Generation weight offload per GPU (default: 0)
  --tensor-parallel-size N   Generation GPU count (default: 1)
  --results-dir DIR          Repo-relative result root (default: results/correlation_pipeline)
  --generation-dir DIR       Generation input/output (default: RESULTS_DIR/generation)
  --max-items N              Limit generated problems per dataset (smoke test)
  --samples-per-problem N    Override benchmark sampling counts (smoke test)
  --limit N                  Limit saved traces per dataset for tagging/forward/analysis
  --max-sentences N          Error-rate-weighted sentence cap across benchmarks (default: 100000)
  --skip-sampling            Use generation inputs directly (for already sampled inputs)
  --judge-program FILE       Repo-relative frozen GEPA selected_program JSON
  --judge-model NAME         vLLM judge checkpoint (default: unsloth/Qwen3.8-27B-NVFP4)
  --position-bins N          Fixed mean-length bins (default: 10)
  --quantization MODE        Forward quantization (default: model profile)
  --all-router-layers        Explicitly override fixed probe selection with every MoE layer
  --bootstrap-samples N      Analysis cluster bootstraps (default: 500)
  --skip-generate            Reuse existing generations of --generation-model
  --skip-annotate            Reuse completed matching sentence annotations; keep class views
  --skip-tagging             Run without annotations or class views; keep full/position views
  --skip-forward             Reuse extraction checkpoints with the selected reasoning views
  --skip-analyze             Skip the analysis stage
  --dry-run                  Print commands without Docker calls or output changes
  -h, --help                 Show this help

Environment overrides include VLLM_IMAGE, SERVER_HOST, SERVER_PORT, JUDGE_PORT,
SERVER_READY_TIMEOUT, CUDA_VISIBLE_DEVICES, HF_CACHE_DIR, JUDGE_CTX_SIZE,
MAX_NUM_BATCHED_TOKENS, SPECULATIVE_TOKENS and GPU_MEMORY_UTILIZATION.

Without --model, generation defaults to Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 and
forward/analysis to unsloth/Qwen3.5-35B-A3B. --generation-model (or
GENERATION_MODEL) overrides the generation checkpoint even with --model,
regardless of option order; use matching checkpoints of the same base model.
HELP
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            shift; DATASETS=()
            while [[ $# -gt 0 && "$1" != -* ]]; do DATASETS+=("$1"); shift; done
            [[ ${#DATASETS[@]} -gt 0 ]] || { echo "--datasets requires names" >&2; exit 2; }
            ;;
        --workers|--judge-workers|--model|--generation-model|--max-tokens|--ctx-size|--results-dir|\
        --generation-dir|--max-items|--samples-per-problem|--limit|--max-sentences|--judge-program|\
        --judge-model|--position-bins|--quantization|--bootstrap-samples|\
        --generation-speculation|--generation-quantization|--cpu-offload-gb|--tensor-parallel-size)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "$1 requires a value" >&2; exit 2; }
            case "$1" in
                --workers) WORKERS="$2" ;;
                --judge-workers) JUDGE_WORKERS="$2" ;;
                --model) PIPELINE_MODEL="$2" ;;
                --generation-model) GENERATION_MODEL="$2" ;;
                --max-tokens) MAX_TOKENS="$2" ;;
                --ctx-size) CTX_SIZE="$2" ;;
                --results-dir) RESULTS_DIR="$2" ;;
                --generation-dir) GENERATION_DIR="$2" ;;
                --max-items) MAX_ITEMS="$2" ;;
                --samples-per-problem) SAMPLES_PER_PROBLEM="$2" ;;
                --limit) LIMIT="$2" ;;
                --max-sentences) MAX_SENTENCES="$2" ;;
                --judge-program) JUDGE_PROGRAM="$2" ;;
                --judge-model) JUDGE_MODEL="$2" ;;
                --position-bins) POSITION_BINS="$2" ;;
                --quantization) QUANTIZATION="$2" ;;
                --generation-speculation) GENERATION_SPECULATION="$2" ;;
                --generation-quantization) GENERATION_QUANTIZATION="$2" ;;
                --cpu-offload-gb) CPU_OFFLOAD_GB="$2" ;;
                --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2" ;;
                --bootstrap-samples) BOOTSTRAP_SAMPLES="$2" ;;
            esac
            shift 2 ;;
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --skip-annotate) SKIP_ANNOTATE=true; shift ;;
        --skip-tagging) SKIP_TAGGING=true; shift ;;
        --skip-sampling) SKIP_SAMPLING=true; shift ;;
        --skip-forward) SKIP_FORWARD=true; shift ;;
        --skip-analyze) SKIP_ANALYZE=true; shift ;;
        --all-router-layers) ALL_ROUTER_LAYERS=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
FORWARD_MODEL="${PIPELINE_MODEL:-unsloth/Qwen3.5-35B-A3B}"
GENERATION_MODEL="${GENERATION_MODEL:-${PIPELINE_MODEL:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}}"
# Use the script's own directory: PHYS_DIR may point at an isolated test workspace.
PROBE_CHECK_ARGS=()
if [[ "$DRY_RUN" != true && "$SKIP_FORWARD" != true && "$ALL_ROUTER_LAYERS" != true ]]; then
    PROBE_CHECK_ARGS=(--check-probe-results
        "$PHYS_DIR/results/probeTest/qwen3.5-35b-a3b-gptq-int4/probes/results.json")
fi
PROFILE_OUTPUT="$(python3 "$(dirname "${BASH_SOURCE[0]}")/model_profiles.py" \
    --generation-model "$GENERATION_MODEL" --forward-model "$FORWARD_MODEL" "${PROBE_CHECK_ARGS[@]}")"
mapfile -t PROFILE <<< "$PROFILE_OUTPUT"
GENERATION_FAMILY="${PROFILE[0]}"
GENERATION_REASONING_PARSER="${PROFILE[1]}"
GENERATION_MTP="${PROFILE[2]}"
CTX_SIZE="${CTX_SIZE:-${PROFILE[3]}}"
QUANTIZATION="${QUANTIZATION:-${PROFILE[4]}}"
GENERATION_LANGUAGE_ONLY="${PROFILE[5]}"
case "$GENERATION_SPECULATION" in
    auto) ;;
    mtp) GENERATION_MTP=true ;;
    none) GENERATION_MTP=false ;;
    *) echo "--generation-speculation must be auto, mtp, or none" >&2; exit 2 ;;
esac
[[ "$CPU_OFFLOAD_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "--cpu-offload-gb must be non-negative" >&2; exit 2;
}
for value in "$MAX_TOKENS" "$CTX_SIZE" "$JUDGE_CTX_SIZE" "$WORKERS" "$JUDGE_WORKERS" \
    "$POSITION_BINS" "$BOOTSTRAP_SAMPLES" "$SERVER_READY_TIMEOUT" "$SPECULATIVE_TOKENS" "$MAX_SENTENCES" \
    "$MAX_NUM_BATCHED_TOKENS" "$TENSOR_PARALLEL_SIZE" "${LIMIT:-1}" "${MAX_ITEMS:-1}" "${SAMPLES_PER_PROBLEM:-1}"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Counts must be positive integers" >&2; exit 2; }
done
if (( MAX_TOKENS >= CTX_SIZE )); then
    echo "--max-tokens must be smaller than --ctx-size to leave room for the prompt" >&2; exit 2
fi
GENERATION_DIR="${GENERATION_DIR:-$RESULTS_DIR/generation}"
REASONING_DIR="$RESULTS_DIR/reasoning-vllm-v1"
VIEW_MODES=(full class position)
ANNOTATION_ARGS=(--annotation-dir "$REASONING_DIR/annotations")
if [[ "$SKIP_TAGGING" == true ]]; then
    SKIP_SAMPLING=true
    SKIP_ANNOTATE=true
    VIEW_MODES=(full position)
    ANNOTATION_ARGS=()
    REASONING_DIR="$RESULTS_DIR/reasoning-vllm-untagged-v1"
fi
REPLAY_GENERATION_DIR="$GENERATION_DIR"
if [[ "$SKIP_SAMPLING" != true ]]; then
    REPLAY_GENERATION_DIR="$REASONING_DIR/sampling/generation"
fi
for path in "$RESULTS_DIR" "$GENERATION_DIR" "$JUDGE_PROGRAM"; do
    [[ -n "$path" && "$path" != /* && "/$path/" != */../* ]] || {
        echo "Result, generation and program paths must be below the repository root" >&2; exit 2;
    }
done
if [[ "$SKIP_GENERATE" == true && "$SKIP_ANNOTATE" == true && \
      "$SKIP_FORWARD" == true && "$SKIP_ANALYZE" == true ]]; then
    echo "All stages were skipped; nothing to do." >&2; exit 2
fi
cd "$PHYS_DIR"
if [[ "$DRY_RUN" != true ]]; then
    command -v docker >/dev/null || { echo "Docker is required" >&2; exit 1; }
    [[ -x "$STAGE_LAUNCHER" ]] || { echo "Missing executable stage launcher" >&2; exit 1; }
    if [[ "$SKIP_GENERATE" != true || "$SKIP_ANNOTATE" != true ]]; then
        command -v curl >/dev/null || { echo "curl is required for readiness checks" >&2; exit 1; }
        docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1 || {
            echo "Missing vLLM image. Run: docker pull $VLLM_IMAGE" >&2; exit 1;
        }
    fi
    if [[ "$SKIP_ANNOTATE" != true && ! -s "$JUDGE_PROGRAM" ]]; then
        echo "Missing frozen GEPA program: $JUDGE_PROGRAM" >&2; exit 1
    fi
    mkdir -p "$PHYS_DIR/slurm_logs" "$PHYS_DIR/$REASONING_DIR"
fi
RUN_ID="$(id -u)-$$"
RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
ACTIVE_CONTAINER=""
ACTIVE_LOG=""
LIMIT_ARGS=()
[[ -n "$LIMIT" ]] && LIMIT_ARGS=(--limit "$LIMIT")
LAYER_ARGS=()
[[ "$ALL_ROUTER_LAYERS" != true ]] || LAYER_ARGS=(--all-router-layers)
stage_number=0
TOTAL_STAGES=0
for skipped in "$SKIP_GENERATE" "$SKIP_ANNOTATE" "$SKIP_FORWARD" "$SKIP_ANALYZE"; do
    [[ "$skipped" == true ]] || TOTAL_STAGES=$((TOTAL_STAGES + 1))
done
if [[ "$SKIP_SAMPLING" != true && "$SKIP_ANNOTATE" != true ]]; then
    TOTAL_STAGES=$((TOTAL_STAGES + 1))
fi
update() { printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }
stop_server() {
    if [[ -n "$ACTIVE_CONTAINER" ]]; then
        update "Stopping $ACTIVE_CONTAINER and releasing GPU memory"
        if [[ "$DRY_RUN" == true ]]; then
            print_command docker stop --timeout 30 "$ACTIVE_CONTAINER"
            print_command docker rm "$ACTIVE_CONTAINER"
        else
            docker logs "$ACTIVE_CONTAINER" >"$ACTIVE_LOG" 2>&1 || true
            if docker container inspect "$ACTIVE_CONTAINER" >/dev/null 2>&1; then
                docker stop --timeout 30 "$ACTIVE_CONTAINER" >/dev/null
                docker rm "$ACTIVE_CONTAINER" >/dev/null
            fi
        fi
        ACTIVE_CONTAINER=""
    fi
}
cleanup() {
    exit_code=$?
    trap - EXIT
    stop_server
    if [[ $exit_code -ne 0 ]]; then
        update "Pipeline stopped with exit code $exit_code"
        [[ ! -f "$ACTIVE_LOG" ]] || echo "Server log: $ACTIVE_LOG" >&2
    fi
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_server() {
    local stage="$1" model="$2" context="$3" port="$4" workers="$5"
    [[ -z "$ACTIVE_CONTAINER" ]] || { echo "Previous server is still active" >&2; exit 1; }
    local container="correlation-vllm-$stage-$RUN_ID"
    if [[ "$DRY_RUN" != true ]] && docker container inspect "$container" >/dev/null 2>&1; then
        echo "Container already exists: $container" >&2; exit 1
    fi
    local -a command=(
        docker run --detach --name "$container" --gpus "\"device=$CUDA_VISIBLE_DEVICES\""
        --ipc=host -v "$HF_CACHE_DIR:$HF_CACHE_DIR" -e "HF_HOME=$HF_CACHE_DIR"
        -p "$SERVER_HOST:$port:$port" --entrypoint vllm "$VLLM_IMAGE" serve "$model"
        --served-model-name "$model" --host 0.0.0.0 --port "$port" --api-key "$API_KEY"
        --max-model-len "$context" --enable-prefix-caching --max-num-seqs "$workers"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --generation-config vllm
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    )
    # The dense NVFP4 judge needs graph memory headroom on a single 32 GB 5090.
    if [[ "$stage" == judge ]]; then
        command+=(--enforce-eager --kv-cache-dtype fp8 --reasoning-parser qwen3
            --language-model-only
            --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPECULATIVE_TOKENS}")
    else
        command+=(--reasoning-parser "$GENERATION_REASONING_PARSER"
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --cpu-offload-gb "$CPU_OFFLOAD_GB")
        [[ "$GENERATION_LANGUAGE_ONLY" != true ]] || command+=(--language-model-only)
        [[ -z "$GENERATION_QUANTIZATION" ]] || command+=(--quantization "$GENERATION_QUANTIZATION")
        if [[ "$GENERATION_MTP" == true ]]; then
            command+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPECULATIVE_TOKENS}")
        fi
    fi
    [[ -z "${HF_TOKEN:-}" ]] || command=("${command[@]:0:2}" -e HF_TOKEN "${command[@]:2}")
    update "Starting vLLM $stage: $model ($workers concurrent requests)"
    print_command "${command[@]}"
    ACTIVE_CONTAINER="$container"
    ACTIVE_LOG="$PHYS_DIR/slurm_logs/correlation-vllm-$stage-$RUN_TIMESTAMP.log"
    if [[ "$DRY_RUN" != true ]]; then
        "${command[@]}" >/dev/null
        local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
        until curl --silent --fail --max-time 5 "http://$SERVER_HOST:$port/health" >/dev/null 2>&1; do
            if (( SECONDS >= deadline )) || \
                [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]]; then
                echo "vLLM $stage exited or did not become ready" >&2
                docker logs --tail 60 "$container" >&2 || true
                exit 1
            fi
            sleep 2
        done
    fi
}
run_stage() {
    stage_number=$((stage_number + 1))
    update "Stage $stage_number/$TOTAL_STAGES: $1"; shift
    print_command "$@"
    if [[ "$DRY_RUN" == true ]]; then
        update "Stage $stage_number/$TOTAL_STAGES planned (dry run; not executed)"
    else
        "$@"
        update "Stage $stage_number/$TOTAL_STAGES complete"
    fi
}
echo "=== vLLM correlation experiment ==="
echo "  Datasets:          ${DATASETS[*]}"
echo "  Generation model:  $GENERATION_MODEL"
echo "  Forward model:     $FORWARD_MODEL"
echo "  Generation profile: $GENERATION_FAMILY (parser=$GENERATION_REASONING_PARSER, MTP=$GENERATION_MTP)"
echo "  Forward quantization: $QUANTIZATION"
echo "  Judge model:       $JUDGE_MODEL"
echo "  Concurrent calls:  generation=$WORKERS tagging=$JUDGE_WORKERS"
echo "  Generation path:   $GENERATION_DIR"
echo "  Replay input:      $REPLAY_GENERATION_DIR"
if [[ "$SKIP_SAMPLING" != true ]]; then
    echo "  Sentence sampling: error-rate weighted, cap=$MAX_SENTENCES, sample-id=0, seed=42"
fi
echo "  Reasoning outputs: $REASONING_DIR"
echo "  Reasoning views:   ${VIEW_MODES[*]}"
if [[ "$SKIP_GENERATE" != true ]]; then
    start_server generation "$GENERATION_MODEL" "$CTX_SIZE" "$SERVER_PORT" "$WORKERS"
    GENERATE_COMMAND=(
        env HF_CACHE_DIR="$HF_CACHE_DIR" "$STAGE_LAUNCHER" generate --datasets "${DATASETS[@]}"
        --model "$GENERATION_MODEL" --save-token-ids
        --target-model-id "$FORWARD_MODEL"
        --workers "$WORKERS" --max-tokens "$MAX_TOKENS" --output-dir "$GENERATION_DIR"
        --base-url "http://$SERVER_HOST:$SERVER_PORT/v1" --api-key "$API_KEY"
    )
    if [[ "$GENERATION_MTP" == true ]]; then
        GENERATE_COMMAND+=(--draft-model-id "$GENERATION_MODEL")
    else
        GENERATE_COMMAND+=(--draft-model-id "")
    fi
    [[ -n "$MAX_ITEMS" ]] && GENERATE_COMMAND+=(--max-items "$MAX_ITEMS")
    [[ -n "$SAMPLES_PER_PROBLEM" ]] && GENERATE_COMMAND+=(--samples-per-problem "$SAMPLES_PER_PROBLEM")
    run_stage "Generate benchmark reasoning" "${GENERATE_COMMAND[@]}"
    stop_server
fi
if [[ "$SKIP_ANNOTATE" != true ]]; then
    if [[ "$SKIP_SAMPLING" != true ]]; then
        run_stage "Sample sentences by benchmark error rate" \
            env HF_CACHE_DIR="$HF_CACHE_DIR" "$STAGE_LAUNCHER" sample --datasets "${DATASETS[@]}" \
            --generation-dir "$GENERATION_DIR" --generation-model "$GENERATION_MODEL" \
            --output-dir "$REASONING_DIR/sampling" \
            --weight-by error-rate --sample-id 0 --seed 42 --max-sentences "$MAX_SENTENCES"
    fi
    start_server judge "$JUDGE_MODEL" "$JUDGE_CTX_SIZE" "$JUDGE_PORT" "$JUDGE_WORKERS"
    run_stage "Tag sentences with the frozen GEPA classifier" \
        env HF_CACHE_DIR="$HF_CACHE_DIR" "$STAGE_LAUNCHER" annotate --datasets "${DATASETS[@]}" \
        --generation-dir "$REPLAY_GENERATION_DIR" --generation-model "$GENERATION_MODEL" \
        --output-dir "$REASONING_DIR/annotations" --judge-program "$JUDGE_PROGRAM" \
        --judge-model "$JUDGE_MODEL" --workers "$JUDGE_WORKERS" \
        --base-url "http://$SERVER_HOST:$JUDGE_PORT/v1" --api-key "$API_KEY" "${LIMIT_ARGS[@]}"
    stop_server
fi
if [[ "$SKIP_FORWARD" != true ]]; then
    run_stage "Forward extraction for selected reasoning views" \
        env HF_CACHE_DIR="$HF_CACHE_DIR" "$STAGE_LAUNCHER" forward --datasets "${DATASETS[@]}" \
        --model-id "$FORWARD_MODEL" \
        --generation-dir "$REPLAY_GENERATION_DIR" --generation-model "$GENERATION_MODEL" \
        "${ANNOTATION_ARGS[@]}" --output-dir "$REASONING_DIR/forward" \
        --views "${VIEW_MODES[@]}" --position-bins "$POSITION_BINS" \
        --quantization "$QUANTIZATION" "${LAYER_ARGS[@]}" "${LIMIT_ARGS[@]}"
fi
if [[ "$SKIP_ANALYZE" != true ]]; then
    run_stage "Correctness, metric-pair, expert and reasoning-view correlations" \
        env HF_CACHE_DIR="$HF_CACHE_DIR" "$STAGE_LAUNCHER" analyze --datasets "${DATASETS[@]}" \
        --model-id "$FORWARD_MODEL" \
        --forward-dir "$REASONING_DIR/forward" --output-dir "$REASONING_DIR/analysis" \
        --views "${VIEW_MODES[@]}" --bootstrap-samples "$BOOTSTRAP_SAMPLES" "${LIMIT_ARGS[@]}"
fi
stop_server
trap - EXIT INT TERM
if [[ "$DRY_RUN" == true ]]; then
    update "Dry-run plan complete; no stages executed or results written"
else
    update "Correlation experiment complete"
fi
echo "  Generation:  $GENERATION_DIR/"
if [[ "$SKIP_TAGGING" != true ]]; then echo "  Annotations: $REASONING_DIR/annotations/"; fi
echo "  Forward:     $REASONING_DIR/forward/"
echo "  Analysis:    $REASONING_DIR/analysis/"

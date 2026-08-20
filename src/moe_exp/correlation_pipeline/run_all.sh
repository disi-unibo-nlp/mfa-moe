#!/usr/bin/env bash
set -euo pipefail

# One-command correlation pilot:
#   native-MTP llama.cpp server -> generation -> cleanup -> Unsloth forward -> analysis

PHYS_DIR="${PHYS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/llms}"
LLAMACPP_LAUNCHER="${LLAMACPP_LAUNCHER:-${PHYS_DIR}/src/common/llamacpp/serve_qwen3_5_35b_a3b_mtp.sh}"
STAGE_LAUNCHER="${STAGE_LAUNCHER:-${PHYS_DIR}/src/moe_exp/correlation_pipeline/run_docker.sh}"
SERVER_CONTAINER="${SERVER_CONTAINER:-correlation-qwen35-mtp}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8080}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORKERS="${WORKERS:-1}"
QUANTIZATION="${QUANTIZATION:-unsloth-4bit}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-500}"
DRY_RUN="${DRY_RUN:-false}"

DATASETS=(math500 aime24 minerva)
MAX_ITEMS=""
SAMPLES_PER_PROBLEM=""
SKIP_GENERATE=false
SKIP_FORWARD=false
SKIP_ANALYZE=false

usage() {
    cat <<'EOF'
Usage: src/moe_exp/correlation_pipeline/run_all.sh [options]

Runs the complete Unsloth Qwen3.5 + native-MTP correlation pilot in one command.

Options:
  --datasets NAME...          Datasets to run (default: math500 aime24 minerva)
    --workers N                 Concurrent llama.cpp requests (default: 1 for MTP)
  --max-items N               Limit problems per dataset (smoke test)
  --samples-per-problem N     Override benchmark sampling counts (smoke test)
    --quantization MODE         Forward quantization (default: unsloth-4bit)
  --bootstrap-samples N       Analysis cluster bootstraps (default: 500)
  --skip-generate             Reuse existing generated traces
  --skip-forward              Reuse existing forward tensors
  --skip-analyze              Stop after forward extraction
  -h, --help                  Show this help

Environment overrides include CUDA_VISIBLE_DEVICES, SERVER_READY_TIMEOUT,
PHYS_DIR, HF_CACHE_DIR, IMAGE_NAME, MODEL_NAME, and MODEL_REPO.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            shift
            DATASETS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DATASETS+=("$1")
                shift
            done
            if [[ ${#DATASETS[@]} -eq 0 ]]; then
                echo "--datasets requires at least one dataset" >&2
                exit 2
            fi
            ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --max-items) MAX_ITEMS="$2"; shift 2 ;;
        --samples-per-problem) SAMPLES_PER_PROBLEM="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --bootstrap-samples) BOOTSTRAP_SAMPLES="$2"; shift 2 ;;
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --skip-forward) SKIP_FORWARD=true; shift ;;
        --skip-analyze) SKIP_ANALYZE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$SKIP_GENERATE" == true && "$SKIP_FORWARD" == true && "$SKIP_ANALYZE" == true ]]; then
    echo "All stages were skipped; nothing to do." >&2
    exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required; host Python 3.10 is intentionally not used." >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to wait for llama.cpp readiness." >&2
    exit 1
fi
if [[ ! -x "$LLAMACPP_LAUNCHER" || ! -x "$STAGE_LAUNCHER" ]]; then
    echo "Pipeline launchers are not executable. From the repository root run:" >&2
    echo "  chmod +x src/common/llamacpp/serve_qwen3_5_35b_a3b_mtp.sh" >&2
    echo "  chmod +x src/moe_exp/correlation_pipeline/run_docker.sh" >&2
    exit 1
fi

mkdir -p "$PHYS_DIR/slurm_logs" "$PHYS_DIR/results/correlation_pipeline"
RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SERVER_LOG="${PHYS_DIR}/slurm_logs/correlation-mtp-${RUN_TIMESTAMP}.log"
SERVER_STARTED=false
SERVER_PID=""

stage_number=0
TOTAL_STAGES=0
[[ "$SKIP_GENERATE" != true ]] && TOTAL_STAGES=$((TOTAL_STAGES + 1))
[[ "$SKIP_FORWARD" != true ]] && TOTAL_STAGES=$((TOTAL_STAGES + 1))
[[ "$SKIP_ANALYZE" != true ]] && TOTAL_STAGES=$((TOTAL_STAGES + 1))

update() {
    printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

stop_server() {
    if [[ "$SERVER_STARTED" == true ]]; then
        update "Stopping native-MTP llama.cpp server and releasing GPU memory"
        if docker container inspect "$SERVER_CONTAINER" >/dev/null 2>&1; then
            docker stop --timeout 30 "$SERVER_CONTAINER" >/dev/null 2>&1 || true
        fi
        if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
            kill "$SERVER_PID" 2>/dev/null || true
        fi
        if [[ -n "$SERVER_PID" ]]; then
            wait "$SERVER_PID" 2>/dev/null || true
        fi
        # The launcher may have crossed container creation while it was being
        # stopped. Recheck the stable name after waiting for its process.
        if docker container inspect "$SERVER_CONTAINER" >/dev/null 2>&1; then
            docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
        fi
        SERVER_STARTED=false
        SERVER_PID=""
    fi
}

cleanup() {
    exit_code=$?
    stop_server
    if [[ $exit_code -ne 0 ]]; then
        update "Pipeline stopped with exit code $exit_code"
        if [[ -f "$SERVER_LOG" ]]; then
            echo "MTP server log: $SERVER_LOG" >&2
        fi
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

run_stage() {
    stage_number=$((stage_number + 1))
    update "Stage ${stage_number}/${TOTAL_STAGES}: $1"
    shift
    print_command "$@"
    if [[ "$DRY_RUN" != true ]]; then
        "$@"
    fi
    update "Stage ${stage_number}/${TOTAL_STAGES} complete"
}

echo "=== Unsloth Qwen3.5 + native-MTP correlation pilot ==="
echo "  Datasets:           ${DATASETS[*]}"
echo "  GPU:                $CUDA_VISIBLE_DEVICES"
echo "  Generation workers: $WORKERS"
echo "  Forward precision:  $QUANTIZATION"
echo "  Bootstrap samples:  $BOOTSTRAP_SAMPLES"
echo "  Hugging Face cache:  $HF_CACHE_DIR"
echo "  Host Python:        unused (stages run in Python 3.11 Docker image)"

if [[ "$SKIP_GENERATE" != true ]]; then
    if docker container inspect "$SERVER_CONTAINER" >/dev/null 2>&1; then
        if [[ "$(docker inspect "$SERVER_CONTAINER" --format '{{.State.Running}}')" == "true" ]]; then
            echo "A running container named $SERVER_CONTAINER already exists. Stop it before running:" >&2
            echo "  docker rm -f $SERVER_CONTAINER" >&2
            exit 1
        fi
        update "Removing stale stopped container $SERVER_CONTAINER"
        docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
    fi

    update "Starting Unsloth Qwen3.5-35B-A3B with native MTP"
    if [[ "$DRY_RUN" == true ]]; then
        print_command env \
            CONTAINER_NAME="$SERVER_CONTAINER" \
            HOST="$SERVER_HOST" \
            PORT="$SERVER_PORT" \
            GPU_DEVICE="$CUDA_VISIBLE_DEVICES" \
            HF_CACHE_DIR="$HF_CACHE_DIR" \
            "$LLAMACPP_LAUNCHER"
    else
        env \
            CONTAINER_NAME="$SERVER_CONTAINER" \
            HOST="$SERVER_HOST" \
            PORT="$SERVER_PORT" \
            GPU_DEVICE="$CUDA_VISIBLE_DEVICES" \
            HF_CACHE_DIR="$HF_CACHE_DIR" \
            "$LLAMACPP_LAUNCHER" >"$SERVER_LOG" 2>&1 &
        SERVER_PID=$!
        SERVER_STARTED=true

        update "Waiting for llama.cpp health at http://${SERVER_HOST}:${SERVER_PORT}/health"
        start_seconds="$(date +%s)"
        while true; do
            if curl --silent --fail --max-time 5 \
                "http://${SERVER_HOST}:${SERVER_PORT}/health" >/dev/null 2>&1; then
                break
            fi
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "MTP server exited before becoming healthy. Last log lines:" >&2
                wait "$SERVER_PID" 2>/dev/null || true
                tail -80 "$SERVER_LOG" >&2 || true
                exit 1
            fi
            now_seconds="$(date +%s)"
            if (( now_seconds - start_seconds >= SERVER_READY_TIMEOUT )); then
                echo "MTP server did not become healthy within ${SERVER_READY_TIMEOUT}s." >&2
                tail -80 "$SERVER_LOG" >&2 || true
                exit 1
            fi
            sleep 5
        done
        update "MTP server is healthy; log: $SERVER_LOG"
    fi

    GENERATE_COMMAND=(
        env HF_CACHE_DIR="$HF_CACHE_DIR" "$STAGE_LAUNCHER" generate
        --datasets "${DATASETS[@]}"
        --workers "$WORKERS"
        --base-url "http://${SERVER_HOST}:${SERVER_PORT}/v1"
    )
    [[ -n "$MAX_ITEMS" ]] && GENERATE_COMMAND+=(--max-items "$MAX_ITEMS")
    [[ -n "$SAMPLES_PER_PROBLEM" ]] && \
        GENERATE_COMMAND+=(--samples-per-problem "$SAMPLES_PER_PROBLEM")
    run_stage "Generate benchmark reasoning traces" "${GENERATE_COMMAND[@]}"
    stop_server
else
    update "Generation skipped; reusing existing trace files"
fi

if [[ "$SKIP_FORWARD" != true ]]; then
    FORWARD_COMMAND=(
        env HF_CACHE_DIR="$HF_CACHE_DIR" "$STAGE_LAUNCHER" forward
        --datasets "${DATASETS[@]}"
        --quantization "$QUANTIZATION"
    )
    run_stage "Teacher-forced router and hidden-state extraction" "${FORWARD_COMMAND[@]}"
else
    update "Forward extraction skipped; reusing existing tensor files"
fi

if [[ "$SKIP_ANALYZE" != true ]]; then
    ANALYZE_COMMAND=(
        env HF_CACHE_DIR="$HF_CACHE_DIR" "$STAGE_LAUNCHER" analyze
        --datasets "${DATASETS[@]}"
        --bootstrap-samples "$BOOTSTRAP_SAMPLES"
    )
    run_stage "Cluster-aware correlation analysis" "${ANALYZE_COMMAND[@]}"
else
    update "Analysis skipped"
fi

trap - EXIT INT TERM
stop_server
update "Correlation pilot complete"
echo "  Generation: results/correlation_pipeline/generation/"
echo "  Forward:    results/correlation_pipeline/forward/"
echo "  Analysis:   results/correlation_pipeline/analysis/"
#!/bin/bash
#SBATCH --job-name=exp0a-gepa
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --nodelist=faretra

# Experiment 0a: start the local llama.cpp judge and run GEPA in two Docker
# containers inside one SLURM allocation.
#
# Required host files (defaults match the other launchers in this repository):
#   /home/tassinari/moe-mfaExperiments/data/Schoenfeld_Reasoning
#   /llms/Qwen3.6-27B-UD-Q4_K_XL.gguf
#
# Example:
#   sbatch run_experiment0a.sh --gepa-auto light
#   sbatch run_experiment0a.sh --max-full-evals 10 --seed 23

set -euo pipefail

PHYS_DIR="${PHYS_DIR:-/home/tassinari/moe-mfaExperiments}"
DATASET_DIR="${DATASET_DIR:-${PHYS_DIR}/data/Schoenfeld_Reasoning}"
MODEL_DIR="${MODEL_DIR:-/llms}"
MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B-UD-Q4_K_XL.gguf}"
PROJECT_IMAGE="${PROJECT_IMAGE:-moe-mfa-experiments:latest}"
LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-llama.cpp:localcuda}"
OUTPUT_DIR="${OUTPUT_DIR:-results/exp0a/qwen3.6-27b}"
API_KEY="${LLAMA_API_KEY:-local-llamacpp-key}"

# Conservative one-GPU defaults. Increase PARALLEL and NUM_THREADS together
# only when the GPU has enough memory for multiple KV-cache slots.
CTX_SIZE="${CTX_SIZE:-8192}"
PARALLEL="${PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
GPU_LAYERS="${GPU_LAYERS:-999}"
NUM_THREADS="${NUM_THREADS:-1}"
MAX_TOKENS="${MAX_TOKENS:-64}"
REFLECTION_MAX_TOKENS="${REFLECTION_MAX_TOKENS:-2048}"
REFLECTION_TEMPERATURE="${REFLECTION_TEMPERATURE:-0.7}"
TRAIN_DOCUMENTS="${TRAIN_DOCUMENTS:-26}"
VAL_DOCUMENTS="${VAL_DOCUMENTS:-6}"
SEED="${SEED:-42}"
PROMPT_VARIANT="${PROMPT_VARIANT:-few-shot}"
FEW_SHOT_EXAMPLES="${FEW_SHOT_EXAMPLES:-21}"
GEPA_REWARD="${GEPA_REWARD:-balanced}"
SELECTION_METRIC="${SELECTION_METRIC:-balanced_accuracy}"
MAX_CLASS_RECALL_DROP="${MAX_CLASS_RECALL_DROP:-0.10}"
CV_FOLDS="${CV_FOLDS:-0}"
CV_INNER_VAL_DOCUMENTS="${CV_INNER_VAL_DOCUMENTS:-5}"
LOCKED_TEST_DOCUMENTS="${LOCKED_TEST_DOCUMENTS:-6}"
EVALUATE_LOCKED_TEST=false
RUNNER_MEMORY="${RUNNER_MEMORY:-16g}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BUDGET_KIND="gepa-auto"
BUDGET_VALUE="light"
BUDGET_COUNT=0
TEST_DOCUMENTS=""
MAX_UNITS=""

usage() {
    cat <<'EOF'
Usage: sbatch run_experiment0a.sh [options]

GEPA budget (choose at most one; default: --gepa-auto light):
  --gepa-auto light|medium|heavy
  --max-full-evals N
  --max-metric-calls N

Experiment options:
  --dataset-dir PATH          Host path to Schoenfeld_Reasoning
  --model-dir PATH            Host directory containing the GGUF
  --model-name FILE           GGUF filename
  --output-dir PATH           Path relative to the repository
  --train-documents N         Default: 26
  --val-documents N           Default: 6
  --seed N                    Default: 42
  --prompt-variant NAME       base or few-shot (default: few-shot)
  --few-shot-examples N       Curated contrastive examples (default: 21)
  --gepa-reward NAME          balanced or exact (default: balanced)
  --selection-metric NAME     balanced_accuracy, macro_f1, or accuracy
  --max-class-recall-drop X   Validation safety threshold (default: 0.10)
  --cv-folds N                Nested grouped CV folds; 0 runs a final fit
  --cv-inner-val-documents N  Inner validation responses per CV fold (default: 5)
  --locked-test-documents N   Responses excluded from CV (default: 6)
  --evaluate-locked-test      Explicitly evaluate locked test after final selection
  --num-threads N             Concurrent evaluator calls (default: 1)
  --max-tokens N              Classification generation cap (default: 64)
  --reflection-max-tokens N   GEPA reflection cap (default: 2048)
  --reflection-temperature X  GEPA reflection sampling (default: 0.7)

llama.cpp options:
  --ctx-size N                Total server context (default: 8192)
  --parallel N                Server slots (default: 1)
  --batch-size N              Default: 512
  --gpu-layers N              Default: 999 (full offload)

Smoke-test only:
  --test-documents N
  --max-units-per-document N
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gepa-auto) BUDGET_KIND="gepa-auto"; BUDGET_VALUE="$2"; BUDGET_COUNT=$((BUDGET_COUNT + 1)); shift 2 ;;
        --max-full-evals) BUDGET_KIND="max-full-evals"; BUDGET_VALUE="$2"; BUDGET_COUNT=$((BUDGET_COUNT + 1)); shift 2 ;;
        --max-metric-calls) BUDGET_KIND="max-metric-calls"; BUDGET_VALUE="$2"; BUDGET_COUNT=$((BUDGET_COUNT + 1)); shift 2 ;;
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --train-documents) TRAIN_DOCUMENTS="$2"; shift 2 ;;
        --val-documents) VAL_DOCUMENTS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --prompt-variant) PROMPT_VARIANT="$2"; shift 2 ;;
        --few-shot-examples) FEW_SHOT_EXAMPLES="$2"; shift 2 ;;
        --gepa-reward) GEPA_REWARD="$2"; shift 2 ;;
        --selection-metric) SELECTION_METRIC="$2"; shift 2 ;;
        --max-class-recall-drop) MAX_CLASS_RECALL_DROP="$2"; shift 2 ;;
        --cv-folds) CV_FOLDS="$2"; shift 2 ;;
        --cv-inner-val-documents) CV_INNER_VAL_DOCUMENTS="$2"; shift 2 ;;
        --locked-test-documents) LOCKED_TEST_DOCUMENTS="$2"; shift 2 ;;
        --evaluate-locked-test) EVALUATE_LOCKED_TEST=true; shift ;;
        --num-threads) NUM_THREADS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --reflection-max-tokens) REFLECTION_MAX_TOKENS="$2"; shift 2 ;;
        --reflection-temperature) REFLECTION_TEMPERATURE="$2"; shift 2 ;;
        --ctx-size) CTX_SIZE="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        --test-documents) TEST_DOCUMENTS="$2"; shift 2 ;;
        --max-units-per-document) MAX_UNITS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if (( BUDGET_COUNT > 1 )); then
    echo "Choose only one GEPA budget option." >&2
    exit 1
fi
if [[ "$PROMPT_VARIANT" != "base" && "$PROMPT_VARIANT" != "few-shot" ]]; then
    echo "--prompt-variant must be base or few-shot" >&2
    exit 1
fi
if [[ "$GEPA_REWARD" != "balanced" && "$GEPA_REWARD" != "exact" ]]; then
    echo "--gepa-reward must be balanced or exact" >&2
    exit 1
fi
case "$SELECTION_METRIC" in
    balanced_accuracy|macro_f1|accuracy) ;;
    *) echo "--selection-metric must be balanced_accuracy, macro_f1, or accuracy" >&2; exit 1 ;;
esac
if (( CV_FOLDS == 1 )); then
    echo "--cv-folds must be 0 or at least 2" >&2
    exit 1
fi
if (( CV_FOLDS > 0 )) && [[ "$EVALUATE_LOCKED_TEST" == true ]]; then
    echo "Nested CV never evaluates the locked test." >&2
    exit 1
fi
case "$BUDGET_KIND:$BUDGET_VALUE" in
    gepa-auto:light|gepa-auto:medium|gepa-auto:heavy) ;;
    gepa-auto:*) echo "--gepa-auto must be light, medium, or heavy" >&2; exit 1 ;;
esac

if [[ ! -d "$PHYS_DIR" ]]; then
    echo "Repository directory does not exist: $PHYS_DIR" >&2
    exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Dataset directory does not exist: $DATASET_DIR" >&2
    echo "Clone https://github.com/MingLiiii/Schoenfeld_Reasoning there first." >&2
    exit 1
fi
if [[ ! -s "$MODEL_DIR/$MODEL_NAME" ]]; then
    echo "GGUF model does not exist or is empty: $MODEL_DIR/$MODEL_NAME" >&2
    exit 1
fi
if ! docker image inspect "$PROJECT_IMAGE" >/dev/null 2>&1; then
    echo "Missing Docker image: $PROJECT_IMAGE" >&2
    echo "Build it from the repository root with: docker build -t $PROJECT_IMAGE ." >&2
    exit 1
fi
if ! docker image inspect "$LLAMACPP_IMAGE" >/dev/null 2>&1; then
    echo "Missing Docker image: $LLAMACPP_IMAGE" >&2
    echo "Build it with: docker build -t $LLAMACPP_IMAGE src/common/llamacpp" >&2
    exit 1
fi

if (( NUM_THREADS > PARALLEL )); then
    echo "--num-threads ($NUM_THREADS) cannot exceed llama.cpp --parallel ($PARALLEL)" >&2
    exit 1
fi

mkdir -p "$PHYS_DIR/$OUTPUT_DIR"
# NFS root_squash maps container root to nobody on the current cluster.
chmod -R 777 "$PHYS_DIR/$OUTPUT_DIR"

JOB_TAG="${SLURM_JOB_ID:-manual}-$$"
NETWORK_NAME="exp0a-${JOB_TAG}"
SERVER_CONTAINER="exp0a-llamacpp-${JOB_TAG}"

cleanup() {
    docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
    docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker network create "$NETWORK_NAME" >/dev/null

echo "=== Experiment 0a ==="
echo "  Job:        ${SLURM_JOB_ID:-manual}"
echo "  Dataset:    $DATASET_DIR"
echo "  Model:      $MODEL_DIR/$MODEL_NAME"
echo "  GPU:        $CUDA_VISIBLE_DEVICES"
echo "  GEPA:       --$BUDGET_KIND $BUDGET_VALUE"
echo "  Prompt:     $PROMPT_VARIANT"
echo "  Reward:     $GEPA_REWARD; selection=$SELECTION_METRIC"
echo "  CV:         folds=$CV_FOLDS; locked-test=$LOCKED_TEST_DOCUMENTS"
echo "  Split:      train=$TRAIN_DOCUMENTS, val=$VAL_DOCUMENTS, test=remainder"
echo "  Concurrency: evaluator=$NUM_THREADS, server slots=$PARALLEL"
echo "  Output:     $PHYS_DIR/$OUTPUT_DIR"

docker run --detach \
    --name "$SERVER_CONTAINER" \
    --network "$NETWORK_NAME" \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --ipc=host \
    -v "$MODEL_DIR":/models:ro \
    "$LLAMACPP_IMAGE" \
    --model "/models/$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8080 \
    --api-key "$API_KEY" \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --batch-size "$BATCH_SIZE" \
    --n-gpu-layers "$GPU_LAYERS" \
    --flash-attn on \
    --reasoning-budget 0 >/dev/null

echo "Waiting for llama.cpp to become healthy..."
SERVER_READY=false
for _ in $(seq 1 180); do
    if ! docker inspect --format '{{.State.Running}}' "$SERVER_CONTAINER" 2>/dev/null | grep -q true; then
        echo "llama.cpp exited during startup:" >&2
        docker logs "$SERVER_CONTAINER" >&2
        exit 1
    fi
    if docker run --rm \
        --network "$NETWORK_NAME" \
        --entrypoint curl \
        "$PROJECT_IMAGE" \
        --silent --fail --max-time 5 "http://${SERVER_CONTAINER}:8080/health" >/dev/null 2>&1; then
        SERVER_READY=true
        break
    fi
    sleep 10
done
if [[ "$SERVER_READY" != true ]]; then
    echo "llama.cpp did not become healthy within 30 minutes:" >&2
    docker logs "$SERVER_CONTAINER" >&2
    exit 1
fi

RUN_ARGS=(
    python -m moe_exp.experiment0a.run
    --dataset-dir /data/schoenfeld
    --api-base "http://${SERVER_CONTAINER}:8080/v1"
    --api-key "$API_KEY"
    --model local-llamacpp
    --train-documents "$TRAIN_DOCUMENTS"
    --val-documents "$VAL_DOCUMENTS"
    --seed "$SEED"
    --prompt-variant "$PROMPT_VARIANT"
    --few-shot-examples "$FEW_SHOT_EXAMPLES"
    --gepa-reward "$GEPA_REWARD"
    --selection-metric "$SELECTION_METRIC"
    --max-class-recall-drop "$MAX_CLASS_RECALL_DROP"
    --cv-folds "$CV_FOLDS"
    --cv-inner-val-documents "$CV_INNER_VAL_DOCUMENTS"
    --locked-test-documents "$LOCKED_TEST_DOCUMENTS"
    --num-threads "$NUM_THREADS"
    --max-tokens "$MAX_TOKENS"
    --reflection-max-tokens "$REFLECTION_MAX_TOKENS"
    --reflection-temperature "$REFLECTION_TEMPERATURE"
    --output-dir "/workspace/$OUTPUT_DIR"
    "--$BUDGET_KIND" "$BUDGET_VALUE"
)
if [[ "$EVALUATE_LOCKED_TEST" == true ]]; then
    RUN_ARGS+=(--evaluate-locked-test)
fi
if [[ -n "$TEST_DOCUMENTS" ]]; then
    RUN_ARGS+=(--test-documents "$TEST_DOCUMENTS")
fi
if [[ -n "$MAX_UNITS" ]]; then
    RUN_ARGS+=(--max-units-per-document "$MAX_UNITS")
fi

echo "llama.cpp is ready; starting GEPA."
docker run --rm \
    --network "$NETWORK_NAME" \
    --memory="$RUNNER_MEMORY" \
    -v "$PHYS_DIR":/workspace \
    -v "$DATASET_DIR":/data/schoenfeld:ro \
    "$PROJECT_IMAGE" \
    "${RUN_ARGS[@]}"

echo "=== Experiment 0a complete ==="
echo "Results: $PHYS_DIR/$OUTPUT_DIR"

#!/bin/bash
#SBATCH --job-name=moe-pipeline
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --nodelist=faretra

# Full pipeline: Exp1 → Exp2 → Event Routing → Exp3
# Usage:
#   ./run_pipeline.sh --model allenai/OLMoE-1B-7B-0924-Instruct --dataset gsm8k [--max-items 50]
#   sbatch run_pipeline.sh --model allenai/OLMoE-1B-7B-0924-Instruct --dataset gsm8k

set -euo pipefail

PHYS_DIR="/home/tassinari/moe-mfaExperiments"
LLM_CACHE_DIR="/llms"
IMAGE_NAME="moe-mfa-experiments:latest"

# --- Parse arguments ---
MODEL=""
DATASET=""
MAX_ITEMS=""
OUTPUT_DIR="results"
TOP_K=8
WINDOW=5
SAMPLES=1000
CHUNK_SIZE=20
SELF_CHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --max-items) MAX_ITEMS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --window) WINDOW="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        --self-check) SELF_CHECK=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" || -z "$DATASET" ]]; then
    echo "Usage: $0 --model <HF_MODEL_ID> --dataset <DATASET> [--max-items N] [--self-check]"
    echo "  Datasets: gsm8k, math, processbench, prm800k"
    echo "  --self-check: use the self-checking prompt (generation datasets only;"
    echo "                runs as <dataset>_selfcheck through all stages)"
    exit 1
fi

# Self-check uses a verification-encouraging prompt and only applies to generation
# datasets. Exp1 writes its output under "<dataset>_selfcheck", so the whole
# downstream chain must use that name for its paths.
DATASET_DIR="$DATASET"
SELFCHECK_FLAG=""
if [[ "$SELF_CHECK" == true ]]; then
    case "$DATASET" in
        processbench|prm800k)
            echo "Error: --self-check does not apply to given-solution dataset '$DATASET'"
            echo "       (its chain is pre-written; there is nothing to generate)."
            exit 1 ;;
    esac
    DATASET_DIR="${DATASET}_selfcheck"
    SELFCHECK_FLAG="--self-check"
fi

MODEL_SLUG="${MODEL////--}"
BASE_DIR="${OUTPUT_DIR}/${MODEL_SLUG}/${DATASET_DIR}"

# Pre-create output dirs
mkdir -p "$PHYS_DIR/$BASE_DIR"
chmod -R 777 "$PHYS_DIR/results"

echo "=== MoE Pipeline ==="
echo "  Model:      $MODEL"
echo "  Dataset:    $DATASET"
echo "  Self-check: $SELF_CHECK"
echo "  Output:     $BASE_DIR"
echo ""

# --- Build common docker run prefix ---
DOCKER_RUN="docker run \
    -v $PHYS_DIR:/workspace \
    -v $LLM_CACHE_DIR:$LLM_CACHE_DIR \
    -e HF_HOME=$LLM_CACHE_DIR \
    --rm \
    --memory=30g \
    --gpus '\"device='$CUDA_VISIBLE_DEVICES'\"' \
    $IMAGE_NAME \
    bash -c"

run_in_docker() {
    docker run \
        -v "$PHYS_DIR":/workspace \
        -v "$LLM_CACHE_DIR":"$LLM_CACHE_DIR" \
        -e HF_HOME="$LLM_CACHE_DIR" \
        --rm \
        --memory="30g" \
        --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
        "$IMAGE_NAME" \
        bash -c "cd /workspace && $1"
}

# --- Build max-items flag ---
ITEMS_FLAG=""
if [[ -n "$MAX_ITEMS" ]]; then
    ITEMS_FLAG="--max-items $MAX_ITEMS"
fi

LIMIT_FLAG=""
if [[ -n "$MAX_ITEMS" ]]; then
    LIMIT_FLAG="--limit $MAX_ITEMS"
fi

# ==========================================================================
# Stage 1: Experiment 1 — CoT trace generation
# ==========================================================================
TRACES_PATH="${BASE_DIR}/traces.jsonl"

echo ">>> Stage 1/4: Experiment 1 — CoT Trace Generation"
if [[ -f "$PHYS_DIR/$TRACES_PATH" ]]; then
    echo "    traces.jsonl already exists, skipping. Delete to re-run."
else
    run_in_docker "python -m moe_exp.experiment1.run \
        --model $MODEL \
        --datasets $DATASET \
        --output-dir $OUTPUT_DIR \
        $SELFCHECK_FLAG \
        $ITEMS_FLAG"
fi
echo ""

# ==========================================================================
# Stage 2: Experiment 2 — Router logit extraction
# ==========================================================================
ROUTING_PATH="${BASE_DIR}/traces_with_routing.jsonl"

echo ">>> Stage 2/4: Experiment 2 — Router Logit Extraction"
if [[ -f "$PHYS_DIR/$ROUTING_PATH" ]]; then
    echo "    traces_with_routing.jsonl already exists, skipping. Delete to re-run."
else
    run_in_docker "python -m moe_exp.experiment2.run \
        --input $TRACES_PATH \
        --output $ROUTING_PATH \
        --model_id $MODEL \
        --top-k $TOP_K \
        $LIMIT_FLAG"
fi
echo ""

# ==========================================================================
# Stage 3: Event routing analysis
# ==========================================================================
EVENT_ROUTING_PATH="${BASE_DIR}/event_routing.json"

echo ">>> Stage 3/4: Event Routing Analysis"
if [[ -f "$PHYS_DIR/$EVENT_ROUTING_PATH" ]]; then
    echo "    event_routing.json already exists, skipping. Delete to re-run."
else
    run_in_docker "python -m moe_exp.analysis.event_routing \
        --input $ROUTING_PATH \
        --output $EVENT_ROUTING_PATH \
        --window $WINDOW \
        --model_id $MODEL \
        $LIMIT_FLAG"
fi
echo ""

# ==========================================================================
# Stage 4: Experiment 3 — Geometric routing correlation
# ==========================================================================
GEOMETRY_PATH="${BASE_DIR}/geometry_correlation.json"

echo ">>> Stage 4/4: Experiment 3 — Geometric Correlation"
if [[ -f "$PHYS_DIR/$GEOMETRY_PATH" ]]; then
    echo "    geometry_correlation.json already exists, skipping. Delete to re-run."
else
    run_in_docker "python -m moe_exp.experiment3.run \
        --input $TRACES_PATH \
        --output $GEOMETRY_PATH \
        --model_id $MODEL \
        --samples $SAMPLES \
        --chunk-size $CHUNK_SIZE \
        $LIMIT_FLAG"
fi
echo ""

echo "=== Pipeline complete ==="
echo "  Outputs: $PHYS_DIR/$BASE_DIR"

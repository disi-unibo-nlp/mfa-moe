#!/bin/bash
#SBATCH --job-name=moe-pipeline
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --nodelist=faretra

# Full pipeline: Exp1 → Exp2 → Event Routing → Exp3 → Exp4 → Exp5
# Usage:
#   ./run_pipeline.sh --model allenai/OLMoE-1B-7B-0924-Instruct --dataset gsm8k [--max-items 50]
#   sbatch run_pipeline.sh --model allenai/OLMoE-1B-7B-0924-Instruct --dataset gsm8k
#   ./run_pipeline.sh --local --model ... --dataset gsm8k   # no Docker (e.g. vast.ai)

set -euo pipefail

# Cluster paths (Docker mode). In --local mode PHYS_DIR becomes the repo root.
PHYS_DIR="/home/tassinari/moe-mfaExperiments"
LLM_CACHE_DIR="/llms"
IMAGE_NAME="moe-mfa-experiments:latest"

# GPU selection: set by SLURM; default to device 0 for direct invocation.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Parse arguments ---
MODEL=""
DATASET=""
MAX_ITEMS=""
OUTPUT_DIR="results"
# Empty = let each stage infer top-k from the model config (num_experts_per_tok),
# so Exp2 and event_routing always agree (OLMoE=8, Qwen1.5-MoE=4, ...).
TOP_K=""
WINDOW=5
SAMPLES=1000
CHUNK_SIZE=20
SELF_CHECK=false
LOCAL=false
SKIP_EXP4=false
PROBE_FOLDS=5
PROBE_BOOTSTRAP=500

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
        --local) LOCAL=true; shift ;;
        --skip-exp4) SKIP_EXP4=true; shift ;;
        --probe-folds) PROBE_FOLDS="$2"; shift 2 ;;
        --probe-bootstrap) PROBE_BOOTSTRAP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" || -z "$DATASET" ]]; then
    echo "Usage: $0 --model <HF_MODEL_ID> --dataset <DATASET> [--max-items N] [--self-check] [--local]"
    echo "  Datasets: gsm8k, math, processbench, prm800k"
    echo "  --self-check: use the self-checking prompt (generation datasets only;"
    echo "                runs as <dataset>_selfcheck through all stages)"
    echo "  --local:      run stages natively instead of in Docker (for environments"
    echo "                that are already containers, e.g. vast.ai instances)"
    echo "  --skip-exp4:  skip hidden-state storage and prospective prefix probes"
    echo "  --probe-folds N: cross-validation folds for Experiment 4 (default: 5)"
    echo "  --probe-bootstrap N: trace-bootstrap replicates for Experiment 4 (default: 500)"
    exit 1
fi

# Local mode: run stages natively from the repo root. Auto-enabled when docker
# is unavailable (a vast.ai instance is already a container).
if [[ "$LOCAL" != true ]] && ! command -v docker >/dev/null 2>&1; then
    echo "docker not found — falling back to --local mode."
    LOCAL=true
fi
if [[ "$LOCAL" == true ]]; then
    PHYS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
# Per-experiment layout, matching the README and manual runs:
#   results/exp1/<model>/<dataset>/traces.jsonl
#   results/exp2/<model>/<dataset>/traces_with_routing.jsonl + event_routing.json
#   results/exp3/<model>/<dataset>/geometry_correlation.json
#   results/exp4/<model>/<dataset>/prospective_probes.json
#   results/exp5/<model>/<dataset>/expert_events.json + expert_arrays.npz
EXP1_DIR="${OUTPUT_DIR}/exp1/${MODEL_SLUG}/${DATASET_DIR}"
EXP2_DIR="${OUTPUT_DIR}/exp2/${MODEL_SLUG}/${DATASET_DIR}"
EXP3_DIR="${OUTPUT_DIR}/exp3/${MODEL_SLUG}/${DATASET_DIR}"
EXP4_DIR="${OUTPUT_DIR}/exp4/${MODEL_SLUG}/${DATASET_DIR}"
EXP5_DIR="${OUTPUT_DIR}/exp5/${MODEL_SLUG}/${DATASET_DIR}"

# Pre-create output dirs
mkdir -p "$PHYS_DIR/$EXP1_DIR" "$PHYS_DIR/$EXP2_DIR" "$PHYS_DIR/$EXP3_DIR" "$PHYS_DIR/$EXP4_DIR" "$PHYS_DIR/$EXP5_DIR"
if [[ "$LOCAL" != true ]]; then
    # NFS root_squash maps container root → nobody; open up perms so it can write.
    chmod -R 777 "$PHYS_DIR/$OUTPUT_DIR"
fi

echo "=== MoE Pipeline ==="
echo "  Model:      $MODEL"
echo "  Dataset:    $DATASET"
echo "  Self-check: $SELF_CHECK"
echo "  Mode:       $([[ "$LOCAL" == true ]] && echo local || echo docker)"
echo "  Exp4:       $([[ "$SKIP_EXP4" == true ]] && echo skipped || echo enabled)"
echo "  Output:     ${OUTPUT_DIR}/exp{1,2,3,4,5}/${MODEL_SLUG}/${DATASET_DIR}"
echo ""

run_stage() {
    if [[ "$LOCAL" == true ]]; then
        (cd "$PHYS_DIR" && bash -c "$1")
    else
        docker run \
            -v "$PHYS_DIR":/workspace \
            -v "$LLM_CACHE_DIR":"$LLM_CACHE_DIR" \
            -e HF_HOME="$LLM_CACHE_DIR" \
            --rm \
            --memory="30g" \
            --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
            "$IMAGE_NAME" \
            bash -c "cd /workspace && $1"
    fi
}

require_nonempty_output() {
    local path="$1"
    local stage="$2"
    if [[ ! -s "$PHYS_DIR/$path" ]]; then
        echo "ERROR: $stage did not produce a non-empty output: $PHYS_DIR/$path" >&2
        return 1
    fi
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

# Pass an explicit --top-k to ALL top-k consumers (Exp2, event_routing, Exp5),
# or to none of them, so the stages never disagree on k.
TOPK_FLAG=""
if [[ -n "$TOP_K" ]]; then
    TOPK_FLAG="--top-k $TOP_K"
fi

# Experiment 4 needs token-level hidden states aligned with the router tensors.
# Keep this optional because hidden tensors require substantial disk space.
HIDDEN_FLAG="--extract-hidden-states"
if [[ "$SKIP_EXP4" == true ]]; then
    HIDDEN_FLAG=""
fi

# ==========================================================================
# Stage 1: Experiment 1 — CoT trace generation
# ==========================================================================
TRACES_PATH="${EXP1_DIR}/traces.jsonl"

echo ">>> Stage 1/6: Experiment 1 — CoT Trace Generation"
if [[ -s "$PHYS_DIR/$TRACES_PATH" ]]; then
    echo "    traces.jsonl already exists, skipping. Delete to re-run."
else
    if [[ -e "$PHYS_DIR/$TRACES_PATH" ]]; then
        echo "    existing traces.jsonl is empty; rerunning Experiment 1."
    fi
    run_stage "python -m moe_exp.experiment1.run \
        --model $MODEL \
        --datasets $DATASET \
        --output-dir ${OUTPUT_DIR}/exp1 \
        $SELFCHECK_FLAG \
        $ITEMS_FLAG"
fi
require_nonempty_output "$TRACES_PATH" "Experiment 1"
echo ""

# ==========================================================================
# Stage 2: Experiment 2 — Router logit extraction
# ==========================================================================
ROUTING_PATH="${EXP2_DIR}/traces_with_routing.jsonl"

echo ">>> Stage 2/6: Experiment 2 — Router/Hidden-State Extraction"
if [[ -s "$PHYS_DIR/$ROUTING_PATH" && "$PHYS_DIR/$ROUTING_PATH" -nt "$PHYS_DIR/$TRACES_PATH" ]]; then
    echo "    traces_with_routing.jsonl already exists, skipping. Delete to re-run."
else
    run_stage "python -m moe_exp.experiment2.run \
        --input $TRACES_PATH \
        --output $ROUTING_PATH \
        --model_id $MODEL \
        $HIDDEN_FLAG \
        $TOPK_FLAG \
        $LIMIT_FLAG"
fi
require_nonempty_output "$ROUTING_PATH" "Experiment 2"
echo ""

# ==========================================================================
# Stage 3: Event routing analysis
# ==========================================================================
EVENT_ROUTING_PATH="${EXP2_DIR}/event_routing.json"

echo ">>> Stage 3/6: Event Routing Analysis"
if [[ -s "$PHYS_DIR/$EVENT_ROUTING_PATH" && "$PHYS_DIR/$EVENT_ROUTING_PATH" -nt "$PHYS_DIR/$ROUTING_PATH" ]]; then
    echo "    event_routing.json already exists, skipping. Delete to re-run."
else
    run_stage "python -m moe_exp.analysis.event_routing \
        --input $ROUTING_PATH \
        --output $EVENT_ROUTING_PATH \
        --window $WINDOW \
        --model_id $MODEL \
        $TOPK_FLAG \
        $LIMIT_FLAG"
fi
require_nonempty_output "$EVENT_ROUTING_PATH" "event-routing analysis"
echo ""

# ==========================================================================
# Stage 4: Experiment 3 — Geometric routing correlation
# ==========================================================================
GEOMETRY_PATH="${EXP3_DIR}/geometry_correlation.json"

echo ">>> Stage 4/6: Experiment 3 — Geometric Correlation"
if [[ -s "$PHYS_DIR/$GEOMETRY_PATH" && "$PHYS_DIR/$GEOMETRY_PATH" -nt "$PHYS_DIR/$TRACES_PATH" ]]; then
    echo "    geometry_correlation.json already exists, skipping. Delete to re-run."
else
    run_stage "python -m moe_exp.experiment3.run \
        --input $TRACES_PATH \
        --output $GEOMETRY_PATH \
        --model_id $MODEL \
        --samples $SAMPLES \
        --chunk-size $CHUNK_SIZE \
        $LIMIT_FLAG"
fi
require_nonempty_output "$GEOMETRY_PATH" "Experiment 3"
echo ""

# ==========================================================================
# Stage 5: Experiment 4 — Prospective prefix-only failure prediction
# ========================================================================== 
PROSPECTIVE_PATH="${EXP4_DIR}/prospective_probes.json"

echo ">>> Stage 5/6: Experiment 4 — Prospective Prefix Probes"
if [[ "$SKIP_EXP4" == true ]]; then
    echo "    skipped by --skip-exp4"
elif [[ -s "$PHYS_DIR/$PROSPECTIVE_PATH" && "$PHYS_DIR/$PROSPECTIVE_PATH" -nt "$PHYS_DIR/$ROUTING_PATH" ]]; then
    echo "    prospective_probes.json already exists, skipping. Delete to re-run."
else
    run_stage "python -m moe_exp.experiment4.run \
        --input $ROUTING_PATH \
        --output $PROSPECTIVE_PATH \
        --model-id $MODEL \
        --folds $PROBE_FOLDS \
        --bootstrap-samples $PROBE_BOOTSTRAP \
        $LIMIT_FLAG"
fi
if [[ "$SKIP_EXP4" != true ]]; then
    require_nonempty_output "$PROSPECTIVE_PATH" "Experiment 4"
fi
echo ""

# ========================================================================== 
# Stage 6: Experiment 5 — Expert behavior around reasoning events
# ========================================================================== 
EXPERT_EVENTS_PATH="${EXP5_DIR}/expert_events.json"

echo ">>> Stage 6/6: Experiment 5 — Expert Behavior Around Events"
if [[ -s "$PHYS_DIR/$EXPERT_EVENTS_PATH" && "$PHYS_DIR/$EXPERT_EVENTS_PATH" -nt "$PHYS_DIR/$ROUTING_PATH" ]]; then
    echo "    expert_events.json already exists, skipping. Delete to re-run."
else
    run_stage "python -m moe_exp.experiment5.run \
        --input $ROUTING_PATH \
        --output $EXPERT_EVENTS_PATH \
        --window $WINDOW \
        --model_id $MODEL \
        $TOPK_FLAG \
        $LIMIT_FLAG"
fi
require_nonempty_output "$EXPERT_EVENTS_PATH" "Experiment 5"
echo ""

echo "=== Pipeline complete ==="
echo "  Outputs:"
echo "    $PHYS_DIR/$EXP1_DIR"
echo "    $PHYS_DIR/$EXP2_DIR"
echo "    $PHYS_DIR/$EXP3_DIR"
echo "    $PHYS_DIR/$EXP4_DIR"
echo "    $PHYS_DIR/$EXP5_DIR"

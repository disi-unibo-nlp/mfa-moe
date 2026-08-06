#!/bin/bash
#SBATCH --job-name=episode-probes
#SBATCH --output=slurm_logs/probeTest-%j.out
#SBATCH --error=slurm_logs/probeTest-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --nodelist=faretra

# Submit from the repository root. The default matches the existing launcher
# for the machine referred to as "40"; Slurm options can override it at submit.

set -euo pipefail

PHYS_DIR="${PHYS_DIR:-/home/tassinari/moe-mfaExperiments}"
DATASET_DIR="${DATASET_DIR:-${PHYS_DIR}/data/Schoenfeld_Reasoning}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/gringotts/hf_home}"
RESULTS_ROOT="${RESULTS_ROOT:-/gringotts/home/tassinari/results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/probeTest/qwen3.6-35b-a3b}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
MODEL_REVISION="${MODEL_REVISION:-main}"
QUANTIZATION="${QUANTIZATION:-bnb-4bit}"
IMAGE_NAME="${IMAGE_NAME:-moe-mfa-experiments:latest}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOCAL=false
MAX_DOCUMENTS=""
INCLUDE_THINK_BOUNDARY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) LOCAL=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --model-revision) MODEL_REVISION="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-documents) MAX_DOCUMENTS="$2"; shift 2 ;;
        --include-think-boundary-units) INCLUDE_THINK_BOUNDARY=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Dataset directory does not exist: $DATASET_DIR" >&2
    exit 1
fi
mkdir -p "$HF_CACHE_DIR" "$OUTPUT_DIR" "$PHYS_DIR/slurm_logs"
export HF_HOME="$HF_CACHE_DIR"

echo "=== Schoenfeld gold boundary probes ==="
echo "  Model:        $MODEL"
echo "  Dataset:      $DATASET_DIR"
echo "  HF_HOME:      $HF_HOME"
echo "  Output:       $OUTPUT_DIR"
echo "  Quantization: $QUANTIZATION"
echo "  GPU:          $CUDA_VISIBLE_DEVICES"
echo "  Node:         ${SLURMD_NODENAME:-pending Slurm assignment}"

EXTRA_ARGS=()
if [[ -n "$MAX_DOCUMENTS" ]]; then
    EXTRA_ARGS+=(--max-documents "$MAX_DOCUMENTS")
fi
if [[ "$INCLUDE_THINK_BOUNDARY" == true ]]; then
    EXTRA_ARGS+=(--include-think-boundary-units)
fi

if [[ "$LOCAL" == true ]] || ! command -v docker >/dev/null 2>&1; then
    cd "$PHYS_DIR"
    python -m moe_exp.probeTest.run all \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL" \
        --model-revision "$MODEL_REVISION" \
        --quantization "$QUANTIZATION" \
        "${EXTRA_ARGS[@]}"
else
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo "Missing Docker image: $IMAGE_NAME" >&2
        echo "Build it from the repository root: docker build -t $IMAGE_NAME ." >&2
        exit 1
    fi
    DOCKER_ENV=(-e HF_HOME=/hf_home -e HOME=/tmp)
    if [[ -n "${HF_TOKEN:-}" ]]; then
        DOCKER_ENV+=(-e HF_TOKEN="$HF_TOKEN")
    fi
    DOCKER_EXTRA=()
    if [[ -n "$MAX_DOCUMENTS" ]]; then
        DOCKER_EXTRA+=(--max-documents "$MAX_DOCUMENTS")
    fi
    if [[ "$INCLUDE_THINK_BOUNDARY" == true ]]; then
        DOCKER_EXTRA+=(--include-think-boundary-units)
    fi
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        --gpus "device=$CUDA_VISIBLE_DEVICES" \
        --ipc=host \
        --memory=64g \
        -v "$PHYS_DIR":/workspace:ro \
        -v "$DATASET_DIR":/data/schoenfeld:ro \
        -v "$HF_CACHE_DIR":/hf_home \
        -v "$OUTPUT_DIR":/output \
        "${DOCKER_ENV[@]}" \
        "$IMAGE_NAME" \
        python -m moe_exp.probeTest.run all \
            --dataset-dir /data/schoenfeld \
            --output-dir /output \
            --model "$MODEL" \
            --model-revision "$MODEL_REVISION" \
            --quantization "$QUANTIZATION" \
            "${DOCKER_EXTRA[@]}"
fi

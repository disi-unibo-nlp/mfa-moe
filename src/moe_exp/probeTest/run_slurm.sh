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
HF_CACHE_DIR="${HF_CACHE_DIR:-/llms}"
RESULTS_ROOT="${RESULTS_ROOT:-${PHYS_DIR}/results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/probeTest/qwen3.5-35b-a3b-gptq-int4}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}"
MODEL_REVISION="${MODEL_REVISION:-main}"
QUANTIZATION="${QUANTIZATION:-gptq-4bit}"
IMAGE_NAME="${IMAGE_NAME:-moe-mfa-experiments:latest}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOCAL=false
LABEL_ONLY=false
SKIP_BENCHMARK_LABELING=false
MAX_DOCUMENTS=""
EXAMPLES_PER_BENCHMARK="${EXAMPLES_PER_BENCHMARK:-20}"
MAX_BENCHMARK_INPUT_TOKENS="${MAX_BENCHMARK_INPUT_TOKENS:-4096}"
INCLUDE_THINK_BOUNDARY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) LOCAL=true; shift ;;
        --label-only) LABEL_ONLY=true; shift ;;
        --skip-benchmark-labeling) SKIP_BENCHMARK_LABELING=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --model-revision) MODEL_REVISION="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-documents) MAX_DOCUMENTS="$2"; shift 2 ;;
        --examples-per-benchmark) EXAMPLES_PER_BENCHMARK="$2"; shift 2 ;;
        --max-benchmark-input-tokens) MAX_BENCHMARK_INPUT_TOKENS="$2"; shift 2 ;;
        --include-think-boundary-units) INCLUDE_THINK_BOUNDARY=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Dataset directory does not exist: $DATASET_DIR" >&2
    exit 1
fi
mkdir -p "$HF_CACHE_DIR" "$OUTPUT_DIR" "$PHYS_DIR/slurm_logs"
for writable_dir in "$HF_CACHE_DIR" "$OUTPUT_DIR"; do
    if [[ ! -w "$writable_dir" ]]; then
        echo "Directory is not writable by $(id -un) (uid=$(id -u)): $writable_dir" >&2
        echo "Fix its ownership/permissions or choose another path before submitting." >&2
        exit 1
    fi
done
export HF_HOME="$HF_CACHE_DIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_CACHE_DIR}/.cache}"

echo "=== Schoenfeld gold boundary probes ==="
echo "  Model:        $MODEL"
echo "  Dataset:      $DATASET_DIR"
echo "  HF_HOME:      $HF_HOME"
echo "  Output:       $OUTPUT_DIR"
echo "  Quantization: $QUANTIZATION"
echo "  GPU:          $CUDA_VISIBLE_DEVICES"
echo "  Benchmark N:  $EXAMPLES_PER_BENCHMARK"
echo "  Label only:   $LABEL_ONLY"
echo "  Node:         ${SLURMD_NODENAME:-pending Slurm assignment}"

EXTRA_ARGS=()
if [[ -n "$MAX_DOCUMENTS" ]]; then
    EXTRA_ARGS+=(--max-documents "$MAX_DOCUMENTS")
fi
if [[ "$INCLUDE_THINK_BOUNDARY" == true ]]; then
    EXTRA_ARGS+=(--include-think-boundary-units)
fi
EXTRA_ARGS+=(
    --examples-per-benchmark "$EXAMPLES_PER_BENCHMARK"
    --max-benchmark-input-tokens "$MAX_BENCHMARK_INPUT_TOKENS"
)
if [[ "$SKIP_BENCHMARK_LABELING" == true ]]; then
    EXTRA_ARGS+=(--skip-benchmark-labeling)
fi

if [[ "$LOCAL" == true ]] || ! command -v docker >/dev/null 2>&1; then
    cd "$PHYS_DIR"
    if [[ "$LABEL_ONLY" == true ]]; then
        python -m moe_exp.probeTest.run label \
            --probe-results "$OUTPUT_DIR/probes/results.json" \
            --output-dir "$OUTPUT_DIR/benchmark_labels" \
            --examples-per-benchmark "$EXAMPLES_PER_BENCHMARK" \
            --max-benchmark-input-tokens "$MAX_BENCHMARK_INPUT_TOKENS"
    else
        python -m moe_exp.probeTest.run all \
            --dataset-dir "$DATASET_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --model "$MODEL" \
            --model-revision "$MODEL_REVISION" \
            --quantization "$QUANTIZATION" \
            "${EXTRA_ARGS[@]}"
    fi
else
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo "Missing Docker image: $IMAGE_NAME" >&2
        echo "Build it from the repository root: docker build -t $IMAGE_NAME ." >&2
        exit 1
    fi
    DOCKER_ENV=(
        -e HF_HOME="$HF_CACHE_DIR"
        -e HOME="$HF_CACHE_DIR"
        -e XDG_CACHE_HOME="$XDG_CACHE_HOME"
    )
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
    DOCKER_EXTRA+=(
        --examples-per-benchmark "$EXAMPLES_PER_BENCHMARK"
        --max-benchmark-input-tokens "$MAX_BENCHMARK_INPUT_TOKENS"
    )
    if [[ "$SKIP_BENCHMARK_LABELING" == true ]]; then
        DOCKER_EXTRA+=(--skip-benchmark-labeling)
    fi
    HOST_UID="$(id -u)"
    HOST_GID="$(id -g)"
    DOCKER_USER=(--user "$HOST_UID:$HOST_GID")
    DOCKER_GROUPS=()
    if [[ "$(docker info --format '{{json .SecurityOptions}}')" == *rootless* ]]; then
        # Root in a rootless container maps to the unprivileged daemon user on
        # the host.  Using the host's numeric UID here would map a different,
        # unprivileged container user and make bind mounts appear root-owned.
        DOCKER_USER=(--user 0:0)
    else
        # Rootful Docker drops supplementary groups when --user is used.
        # Shared HPC filesystems may grant access through one of those groups.
        read -r -a HOST_GROUP_IDS <<< "$(id -G)"
        for group_id in "${HOST_GROUP_IDS[@]}"; do
            if [[ "$group_id" != "$HOST_GID" ]]; then
                DOCKER_GROUPS+=(--group-add "$group_id")
            fi
        done
    fi
    DOCKER_RESOURCE_ARGS=()
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        DOCKER_RESOURCE_ARGS+=(--memory=64g)
    fi
    DOCKER_COMMAND=(
        python -m moe_exp.probeTest.run all
        --dataset-dir /data/schoenfeld
        --output-dir /output
        --model "$MODEL"
        --model-revision "$MODEL_REVISION"
        --quantization "$QUANTIZATION"
        "${DOCKER_EXTRA[@]}"
    )
    if [[ "$LABEL_ONLY" == true ]]; then
        DOCKER_COMMAND=(
            python -m moe_exp.probeTest.run label
            --probe-results /output/probes/results.json
            --output-dir /output/benchmark_labels
            --examples-per-benchmark "$EXAMPLES_PER_BENCHMARK"
            --max-benchmark-input-tokens "$MAX_BENCHMARK_INPUT_TOKENS"
        )
    fi
    docker run --rm \
        "${DOCKER_USER[@]}" \
        "${DOCKER_GROUPS[@]}" \
        --gpus "device=$CUDA_VISIBLE_DEVICES" \
        --ipc=host \
        "${DOCKER_RESOURCE_ARGS[@]}" \
        -v "$PHYS_DIR":/workspace:ro \
        -v "$DATASET_DIR":/data/schoenfeld:ro \
        -v "$HF_CACHE_DIR":"$HF_CACHE_DIR" \
        -v "$OUTPUT_DIR":/output \
        "${DOCKER_ENV[@]}" \
        "$IMAGE_NAME" \
        "${DOCKER_COMMAND[@]}"
fi

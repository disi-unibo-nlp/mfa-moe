#!/usr/bin/env bash
set -euo pipefail

# Run correlation stages inside the project Python 3.11 image. Generation uses
# host networking to reach the separately running llama.cpp server on port 8080.

PHYS_DIR="${PHYS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/llms}"
IMAGE_NAME="${IMAGE_NAME:-moe-mfa-experiments:latest}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-false}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <generate|forward|analyze> [stage arguments...]" >&2
    exit 2
fi

STAGE="$1"
shift
case "$STAGE" in
    generate)
        MODULE="moe_exp.correlation_pipeline.generate"
        ;;
    forward)
        MODULE="moe_exp.correlation_pipeline.extract"
        ;;
    analyze)
        MODULE="moe_exp.correlation_pipeline.analyze"
        ;;
    *)
        echo "Unknown stage: $STAGE (choose generate, forward, or analyze)" >&2
        exit 2
        ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required because host Python 3.10 is below the project's Python 3.11 minimum." >&2
    exit 1
fi
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Missing Docker image: $IMAGE_NAME" >&2
    echo "Build it from the repository root: docker build -t $IMAGE_NAME ." >&2
    exit 1
fi
if ! docker run --rm "$IMAGE_NAME" python -c "import math_verify" >/dev/null 2>&1; then
    echo "Docker image $IMAGE_NAME predates the correlation dependencies." >&2
    echo "Rebuild it from the repository root: docker build -t $IMAGE_NAME ." >&2
    exit 1
fi
if [[ "$STAGE" == "forward" ]] && ! docker run --rm "$IMAGE_NAME" python -c \
    "import importlib.util; assert importlib.util.find_spec('unsloth')" >/dev/null 2>&1; then
    echo "Docker image $IMAGE_NAME does not contain the Unsloth forward backend." >&2
    echo "Rebuild it from the repository root: docker build -t $IMAGE_NAME ." >&2
    exit 1
fi

mkdir -p "$HF_CACHE_DIR" "$PHYS_DIR/results/correlation_pipeline"
mkdir -p "$HF_CACHE_DIR/.cache/unsloth_compiled_cache" "$HF_CACHE_DIR/.cache/torchinductor"
for writable_dir in "$HF_CACHE_DIR" "$PHYS_DIR/results/correlation_pipeline"; do
    if [[ ! -w "$writable_dir" ]]; then
        echo "Directory is not writable by $(id -un) (uid=$(id -u)): $writable_dir" >&2
        exit 1
    fi
done

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
DOCKER_USER=(--user "$HOST_UID:$HOST_GID")
DOCKER_GROUPS=()
if [[ "$(docker info --format '{{json .SecurityOptions}}')" == *rootless* ]]; then
    DOCKER_USER=(--user 0:0)
else
    read -r -a HOST_GROUP_IDS <<< "$(id -G)"
    for group_id in "${HOST_GROUP_IDS[@]}"; do
        if [[ "$group_id" != "$HOST_GID" ]]; then
            DOCKER_GROUPS+=(--group-add "$group_id")
        fi
    done
fi

DOCKER_ARGS=(
    docker run --rm
    "${DOCKER_USER[@]}"
    "${DOCKER_GROUPS[@]}"
    -v "$PHYS_DIR:/workspace"
    -v "$HF_CACHE_DIR:$HF_CACHE_DIR"
    -e "HF_HOME=$HF_CACHE_DIR"
    -e "HOME=$HF_CACHE_DIR"
    -e "XDG_CACHE_HOME=$HF_CACHE_DIR/.cache"
    -e "UNSLOTH_DISABLE_STATISTICS=1"
    -e "UNSLOTH_COMPILE_LOCATION=$HF_CACHE_DIR/.cache/unsloth_compiled_cache"
    -e "TORCHINDUCTOR_CACHE_DIR=$HF_CACHE_DIR/.cache/torchinductor"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "PYTHONPATH=/workspace/src"
    -w /workspace
)
if [[ -n "${HF_TOKEN:-}" ]]; then
    DOCKER_ARGS+=(-e "HF_TOKEN=$HF_TOKEN")
fi

case "$STAGE" in
    generate)
        # The OpenAI-compatible llama.cpp endpoint listens on host localhost.
        DOCKER_ARGS+=(--network host)
        ;;
    forward)
        DOCKER_ARGS+=(--gpus "device=$CUDA_VISIBLE_DEVICES" --ipc=host)
        if [[ -z "${SLURM_JOB_ID:-}" ]]; then
            DOCKER_ARGS+=(--memory=64g)
        fi
        ;;
esac

COMMAND=(python -m "$MODULE" "$@")
FULL_COMMAND=("${DOCKER_ARGS[@]}" "$IMAGE_NAME" "${COMMAND[@]}")

echo "=== Correlation pipeline: $STAGE ==="
echo "  Image:    $IMAGE_NAME"
echo "  Python:   container Python 3.11"
echo "  Workspace: $PHYS_DIR"
echo "  HF_HOME:  $HF_CACHE_DIR"
if [[ "$STAGE" == "forward" ]]; then
    echo "  GPU:      $CUDA_VISIBLE_DEVICES"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    printf '%q ' "${FULL_COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${FULL_COMMAND[@]}"
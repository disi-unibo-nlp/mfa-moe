#!/usr/bin/env bash
set -euo pipefail

PHYS_DIR="${PHYS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/llms}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}"
IMAGE_NAME="${IMAGE_NAME:-moe-guiding:vllm-0.29.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-false}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <generate|sanity> [arguments...]" >&2
    exit 2
fi
case "$1" in
    generate|sanity) ;;
    *) echo "Unknown command: $1 (choose generate or sanity)" >&2; exit 2 ;;
esac

BUILD_COMMAND=(docker build --build-arg "VLLM_IMAGE=$VLLM_IMAGE"
    -f "$PHYS_DIR/src/moe_exp/moe_guiding/Dockerfile" -t "$IMAGE_NAME" "$PHYS_DIR")
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
DOCKER_USER=(--user "$HOST_UID:$HOST_GID")
DOCKER_GROUPS=()
if [[ "$DRY_RUN" != true ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker is required; host Python is not used." >&2
        exit 1
    fi
    # Check the daemon separately so a permissions error is not treated as a missing image.
    SECURITY_OPTIONS="$(docker info --format '{{json .SecurityOptions}}')"
    if [[ "$SECURITY_OPTIONS" == *rootless* ]]; then
        DOCKER_USER=(--user 0:0)
    else
        read -r -a HOST_GROUP_IDS <<< "$(id -G)"
        for group_id in "${HOST_GROUP_IDS[@]}"; do
            if [[ "$group_id" != "$HOST_GID" ]]; then
                DOCKER_GROUPS+=(--group-add "$group_id")
            fi
        done
    fi
    mkdir -p "$HF_CACHE_DIR" "$PHYS_DIR/results/moe_guiding"
    for writable_dir in "$HF_CACHE_DIR" "$PHYS_DIR/results/moe_guiding"; do
        if [[ ! -w "$writable_dir" ]]; then
            echo "Directory is not writable by $(id -un): $writable_dir" >&2
            exit 1
        fi
    done
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo "Building $IMAGE_NAME from $VLLM_IMAGE (first run only)."
        "${BUILD_COMMAND[@]}"
    fi
fi

DOCKER_ARGS=(docker run --rm "${DOCKER_USER[@]}" "${DOCKER_GROUPS[@]}"
    -v "$PHYS_DIR:/workspace" -v "$HF_CACHE_DIR:$HF_CACHE_DIR"
    -e "HF_HOME=$HF_CACHE_DIR" -e "HOME=$HF_CACHE_DIR"
    -e "XDG_CACHE_HOME=$HF_CACHE_DIR/.cache"
    -e PYTHONPATH=/workspace/src -e PYTHONDONTWRITEBYTECODE=1
    -e VLLM_PLUGINS=moe_guiding -w /workspace)
if [[ -n "${HF_TOKEN:-}" ]]; then
    DOCKER_ARGS+=(-e HF_TOKEN)
fi
NEEDS_GPU=false
PREVIOUS_ARGUMENT=""
for argument in "$@"; do
    if [[ "$argument" == --device=cuda || ( "$PREVIOUS_ARGUMENT" == --device && "$argument" == cuda ) ]]; then
        NEEDS_GPU=true
    fi
    PREVIOUS_ARGUMENT="$argument"
done
if [[ "$1" == generate || "$NEEDS_GPU" == true ]]; then
    DOCKER_ARGS+=(--gpus "\"device=$CUDA_VISIBLE_DEVICES\"" --ipc=host)
fi
FULL_COMMAND=("${DOCKER_ARGS[@]}" --entrypoint python3 "$IMAGE_NAME"
    -m moe_exp.moe_guiding.run "$@")

echo "MoE guiding: image=$IMAGE_NAME, workspace=$PHYS_DIR, HF_HOME=$HF_CACHE_DIR"
if [[ "$DRY_RUN" == true ]]; then
    echo "Build command (used when the image is missing):"
    printf '%q ' "${BUILD_COMMAND[@]}"
    printf '\n'
    echo "Run command (user mapping is resolved against Docker at runtime):"
    printf '%q ' "${FULL_COMMAND[@]}"
    printf '\n'
    exit 0
fi
exec "${FULL_COMMAND[@]}"

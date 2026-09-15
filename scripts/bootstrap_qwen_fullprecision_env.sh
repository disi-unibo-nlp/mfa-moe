#!/usr/bin/env bash
set -euo pipefail

P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
REPO="$P/repo"
UV="${UV:-/leonardo/home/userexternal/lmolfett/.local/bin/uv}"
BASE_PY="$P/envs/vllm-cu129/bin/python"
BASE_SITE="$P/envs/vllm-cu129/lib/python3.11/site-packages"
SCRATCH_ROOT=/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke
LOCK="$REPO/requirements/qwen-fullprecision-smoke.lock"
ENV_DIR="$SCRATCH_ROOT/envs"

module load python/3.11.7 cuda/12.6

test -x "$UV"
test -x "$BASE_PY"
test -d "$BASE_SITE"
test -s "$LOCK"

LOCK_SHA256="$(sha256sum "$LOCK" | cut -d' ' -f1)"
ENV_ROOT="$ENV_DIR/qwen-fp-fa-dspy-${LOCK_SHA256:0:16}"
POINTER="$ENV_DIR/qwen-fp-fa-dspy-current.path"
mkdir -p "$ENV_DIR"

if [[ -e "$ENV_ROOT" ]]; then
    test -x "$ENV_ROOT/bin/python" || {
        printf 'existing environment is incomplete: %s\n' "$ENV_ROOT" >&2
        exit 1
    }
    test -s "$ENV_ROOT/.lock.sha256"
    [[ "$(<"$ENV_ROOT/.lock.sha256")" == "$LOCK_SHA256" ]] || {
        printf 'existing environment lock digest mismatch: %s\n' "$ENV_ROOT" >&2
        exit 1
    }
else
    "$UV" venv --python "$BASE_PY" --system-site-packages "$ENV_ROOT"
    "$UV" pip install \
        --python "$ENV_ROOT/bin/python" \
        --require-hashes \
        --requirement "$LOCK"
    printf '%s\n' "$LOCK_SHA256" >"$ENV_ROOT/.lock.sha256"
fi

export PYTHONPATH="$ENV_ROOT/lib/python3.11/site-packages:$BASE_SITE:$REPO/src"
"$ENV_ROOT/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys

import dspy
import torch
import vllm
from dspy.teleprompt import GEPA

print(f"overlay_python_ready={sys.executable}")
print(f"torch_version={torch.__version__}")
print(f"vllm_version={getattr(vllm, '__version__', 'unknown')}")
print(f"dspy_version={metadata.version('dspy')}")
print(f"gepa_version={metadata.version('gepa')}")
print(f"gepa_class={GEPA.__module__}.{GEPA.__name__}")
PY

POINTER_TMP="$POINTER.$$"
printf '%s\n' "$ENV_ROOT" >"$POINTER_TMP"
mv -f "$POINTER_TMP" "$POINTER"
printf 'qwen_dspy_env_ready=%s lock_sha256=%s\n' "$ENV_ROOT" "$LOCK_SHA256"

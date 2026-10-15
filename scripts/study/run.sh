#!/usr/bin/env bash
set -euo pipefail
study_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$study_repo/src:$study_repo/scripts/experimental_resume${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -m moe_exp.study "$@"

#!/bin/bash
set -euo pipefail
module load python/3.11.7
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
STAGE=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/native-finalization-tests
export TMPDIR="$STAGE" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/scripts/experimental_resume:$REPO/src:/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/code/s1-9a61e32f48c04c24"
exec "$REPO/.venv-native-finalization-tests/bin/python" -B -m pytest -q -p no:cacheprovider --basetemp="$STAGE/pytest-login" "$REPO/tests/experimental_resume/test_native_finalization_v1.py"

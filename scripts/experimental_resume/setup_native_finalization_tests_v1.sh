#!/bin/bash
# Test-only overlay: scientific inference environment is read-only.
set -euo pipefail
[[ "$(id -un)" == lmolfett && "$(hostname -f)" == *.leonardo.local ]] || exit 2
module load python/3.11.7
umask 077
STAGING=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/native-finalization-tests
TEST_ENV=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/.venv-native-finalization-tests
SCI=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/lib/python3.11/site-packages
mkdir -p "$STAGING"
export TMPDIR="$STAGING" PIP_CACHE_DIR="$STAGING/pip-cache" PYTHONDONTWRITEBYTECODE=1
python -m venv --system-site-packages "$TEST_ENV"
printf '%s\n' "$SCI" > "$TEST_ENV/lib/python3.11/site-packages/scientific.pth"
"$TEST_ENV/bin/python" -m pip install 'pytest>=8.0.0'
"$TEST_ENV/bin/python" -B -c 'import importlib.metadata as m; import numpy; print("pytest", m.version("pytest"), "numpy", numpy.__file__)'

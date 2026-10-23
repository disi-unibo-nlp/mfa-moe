#!/bin/bash
# Seal a whole-reasoning relink release (the final combine rerun under the eight-model efficiency link,
# no steering) and submit its graph: allocated tests -> combine, after the parent's combine.
#   scripts/study/whole_relink.sh NAME LOCAL_TESTS PARENT_COMBINE_JOB
#   e.g. scripts/study/whole_relink.sh whole-reasoning-20261021-r19 120 61927867
# Needs a clean src/tests/scripts tree (commit first) and driver/EFFICIENCY_LINK_V2.json.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
name="${1:?release name}"; tests="${2:?locally passed tests}"; after="${3:?parent combine job ID}"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
TMP=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/claude-review/tmp
mkdir -p "$TMP"

# 1. Seal: receipts, RELINK.json, SOURCES.json (the parent's), the code snapshot and launch.env.
TMPDIR="$TMP" PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -B -m moe_exp.steering.whole_relink seal --name "$name" --local-tests "$tests"
set -a; source "report/$name/launch.env"; set +a
REL=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/releases/$COMMIT

# 2. Submit from the snapshot.
(cd "$REL" && env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$REL/src TMPDIR="$TMP" \
  STUDY_DATA_ROOT=$REPO/report/final-routing-20261015-r1 WHOLE_ROOT=$REPO/report/$name WHOLE_MODELS=$MODELS \
  WHOLE_CPU_CEILING=$CPU_CEILING WHOLE_GPU_CEILING=0 WHOLE_LINK=$LINK OMP_NUM_THREADS=1 \
  "$PY" -B -m moe_exp.steering.whole_relink launch --after "$after")

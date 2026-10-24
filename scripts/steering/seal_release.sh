#!/bin/bash
# Seal a whole-reasoning style release from the committed code (nothing is submitted):
#   scripts/steering/seal_release.sh KIND NAME TESTS [OPTION...]
# KIND is complement, relink, outcome, capture, sweep or continuation; NAME the new release (report/NAME); TESTS the
# number of tests that passed locally. OPTIONs go to steering.release: --cutoff TIME (every kind but relink: no GPU row
# starts after this Rome time), --models and --shards (outcome, capture, sweep; models are Hugging Face ids or short
# keys), --half, --capture and --pick (sweep), --parent, --shards, --chain, --link and --max-gpu-hours (continuation).
# Needs a clean src, tests and scripts tree. Then submit the jobs with scripts/steering/submit_release_jobs.sh report/NAME.
#   e.g. scripts/steering/seal_release.sh complement whole-reasoning-20261021-r18 107 --cutoff 2026-10-24T06:00
#        scripts/steering/seal_release.sh capture outcome-capture-20261023-r2 60 --shards gpt:3,glm:15 --cutoff 2026-10-30T23:00
#        scripts/steering/seal_release.sh sweep outcome-steering-20261023-r6 70 --half A --capture outcome-capture-20261023-r2 \
#            --models gpt --shards 15 --cutoff 2026-10-30T23:00
#        scripts/steering/seal_release.sh continuation whole-reasoning-20261021-r21 122 \
#            --parent whole-reasoning-20261021-r18 --cutoff 2026-10-27T06:00
set -euo pipefail
cd "$(dirname "$0")/../.."
kind=${1:?KIND}; name=${2:?NAME}; tests=${3:?TESTS}; shift 3
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -m steering.release "$kind" "$name" --tests "$tests" "$@"

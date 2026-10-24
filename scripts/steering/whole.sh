#!/bin/bash
# Seal a whole-reasoning style release from the committed code, then submit its jobs:
#   scripts/steering/whole.sh KIND NAME TESTS [OPTION...]
# KIND is complement, relink, outcome, capture, sweep or continuation; NAME the new release (report/NAME); TESTS the
# number of tests that passed locally. OPTIONs go to steering.release: --cutoff TIME (every kind but relink: no GPU row
# starts after this Rome time), --models and --shards (outcome, capture, sweep), --half, --capture and --pick (sweep),
# --parent, --shards, --chain, --link and --max-gpu-hours (continuation). Needs a clean src, tests and scripts tree.
#   e.g. scripts/steering/whole.sh complement whole-reasoning-20261021-r18 107 --cutoff 2026-10-24T06:00
#        scripts/steering/whole.sh capture outcome-capture-20261023-r2 60 --shards gpt:3,glm:15 --cutoff 2026-10-30T23:00
#        scripts/steering/whole.sh sweep outcome-steering-20261023-r6 70 --half A --capture outcome-capture-20261023-r2 \
#            --models gpt --shards 15 --cutoff 2026-10-30T23:00
# AFTER="JOB ..." names running jobs the final analysis also waits for (a relink waits for its parent's combine, a
# continuation for the parent's extraction jobs, which `python -m steering.followup report/PARENT` lists):
#   AFTER=61927867 scripts/steering/whole.sh relink whole-reasoning-20261021-r19 120
#   scripts/steering/whole.sh continuation whole-reasoning-20261021-r21 122 \
#       --parent whole-reasoning-20261021-r18 --cutoff 2026-10-27T06:00
set -euo pipefail
cd "$(dirname "$0")/../.."
kind=$1; name=$2; tests=$3; shift 3
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -m steering.release $kind $name --tests $tests "$@"
scripts/steering/launch.sh $name ${AFTER:-}

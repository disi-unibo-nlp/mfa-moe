#!/bin/bash
# Seal a whole-reasoning style release from the committed code, then submit its jobs:
#   scripts/steering/whole.sh KIND NAME TESTS [OPTION...]
# KIND is complement, relink or outcome; NAME the new release (report/NAME); TESTS the number of tests that
# passed locally. OPTIONs go to steering.release: --cutoff TIME (complement, outcome: no GPU row starts after this
# Rome time), --models and --shards (outcome). Needs a clean src, tests and scripts tree.
#   e.g. scripts/steering/whole.sh complement whole-reasoning-20261021-r18 107 --cutoff 2026-10-24T06:00
set -euo pipefail
cd "$(dirname "$0")/../.."
kind=$1; name=$2; tests=$3; shift 3
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -m steering.release $kind $name --tests $tests "$@"
scripts/steering/launch.sh $name

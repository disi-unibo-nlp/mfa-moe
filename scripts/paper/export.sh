#!/bin/bash
# Write the paper tables and figure data from the sealed receipts (never edit the outputs by hand):
#   scripts/paper/export.sh [EFFECTS_WHOLE.json]     default: the final whole-reasoning receipt
set -euo pipefail
module load python/3.11.7
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python
$PY -m paper.export_routing_results
$PY -m paper.export_whole_results ${1:+--effects "$1"}
$PY -m paper.export_cl_exhibits

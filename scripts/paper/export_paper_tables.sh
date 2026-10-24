#!/bin/bash
# Write the paper tables and figure data from the sealed receipts (never edit the outputs by hand):
#   scripts/paper/export_paper_tables.sh STUDY_DIR [EFFECTS_WHOLE_JSON]
# STUDY_DIR is the routing study (report/final-routing-20261015-r1): its driver/ folder holds the pulsed-steering
# effects and the efficiency link. EFFECTS_WHOLE_JSON defaults to the final whole-reasoning receipt
# (paper.export_whole_results.DEFAULT_EFFECTS).
set -euo pipefail
study=$(realpath "${1:?STUDY_DIR}")
test -f "$study/driver/analysis-final/EFFECTS.json" || { echo "$study holds no driver/analysis-final/EFFECTS.json: it is not the routing study" >&2; exit 1; }
module load python/3.11.7
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 STUDY_DATA_ROOT=$study
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/correlation-client-3.11/bin/python
$PY -m paper.export_routing_results
$PY -m paper.export_whole_results ${2:+--effects "$2"}
$PY -m paper.export_cl_exhibits

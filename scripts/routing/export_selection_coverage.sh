#!/bin/bash
# Held-out coverage table of the selection figure (selection_coverage.csv), from the stored SELECTION.json receipts of the
# models in CAMPAIGN_DIR/CAMPAIGN.json (report/expert-memory-study-v1). OUT_DIR must be new:
#   scripts/routing/export_selection_coverage.sh CAMPAIGN_DIR OUT_DIR
set -euo pipefail
test -e "${1:?CAMPAIGN_DIR}/CAMPAIGN.json"
sbatch -J selection-export scripts/slurm/run_module_cpu.sbatch routing.expert_selection export "$1" "${2:?OUT_DIR}"

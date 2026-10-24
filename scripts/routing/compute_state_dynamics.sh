#!/bin/bash
# Class occupancy, dwell, transitions and expert nominations of the captured models, from the saved sentences (one job
# per model, a few minutes each). CAMPAIGN_DIR holds CAMPAIGN.json (report/research-campaign-v3-20261014-r5-r2); the models
# are the ones given as Hugging Face ids, else all the ones it lists. OUT_DIR must be new; it receives all outputs:
#   scripts/routing/compute_state_dynamics.sh CAMPAIGN_DIR OUT_DIR [HF_MODEL...]
set -euo pipefail
campaign=${1:?CAMPAIGN_DIR}; out=${2:?OUT_DIR}; test -e "$campaign/CAMPAIGN.json"
models=("${@:3}"); test ${#models[@]} -gt 0 || mapfile -t models < <(jq -r '.models | keys[]' "$campaign/CAMPAIGN.json")
for m in "${models[@]}"; do
  sbatch -J dynamics-${m##*/} scripts/slurm/run_module_cpu.sbatch routing.state_dynamics "$campaign" "$m" --out "$out"
done

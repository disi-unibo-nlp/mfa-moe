#!/bin/bash
# Pulsed steering analysis: the outcomes of every continuation of every model that has a manifest in the study (main
# design: driver/MODEL/ASSIGNMENTS.json, strong design: STRONG.json), then the effects of each design into
# OUT_DIR/main and OUT_DIR/strong (OUT_DIR must not exist; a design without a manifest is skipped):
#   scripts/steering/analyze_pulsed_steering.sh STUDY_DIR OUT_DIR
# To analyse fewer models, run the two modules by hand: python -m steering.pulsed_effects --models HF_MODEL,... --out DIR
set -euo pipefail
study=$(realpath "${1:?STUDY_DIR}"); out=$(realpath -m "${2:?OUT_DIR}")
test -f "$study/BINDING.json" || { echo "$study is no study folder (no BINDING.json)" >&2; exit 1; }
test ! -e "$out" || { echo "$out exists: OUT_DIR must be new" >&2; exit 1; }
shopt -s nullglob
declare -A manifest=([main]=ASSIGNMENTS.json [strong]=STRONG.json)
job=(sbatch --export=ALL,STUDY_DATA_ROOT="$study" --kill-on-invalid-dep=yes -c 1 --mem=12G)
for design in main strong; do
  outcomes=
  for file in "$study"/driver/*/"${manifest[$design]}"; do
    model=$(basename "$(dirname "$file")")
    outcomes+=:$("${job[@]}" --parsable -J "pulsed-outcomes-$design-$model" scripts/slurm/run_module_cpu.sbatch \
                 steering.pulsed_outcomes --model "$model" --design "$design")
  done
  test -n "$outcomes" || { echo "no $design manifest in $study/driver: skipped" >&2; continue; }
  "${job[@]}" --dependency=afterok$outcomes -J "pulsed-effects-$design" scripts/slurm/run_module_cpu.sbatch \
    steering.pulsed_effects --design "$design" --out "$out/$design"
done

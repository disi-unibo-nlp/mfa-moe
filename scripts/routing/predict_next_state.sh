#!/bin/bash
# Next-state prediction for the eight models: one job per model, then the collection after all of them. The receipts go
# into STUDY_DIR (<model>/NEXT_STATE.json and driver/NEXT_STATE.json). Models are Hugging Face ids; none means all eight
# (python -m models.roster, which needs Python 3.11 and PYTHONPATH=src):
#   scripts/routing/predict_next_state.sh STUDY_DIR [HF_MODEL...]
set -euo pipefail
study=${1:?STUDY_DIR}; test -e "$study/BINDING.json"
models=("${@:2}"); test ${#models[@]} -gt 0 || mapfile -t models < <(python -m models.roster)
after=""
for m in "${models[@]}"; do
  after+=:$(sbatch --parsable -J next-${m##*/} -p boost_usr_prod -c 8 --mem=40G -t 03:00:00 scripts/slurm/run_module_cpu.sbatch routing.next_state run --study "$study" --model "$m")
done
sbatch -J next-collect --dependency=afterok$after --kill-on-invalid-dep=yes -t 00:20:00 scripts/slurm/run_module_cpu.sbatch routing.next_state collect --study "$study"

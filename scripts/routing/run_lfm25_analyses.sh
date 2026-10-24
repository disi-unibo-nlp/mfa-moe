#!/bin/bash
# LFM2.5 native analyses: expert selection and nominations (derive) first, then the present-state readout and the class
# dynamics, both after derive succeeds. The study's receipts stay in STUDY_DIR; OUT_DIR must be new and receives the
# readout fits and READOUT_RESULTS.json (the published lfm25/READOUT_RESULTS.json is never rewritten):
#   scripts/routing/run_lfm25_analyses.sh STUDY_DIR OUT_DIR
set -euo pipefail
study=${1:?STUDY_DIR}; out=${2:?OUT_DIR}; test -e "$study/BINDING.json"
# derive also writes the main-design assignments through steering.pulse.plan, which takes its study from STUDY_DATA_ROOT.
derive=$(sbatch --parsable -J lfm-derive --export=ALL,STUDY_DATA_ROOT="$study" scripts/slurm/run_module_cpu.sbatch routing.lfm25_analyses derive --study "$study")
after=(--dependency=afterok:$derive --kill-on-invalid-dep=yes)
sbatch -J lfm-readout "${after[@]}" scripts/slurm/run_module_cpu.sbatch routing.lfm25_analyses readout --study "$study" --out "$out"
sbatch -J lfm-dynamics "${after[@]}" scripts/slurm/run_module_cpu.sbatch routing.lfm25_analyses dynamics --study "$study"

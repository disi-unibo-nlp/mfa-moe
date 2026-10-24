#!/bin/bash
# Present-state readout of the seven captured models: does routing tell a sentence's state beyond its text?
# (family-grouped refits, nulls, bootstraps; about 5 hours, 19 core-hours). ANALYSIS_DIR holds the saved caches the fits
# read; the published run needs the one in src/store.py (ANALYSIS), because its frozen inputs are named by absolute path.
# OUT_DIR must be new:
#   scripts/routing/readout_present_state.sh ANALYSIS_DIR OUT_DIR
set -euo pipefail
test -d "${1:?ANALYSIS_DIR}"
sbatch -J readout -p boost_usr_prod -c 4 --mem=64G -t 24:00:00 scripts/slurm/run_module_cpu.sbatch routing.present_state_readout --analysis-dir "$1" --out "${2:?OUT_DIR}"

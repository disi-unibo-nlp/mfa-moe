#!/bin/bash
# Anticipation: does routing predict the next reasoning state beyond the past text? GPT-OSS and Qwen3.6, about 21 hours
# (84 core-hours). ANALYSIS_DIR holds the saved caches the fits read; the published run needs the one in src/store.py
# (ANALYSIS), because its frozen inputs are named by absolute path. OUT_DIR must be new:
#   scripts/routing/anticipate_next_state.sh ANALYSIS_DIR OUT_DIR
set -euo pipefail
test -d "${1:?ANALYSIS_DIR}"
sbatch -J anticipation -p boost_usr_prod -c 4 --mem=64G -t 24:00:00 scripts/slurm/run_module_cpu.sbatch routing.next_state_anticipation --analysis-dir "$1" --out "${2:?OUT_DIR}"

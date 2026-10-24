#!/bin/bash
# Fit the lexical state readout that labels the steered continuations (about 2 hours). LEXICAL_DIR holds the
# native_<model>.parquet tables (the lexical folder of the study's bulk root). OUT_DIR must be new:
#   scripts/routing/fit_lexical_classifier.sh LEXICAL_DIR OUT_DIR
set -euo pipefail
test -d "${1:?LEXICAL_DIR}"
sbatch -J lexical --export=ALL,THREADS=4 scripts/slurm/run_module_cpu.sbatch routing.lexical --tables "$1" --out "${2:?OUT_DIR}"

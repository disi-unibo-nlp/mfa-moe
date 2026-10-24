#!/bin/bash
# Benchmark accuracy tables (accuracy, budget, continuation, errors, length gap) from the outcome sidecar OUTCOMES_JSON.
# OUT_DIR must be new:
#   scripts/routing/summarize_benchmark_accuracy.sh OUTCOMES_JSON OUT_DIR
set -euo pipefail
test -e "${1:?OUTCOMES_JSON}"
sbatch -J bench-accuracy --export=ALL,THREADS=4 scripts/slurm/run_module_cpu.sbatch routing.benchmark_accuracy "${2:?OUT_DIR}" --outcomes "$1"

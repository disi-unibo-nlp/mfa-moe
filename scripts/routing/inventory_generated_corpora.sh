#!/bin/bash
# Per-model inventory of the generated corpora (CSV tables, receipts and a TeX table) from the saved results. OUT_DIR must
# be new:
#   scripts/routing/inventory_generated_corpora.sh OUT_DIR
set -euo pipefail
sbatch -J bench-inventory --export=ALL,THREADS=4 scripts/slurm/run_module_cpu.sbatch routing.benchmark_inventory "${1:?OUT_DIR}"

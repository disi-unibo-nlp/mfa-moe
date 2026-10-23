#!/bin/bash
# Benchmark accuracy tables and the per-model inventory of the generated corpora:
#   scripts/routing/benchmarks.sh OUT_DIR      e.g. report/benchmarks
O=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/outcomes/outcomes_v3.latest.json
sbatch -J bench-accuracy --export=ALL,THREADS=4 scripts/slurm/cpu.sh moe_exp.routing.benchmark_accuracy --outcomes $O --out $1/accuracy
sbatch -J bench-inventory --export=ALL,THREADS=4 scripts/slurm/cpu.sh moe_exp.routing.benchmark_inventory $1/inventory

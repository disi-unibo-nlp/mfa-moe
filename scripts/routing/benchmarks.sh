#!/bin/bash
# Benchmark accuracy tables and the per-model inventory of the generated corpora:
#   scripts/routing/benchmarks.sh OUT_DIR      e.g. report/benchmarks
out="${1:?OUT_DIR}"
sbatch -J bench-accuracy --export=ALL,THREADS=4 scripts/slurm/cpu.sh routing.benchmark_accuracy "$out/accuracy"
sbatch -J bench-inventory --export=ALL,THREADS=4 scripts/slurm/cpu.sh routing.benchmark_inventory "$out/inventory"

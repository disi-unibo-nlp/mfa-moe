#!/bin/bash
# Fit the lexical state readout that labels the steered continuations (about 2 hours). OUT_DIR must be new:
#   scripts/routing/lexical.sh OUT_DIR
sbatch -J lexical --export=ALL,THREADS=4 scripts/slurm/cpu.sh routing.lexical --out "${1:?OUT_DIR}"

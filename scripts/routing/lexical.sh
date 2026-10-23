#!/bin/bash
# Fit the lexical state readout that labels the steered continuations (about 2 hours):
#   scripts/routing/lexical.sh
sbatch -J lexical --export=ALL,THREADS=4 scripts/slurm/cpu.sh routing.lexical fit --budget 2400

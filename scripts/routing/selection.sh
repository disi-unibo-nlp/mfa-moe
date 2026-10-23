#!/bin/bash
# Held-out coverage table of the selection figure (selection_coverage.csv), from the stored SELECTION.json receipts. OUT_DIR must be new:
#   scripts/routing/selection.sh OUT_DIR
sbatch -J selection-export scripts/slurm/cpu.sh routing.selection export $1

#!/bin/bash
# Class occupancy, dwell, transitions and expert nominations of the seven campaign-v3 models, from the saved sentences
# (one job per model, a few minutes each). OUT_DIR must be new; it receives all outputs:
#   scripts/routing/dynamics.sh OUT_DIR
for m in gpt nemotron qwen35 qwen36 gemma glm qwen330b; do
  sbatch -J dynamics-$m scripts/slurm/cpu.sh routing.dynamics report/research-campaign-v3-20261014-r5-r2 $m --out $1
done

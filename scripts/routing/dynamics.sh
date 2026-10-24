#!/bin/bash
# Class occupancy, dwell, transitions and expert nominations of the seven captured models, from the saved sentences
# (one job per model, a few minutes each). OUT_DIR must be new; it receives all outputs:
#   scripts/routing/dynamics.sh OUT_DIR
out="${1:?OUT_DIR}"
for m in gpt nemotron qwen35 qwen36 gemma glm qwen330b; do
  sbatch -J dynamics-$m scripts/slurm/run_module_cpu.sbatch routing.dynamics report/research-campaign-v3-20261014-r5-r2 $m --out "$out"
done

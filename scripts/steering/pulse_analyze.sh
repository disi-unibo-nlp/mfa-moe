#!/bin/bash
# Pulsed steering analysis: the outcomes of every continuation (main design for the eight models, strong for four),
# then the effects of both designs into OUT/main and OUT/strong (OUT must not exist):
#   scripts/steering/pulse_analyze.sh OUT
set -euo pipefail
out=$(realpath "$1")
jobs=
for model in gpt nemotron qwen35 qwen36 gemma glm qwen330b lfm25; do
  jobs+=:$(sbatch --parsable -J pulse-outcomes-$model -c 1 --mem=12G scripts/slurm/run_module_cpu.sbatch steering.pulse_outcomes --model $model)
done
for model in gpt nemotron qwen330b gemma; do
  jobs+=:$(sbatch --parsable -J pulse-outcomes-strong-$model -c 1 --mem=12G scripts/slurm/run_module_cpu.sbatch steering.pulse_outcomes --model $model --design strong)
done
sbatch -J pulse-effects --dependency=afterok$jobs -c 1 --mem=12G scripts/slurm/run_module_cpu.sbatch steering.pulse_effects --out $out/main
sbatch -J pulse-effects-strong --dependency=afterok$jobs -c 1 --mem=12G \
  scripts/slurm/run_module_cpu.sbatch steering.pulse_effects --design strong --models gpt,nemotron,qwen330b,gemma --out $out/strong

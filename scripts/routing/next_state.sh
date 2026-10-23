#!/bin/bash
# Next-state prediction for the eight models: one job per model, then the collection after all of them:
#   scripts/routing/next_state.sh
for m in gpt nemotron qwen35 qwen36 gemma glm qwen330b lfm25; do
  JOBS+=:$(sbatch --parsable -J next-$m -p boost_usr_prod -c 8 --mem=40G -t 03:00:00 scripts/slurm/cpu.sh moe_exp.routing.next_state run --model $m)
done
sbatch -J next-collect --dependency=afterok$JOBS -p boost_usr_prod -c 1 --mem=4G -t 00:20:00 scripts/slurm/cpu.sh moe_exp.routing.next_state collect

#!/bin/bash
# Anticipation: does routing predict the next reasoning state beyond the past text? GPT-OSS and Qwen3.6, about 21 hours.
# OUT_DIR must be new:
#   scripts/routing/anticipation.sh OUT_DIR
sbatch -J anticipation -p boost_usr_prod -c 4 --mem=64G -t 24:00:00 scripts/slurm/cpu.sh routing.anticipation --out $1

#!/bin/bash
# Anticipation: does routing predict the next reasoning state beyond the past text? Seven models (the first run was cancelled after 21 hours):
#   scripts/routing/anticipation.sh
sbatch -J anticipation -p boost_usr_prod -c 4 --mem=64G -t 24:00:00 scripts/slurm/cpu.sh moe_exp.routing.anticipation --workers 4

#!/bin/bash
# How often each expert is used per reasoning token in every cohort-B attempt (input of outcome.sh):
#   scripts/steering/outcome_usage.sh
sbatch -J os-usage -t 02:00:00 scripts/slurm/cpu.sh steering.outcome usage

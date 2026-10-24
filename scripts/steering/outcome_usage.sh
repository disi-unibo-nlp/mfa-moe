#!/bin/bash
# How often each expert is used per reasoning token in every cohort-B attempt (input of whole.sh outcome):
#   scripts/steering/outcome_usage.sh
sbatch -J os-usage -t 02:00:00 scripts/slurm/run_module_cpu.sbatch steering.outcome usage

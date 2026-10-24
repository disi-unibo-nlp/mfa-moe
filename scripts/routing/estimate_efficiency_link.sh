#!/bin/bash
# Efficiency link for the eight models: the table, the published EFFICIENCY_LINK_V2.json and the link test on the pulse cells:
#   scripts/routing/efficiency_link.sh
sbatch -J efficiency-link -t 01:00:00 --export=ALL,THREADS=4 scripts/slurm/run_module_cpu.sbatch routing.efficiency_link all

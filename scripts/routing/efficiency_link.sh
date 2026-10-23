#!/bin/bash
# Efficiency link for the eight models: the table, the published EFFICIENCY_LINK_V2.json and the link test on the pulse cells:
#   scripts/routing/efficiency_link.sh
sbatch -J efficiency-link -t 01:00:00 scripts/slurm/cpu.sh moe_exp.routing.efficiency_link all

#!/bin/bash
# Efficiency link for the eight models: the table, the published EFFICIENCY_LINK_V2.json and the link test on the pulse
# cells, written into STUDY_DIR (driver/) and its bulk folder:
#   scripts/routing/estimate_efficiency_link.sh STUDY_DIR
set -euo pipefail
test -e "${1:?STUDY_DIR}/BINDING.json"
sbatch -J efficiency-link -t 01:00:00 --export=ALL,THREADS=4 scripts/slurm/run_module_cpu.sbatch routing.efficiency_link all --study "$1"

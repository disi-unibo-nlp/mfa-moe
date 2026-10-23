#!/bin/bash
# Present-state readout of the seven campaign-v3 models (family-grouped refits, nulls, bootstraps; about 5 hours):
#   scripts/routing/readout.sh
sbatch -J readout -p boost_usr_prod -c 4 --mem=64G -t 24:00:00 scripts/slurm/cpu.sh moe_exp.routing.readout --workers 4

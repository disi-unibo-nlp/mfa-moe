#!/bin/bash
# Present-state readout of the seven captured models (family-grouped refits, nulls, bootstraps; about 5 hours).
# OUT_DIR must be new:
#   scripts/routing/readout.sh OUT_DIR
sbatch -J readout -p boost_usr_prod -c 4 --mem=64G -t 24:00:00 scripts/slurm/cpu.sh routing.readout --out "${1:?OUT_DIR}"

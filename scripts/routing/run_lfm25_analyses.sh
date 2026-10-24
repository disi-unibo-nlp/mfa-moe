#!/bin/bash
# LFM2.5 native analyses: expert selection and nominations, then the present-state readout and the class dynamics.
# OUT_DIR must be new; it receives the readout fits and READOUT_RESULTS.json (the receipt in the study folder stays):
#   scripts/routing/lfm.sh OUT_DIR
out="${1:?OUT_DIR}"
D=$(sbatch --parsable -J lfm-derive scripts/slurm/run_module_cpu.sbatch routing.lfm derive)
sbatch -J lfm-readout --dependency=afterok:$D scripts/slurm/run_module_cpu.sbatch routing.lfm readout --out "$out"
sbatch -J lfm-dynamics --dependency=afterok:$D scripts/slurm/run_module_cpu.sbatch routing.lfm dynamics

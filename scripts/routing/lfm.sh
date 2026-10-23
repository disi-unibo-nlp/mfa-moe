#!/bin/bash
# LFM2.5 native analyses: expert selection and nominations, then the present-state readout and the class dynamics:
#   scripts/routing/lfm.sh
D=$(sbatch --parsable -J lfm-derive scripts/slurm/cpu.sh steering.pulse lfm-derive)
sbatch -J lfm-readout --dependency=afterok:$D scripts/slurm/cpu.sh steering.pulse lfm-readout
sbatch -J lfm-dynamics --dependency=afterok:$D scripts/slurm/cpu.sh steering.pulse lfm-dynamics

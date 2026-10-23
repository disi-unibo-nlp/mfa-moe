#!/bin/bash
# Accept a finished capture of one model and cohort (a CPU job that hashes every captured file):
#   scripts/capture/gate.sh MANIFEST_DIR OUTPUT_JSON      e.g. .../campaign-v3/manifests/gpt/A acceptance/capture-gpt-A.json
sbatch -J gate scripts/slurm/cpu.sh capture.gate "$@"

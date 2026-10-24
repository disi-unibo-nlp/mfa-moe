#!/bin/bash
# Pick the sentences to label (cohorts A and B, four parts) from a model's finished traces; login node:
#   scripts/labeling/sample.sh MODEL                     e.g. gpt
R=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/results/correlation_pipeline/card-v3/$1
sbatch -J cohorts-$1 scripts/slurm/cpu.sh labeling.sample_cohorts \
  --model $1 --gen-root $(ls -d $R/generation/*/) --out $R/cohorts

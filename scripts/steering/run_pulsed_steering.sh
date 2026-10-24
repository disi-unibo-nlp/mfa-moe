#!/bin/bash
# Pulsed steering of one model: submit SHARDS GPU jobs, each a disjoint set of question families (a rerun resumes):
#   scripts/steering/run_pulsed_steering.sh STUDY_DIR HF_MODEL SHARDS HOURS [main|strong]
# STUDY_DIR holds the manifest of the design (driver/MODEL/ASSIGNMENTS.json for main, STRONG.json for strong), which
# `python -m steering.pulsed_steering plan` and `plan-strong` write. HOURS is the time limit of each job.
# For every model:  for m in $(python -m models.roster); do scripts/steering/run_pulsed_steering.sh STUDY_DIR $m 2 12; done
set -euo pipefail
study=$(realpath "${1:?STUDY_DIR}"); model=${2:?HF_MODEL}; shards=${3:?SHARDS}; hours=${4:?HOURS}; design=${5:-main}
key=$(STUDY_DATA_ROOT=$study PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -m steering.pulsed_steering check --model "$model" --design "$design")
for ((s = 0; s < shards; s++)); do
  env -u SLURM_JOB_ID sbatch --export=ALL,STUDY_DATA_ROOT="$study" -t "$hours:00:00" -J "fr-$key-$s-of-$shards" \
    scripts/slurm/run_module_gpu.sbatch steering.pulsed_steering run --model "$model" --shard "$s/$shards" --design "$design"
done

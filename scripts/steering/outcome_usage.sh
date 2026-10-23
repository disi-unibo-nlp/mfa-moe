#!/bin/bash
# Submit the CPU job that measures, for every cohort-B attempt, how often each expert is selected per
# reasoning token. Its output (USAGE.npz/.json under the FAST bulk root) feeds the outcome selection.
#   scripts/steering/outcome_usage.sh [MODELS]      # default gpt,qwen330b,nemotron
set -euo pipefail
cd "$(dirname "$0")/../.."
models="${1:-gpt,qwen330b,nemotron}"
logs=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/outcome-steering/logs
mkdir -p "$logs"
sbatch -A iscrc_miosr -p lrd_all_serial --qos=normal -N1 --ntasks=1 --cpus-per-task=4 --mem=30G \
  -t 02:00:00 -J os-usage -o "$logs/usage-%j.out" -e "$logs/usage-%j.err" \
  scripts/slurm/cpu.sh moe_exp.steering.outcome usage --models "$models" --workers 4

#!/bin/bash
# Apply the declared extension rule (ledger D-0003) to a sealed outcome analysis; if it is met, seal and launch the
# next outcome release. Submitted by hand after the analysis job:
#   sbatch scripts/steering/outcome_next.sh EFFECTS_OUTCOME.json NAME TESTS [OPTION...]
#   e.g. sbatch ... report/outcome-steering-20261022-r1/analysis/EFFECTS_OUTCOME.json outcome-steering-20261022-r2 125 \
#          --models gemma,glm,qwen35,qwen36 --shards glm:12,gemma:3,qwen35:3,qwen36:3
#SBATCH --account=iscrc_miosr --partition=lrd_all_serial --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=1 --mem=8G --time=30 --job-name=outcome-next
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
source scripts/slurm/env.sh || exit 1
effects=$1; name=$2; tests=$3; shift 3
if python -m steering.outcome decide --effects "$effects"; then
  scripts/steering/whole.sh outcome $name $tests "$@"
fi

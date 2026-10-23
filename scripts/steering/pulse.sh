#!/bin/bash
# Pulsed steering: submit SHARDS GPU jobs per model (each a disjoint set of question families; a rerun resumes):
#   scripts/steering/pulse.sh SHARDS HOURS MODEL...
set -euo pipefail
n=$1; hours=$2; shift 2
for model in "$@"; do
  test -f "report/final-routing-20261015-r1/driver/$model/ASSIGNMENTS.json"
  for ((s = 0; s < n; s++)); do
    sbatch --gres=gpu:2 -c 16 --mem=120G -t $hours:00:00 -J fr-$model-$s-of-$n scripts/steering/gpu.sh run --model $model --shard $s/$n
  done
done
scripts/slurm/ledger.sh | tail -1

#!/bin/bash
# Submit intervention shards of the reduced final-routing design (login node).
# usage: submit_production.sh SHARDS HOURS MODEL [MODEL ...]
# Each shard is a disjoint set of question families; a job resumes any unfinished units of its shard.
set -euo pipefail
cd /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
n=$1; hours=$2; shift 2
L=/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs/final-routing-20261015-r1/driver-logs
for model in "$@"; do
  test -f "report/final-routing-20261015-r1/driver/$model/ASSIGNMENTS.json"
  for ((s = 0; s < n; s++)); do
    env -u SLURM_JOB_ID sbatch -A iscrc_miosr -p boost_usr_prod --qos=normal -N1 --ntasks=1 --gres=gpu:2 \
      --cpus-per-task=16 --mem=120G -t "$hours:00:00" -J "fr-$model-$s-of-$n" -o "$L/%x-%j.out" \
      scripts/study/driver.sh run --model "$model" --shard "$s/$n"
  done
done
scripts/study/ledger.sh | tail -1

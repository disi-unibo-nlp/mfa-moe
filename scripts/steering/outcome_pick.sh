#!/bin/bash
# Apply the frozen pick rule to the analysis of half A of the outcome sweep; if it picks cells, seal and
# launch half B with them. Submit it by hand right after launching half A, behind half A's analysis job:
#   sbatch --dependency=afterok:ANALYSIS_JOB scripts/steering/outcome_pick.sh HALF_A MODEL NAME TESTS --capture C --shards N --cutoff TIME
#   e.g. ... outcome-steering-20261023-r6 gpt outcome-steering-20261023-r7 70 --capture outcome-capture-20261023-r2 --shards 2 --cutoff 2026-10-30T23:00
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=4 --mem=30G --time=02:00:00 --job-name=os-pick
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
source scripts/slurm/env.sh || exit 1
half_a=$1; model=$2; name=$3; tests=$4; shift 4
status=0
python -m steering.outcome pick --effects "report/$half_a/analysis/EFFECTS_OUTCOME.json" --model "$model" || status=$?
test $status -ne 3 || exit 0  # no cell to confirm: nothing more to run
test $status -eq 0 || exit $status  # anything else is a failure, not a decision
scripts/steering/whole.sh sweep "$name" "$tests" --half B --models "$model" --pick "report/$half_a/analysis/PICK.json" "$@"

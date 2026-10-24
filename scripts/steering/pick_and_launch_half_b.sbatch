#!/bin/bash
# Apply the frozen pick rule to the analysis of half A of the outcome sweep; if it picks cells, seal and
# launch half B with them. Submit it by hand right after launching half A, behind half A's analysis job:
#   sbatch --dependency=afterok:ANALYSIS_JOB scripts/steering/pick_and_launch_half_b.sbatch HALF_A_DIR HF_MODEL NAME TESTS --capture C --shards N --cutoff TIME
#   e.g. ... report/outcome-steering-20261023-r6 openai/gpt-oss-20b outcome-steering-20261023-r7 70 --capture outcome-capture-20261023-r2 --shards 2 --cutoff 2026-10-30T23:00
# HALF_A_DIR is the half-A release (report/NAME_A) and holds analysis/EFFECTS_OUTCOME.json; the pick goes to analysis/PICK.json
# there. NAME is the new release (report/NAME); TESTS and the options go to seal_release.sh sweep.
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=4 --mem=30G --time=02:00:00 --job-name=os-pick
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
source scripts/slurm/env.sh || exit 1
half_a=${1:?HALF_A_DIR}; model=${2:?HF_MODEL}; name=${3:?NAME}; tests=${4:?TESTS}; shift 4
half_a=${half_a%/}
status=0
python -m steering.outcome pick --effects "$half_a/analysis/EFFECTS_OUTCOME.json" --model "$model" || status=$?
test $status -ne 3 || exit 0  # no cell to confirm: nothing more to run
test $status -eq 0 || exit $status  # anything else is a failure, not a decision
scripts/steering/seal_release.sh sweep "$name" "$tests" --half B --models "$model" --pick "$half_a/analysis/PICK.json" "$@"
scripts/steering/submit_release_jobs.sh "report/$name"

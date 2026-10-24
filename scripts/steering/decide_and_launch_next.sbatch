#!/bin/bash
# Apply the declared extension rule to a sealed outcome analysis; if it is met, seal and launch the
# next outcome release. Submitted by hand after the analysis job:
#   sbatch scripts/steering/decide_and_launch_next.sbatch EFFECTS_OUTCOME.json NAME TESTS --cutoff TIME [OPTION...]
#   e.g. sbatch ... report/outcome-steering-20261022-r1/analysis/EFFECTS_OUTCOME.json outcome-steering-20261022-r2 125 \
#          --cutoff 2026-10-25T06:00 --models gemma,glm,qwen35,qwen36 --shards glm:12,gemma:3,qwen35:3,qwen36:3
# NAME is the new release (report/NAME); TESTS and the options go to seal_release.sh outcome.
#SBATCH --account=iscrc_miosr --partition=lrd_all_serial --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=1 --mem=8G --time=30 --job-name=outcome-next
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
source scripts/slurm/env.sh || exit 1
effects=${1:?EFFECTS_OUTCOME.json}; name=${2:?NAME}; tests=${3:?TESTS}; shift 3
status=0
python -m steering.outcome decide --effects "$effects" || status=$?
test $status -ne 3 || exit 0  # the rule is not met: nothing more to run
test $status -eq 0 || exit $status  # anything else is a failure, not a decision
scripts/steering/seal_release.sh outcome "$name" "$tests" "$@"
scripts/steering/submit_release_jobs.sh "report/$name"

#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=120G --time=04:00:00 --job-name=pairs-judge
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# One shard of the next-sentence judge of one model (a rerun resumes); scripts/routing/transition_pairs.sh submits them:
#   sbatch scripts/routing/pairs_judge.sbatch MODEL I/N [SOURCE_CLASSES]
source scripts/slurm/env.sh || exit 1
# The renderer child that builds the judge prompts imports dspy, which only the judge-client environment has.
export PYTHONPATH=$PYTHONPATH:$DSPY/lib/python3.11/site-packages
python -m routing.transition_pairs judge --model "${1:?MODEL}" --shard "${2:?I/N}" --source-classes "${3:-Read,Verify}"

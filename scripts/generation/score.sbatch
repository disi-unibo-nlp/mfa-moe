#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=120G --time=06:00:00 --no-requeue --job-name=score
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# Check answers with the judge model (answer-equivalence audit): sbatch scripts/generation/score.sbatch ITEMS OUT
source scripts/slurm/env.sh || exit 1
start_server judge
python -m generation.answer_equivalence run --items "$1" --out "$2" --base-url $URL --workers 48

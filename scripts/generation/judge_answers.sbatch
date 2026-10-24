#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=120G --time=06:00:00 --no-requeue --job-name=judge-answers
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# Check rejected answers with the judge model (answer-equivalence audit); a rerun skips the items already in VERDICTS_JSONL:
#   sbatch scripts/generation/judge_answers.sbatch ITEMS_JSONL VERDICTS_JSONL
# ITEMS_JSONL is the file that scripts/generation/list_rejected_answers.sh writes.
: "${1:?ITEMS_JSONL}" "${2:?VERDICTS_JSONL}"
test -f "$1" || { echo "no such items file: $1" >&2; exit 1; }
source scripts/slurm/env.sh || exit 1
start_server $JUDGE_MODEL
python -m generation.judge_answers run --items "$1" --out "$2" --base-url $URL --workers 48

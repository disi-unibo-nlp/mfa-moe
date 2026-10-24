#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=120G --time=12:00:00 --no-requeue --job-name=label-sentences
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# Label every selected sentence under TRACE_ROOT with the judge (the frozen program unless JUDGE_PROGRAM is given).
# A rerun resumes from OUT_DIR/checkpoints:
#   sbatch scripts/labeling/label_sentences.sbatch TRACE_ROOT OUT_DIR [JUDGE_PROGRAM]
# TRACE_ROOT holds <dataset>/traces.jsonl with metadata.sentence_selection, e.g. a part root that choose_sentences.sh wrote.
# Pass TRACE_ROOT exactly as in the earlier run: its spelling is part of the checkpoint key.
: "${1:?TRACE_ROOT}" "${2:?OUT_DIR}"
test -d "$1" || { echo "no such folder: $1" >&2; exit 1; }
source scripts/slurm/env.sh || exit 1
start_server $JUDGE_MODEL
export PYTHONPATH=$PYTHONPATH:$DSPY/lib/python3.11/site-packages:$MFA/envs/vllm-cu129/lib/python3.11/site-packages
$DSPY/bin/python -m labeling.label_sentences --trace-root "$1" --output-dir "$2" --base-url $URL ${3:+--judge-program "$3"}

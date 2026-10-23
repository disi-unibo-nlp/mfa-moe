#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=120G --time=12:00:00 --no-requeue --job-name=label
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# Label the sentences of one set of traces with the judge. A rerun resumes from OUTPUT_DIR/checkpoints:
#   sbatch scripts/labeling/label.sbatch TRACE_ROOT OUTPUT_DIR TOTAL     (TOTAL = number of sentences)
source scripts/slurm/env.sh
start_server judge
export PYTHONPATH=$PYTHONPATH:$DSPY/lib/python3.11/site-packages:$MFA/envs/vllm-cu129/lib/python3.11/site-packages
$DSPY/bin/python -m moe_exp.labeling.annotation_batch --trace-root "$1" --output-dir "$2" \
  --part 0 --parts 1 --part-size $3 --total $3 --model Qwen/Qwen3.8-27B \
  --judge-program $MFA/repo/results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json \
  --base-url $URL --batch-size 64 --max-tokens 16384 --reasoning-effort low

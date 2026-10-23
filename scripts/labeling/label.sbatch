#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=120G --time=12:00:00 --no-requeue --job-name=label
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# Label every selected sentence under TRACE_ROOT with the judge. A rerun resumes from OUTPUT_DIR/checkpoints:
#   sbatch scripts/labeling/label.sbatch TRACE_ROOT OUTPUT_DIR
source scripts/slurm/env.sh || exit 1
start_server judge
export PYTHONPATH=$PYTHONPATH:$DSPY/lib/python3.11/site-packages:$MFA/envs/vllm-cu129/lib/python3.11/site-packages
$DSPY/bin/python -m labeling.annotation_batch --trace-root "$1" --output-dir "$2" --base-url $URL

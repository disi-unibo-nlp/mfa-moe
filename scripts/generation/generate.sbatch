#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=120G --time=20:00:00 --no-requeue --job-name=gen
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# Generate one shard of one benchmark with the model's card sampler:
#   sbatch scripts/generation/generate.sbatch MODEL DATASET SHARD NSHARDS     e.g. gpt aime24 0 2
# No new request starts in the last 3 hours of the job (--deadline-margin-hours); resubmit to continue.
source scripts/slurm/env.sh || exit 1
start_server $1
python -m generation.generate --profile $1 --datasets $2 --shard-index $3 --num-shards $4 \
  --base-url $URL --output-dir $MFA/repo/results/correlation_pipeline/card-v3/$1/generation

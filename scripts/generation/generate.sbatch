#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16 --gres=gpu:2 --mem=120G --time=20:00:00 --no-requeue --job-name=gen
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# Generate one shard of one benchmark with the model's card sampler:
#   sbatch scripts/generation/generate.sbatch MODEL DATASET SHARD NSHARDS     e.g. gpt aime24 0 2
source scripts/slurm/env.sh
start_server $1
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)  # stop starting requests 3 hours before the time limit
export VLLM_VERSION_RECORD=$(sed -n 's/^Version: \([0-9.]*\).*/\1/p' $MFA/envs/vllm-cu129/lib/python3.11/site-packages/vllm-*.dist-info/METADATA)
python -m moe_exp.generation.generate --profile $1 --save-token-ids --datasets $2 --num-shards $4 --shard-index $3 \
  --seed 20260925 --workers 16 --timeout 21600 --deadline-epoch $((END - 3 * 3600)) \
  --base-url $URL --output-dir $MFA/repo/results/correlation_pipeline/card-v3/$1/generation

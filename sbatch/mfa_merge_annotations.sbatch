#!/bin/bash
#SBATCH --job-name=mfa-moe-merge
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/merge_logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/merge_logs/%x-%j.err

# Final acceptance for the hybrid labeling: re-check the splitter blast radius, verify
# every reused v1 label against the source traces, then write the merged v2 sequence.
set -euo pipefail

REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python

module load python/3.11.7
cd "$REPO"

echo "=== 1/3 splitter blast radius"
"$PY" -m moe_exp.correlation_pipeline.annotation_merge --verify-splitter

echo "=== 2/3 merge dry-run with --verify-units"
"$PY" -m moe_exp.correlation_pipeline.annotation_merge --dry-run --verify-units

echo "=== 3/3 merge"
"$PY" -m moe_exp.correlation_pipeline.annotation_merge

echo "=== merged outputs"
ls -la /leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen38-mtp3-production-v2-affected/merged/

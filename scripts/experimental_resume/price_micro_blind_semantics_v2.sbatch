#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --job-name=st-micro-blind-price-v2
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
DRIVER="$REPO/scripts/experimental_resume/price_micro_blind_semantics_v2.py"
RATER="$REPO/scripts/experimental_resume/rate_micro_blind_semantics_v2.py"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == b01a55eccda867b718326f110c30a777737310305906d4f8d0dc4d2dd2236250 ]] || exit 2
[[ "$(sha256sum "$RATER" | cut -d' ' -f1)" == 70da67380e6a7de80097cfb806a721e24042b3eae035560c03b0d83fc467d08b ]] || exit 2
module load python/3.11.7
umask 077
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER"

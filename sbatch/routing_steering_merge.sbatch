#!/bin/bash
# Submit only after the entire authorized extraction array succeeds.
#SBATCH --job-name=routing-steering-merge
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/routing-steering-merge-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/routing-steering-merge-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == lrd_all_serial ]] || exit 2
[[ "$(hostname -s)" == login08 || "$(hostname -s)" == login13 ]] || exit 2
[[ $# == 3 ]] || { echo 'Usage: sbatch routing_steering_merge.sbatch INVENTORY OUTPUT FROZEN_SOURCE_ROOT'; exit 2; }
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
REPO="$P/repo"
case "$2" in "$REPO"/results/correlation_pipeline/routing-steering-*) ;; *) exit 2 ;; esac
case "$3" in "$REPO"/results/correlation_pipeline/routing-steering-*/source-*) ;; *) exit 2 ;; esac
[[ -f "$3/src/moe_exp/correlation_pipeline/dynamics/campaign.py" ]] || exit 2
umask 077
export TMPDIR=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$3/src:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"
manifests=()
for shard in {0..3}; do
  [[ -f "$2/shard-$shard/manifest.json" ]] || exit 2
  manifests+=("$2/shard-$shard/manifest.json")
done
srun "$P/envs/vllm-cu129/bin/python" -m moe_exp.correlation_pipeline.dynamics steering merge-extractions \
  --inventory "$1" --manifests "${manifests[@]}" --output "$2/manifest.json"

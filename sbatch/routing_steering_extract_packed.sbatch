#!/bin/bash
# Four frozen logical shards, two workers, two sequential shards per worker.
# Same workload as routing_steering_extract.sbatch, fewer scheduler slots/hours.
#SBATCH --job-name=routing-steering-extract
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/routing-steering-extract-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/routing-steering-extract-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == lrd_all_serial ]] || exit 2
[[ "$(hostname -s)" == login08 || "$(hostname -s)" == login13 ]] || exit 2
[[ $# == 3 ]] || { echo 'Usage: sbatch routing_steering_extract_packed.sbatch INVENTORY OUTPUT FROZEN_SOURCE_ROOT'; exit 2; }
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
inventory="$1"
output="$2"
run_worker() {
  local worker="$1" shard
  for shard in "$worker" "$((worker + 2))"; do
    # No node-local staging is used. Inheriting the job's complete tmpfs GRES
    # would serialize the two otherwise disjoint CPU/memory step allocations.
    srun --exclusive --exact --gres=none --hint=nomultithread --cpu-bind=cores \
      --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=12G \
      "$P/envs/vllm-cu129/bin/python" -m moe_exp.correlation_pipeline.dynamics steering extract \
      --inventory "$inventory" --output "$output/shard-$shard" --include-weights \
      --shard-index "$shard" --shard-count 4 --max-output-bytes 6442450944
  done
}
run_worker 0 &
worker_zero=$!
run_worker 1 &
worker_one=$!
wait "$worker_zero"
wait "$worker_one"

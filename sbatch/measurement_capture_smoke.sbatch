#!/bin/bash
#SBATCH --job-name=mfa-native-metric-smoke
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lorenzo.molfetta@unibo.it
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr ]] || exit 2
case "${1:-}" in gpt|glm|qwen330b) ;; *) exit 2;; esac
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
ROOT="$P/routing-study"
RUNTIME="$ROOT/results/measurement-study/runtime/$SLURM_JOB_ID"
umask 077
mkdir -p "$RUNTIME/tmp" "$RUNTIME/cache"
module load python/3.11.7 cuda/12.6
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/cache"
export TORCH_HOME="$RUNTIME/cache/torch" TRITON_CACHE_DIR="$RUNTIME/cache/triton" DSPY_CACHEDIR="$RUNTIME/cache/dspy"
export HF_HOME="$P/cache/hf" HF_HUB_CACHE="$P/cache/hf/hub" HF_HUB_OFFLINE=1
export PATH="$P/envs/vllm-cu129/bin:$PATH"
if [[ "$1" == gpt ]]; then
  export LOCAL_KERNELS="kernels-community/gpt-oss-triton-kernels=$P/cache/hf/hub/models--kernels-community--gpt-oss-triton-kernels/snapshots/0f351046d799bc4d2c9dbb2c7cf36753204b5916"
fi
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$P/envs/forward-kernels-0.16:$P/repo/src:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
BUNDLE="${2:?Frozen source bundle required}"
[[ "$BUNDLE" == "$ROOT/results/measurement-study/bundles/"* ]] || exit 2
export MEASUREMENT_CAMPAIGN="$BUNDLE/campaign.json"
export PYTHONPATH="$BUNDLE:$PYTHONPATH"
cd "$BUNDLE"
srun "$P/envs/vllm-cu129/bin/python" -u -m measurement_study.smoke "$1" --long

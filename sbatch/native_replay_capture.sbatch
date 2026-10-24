#!/bin/bash
#SBATCH --job-name=mfa-vllm-metric-capture
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
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname)" == lrdn* ]] || exit 2
case "${1:-}" in qwen36|nemotron|gemma) ;; *) exit 2;; esac
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
ROOT="$P/routing-study"
BUNDLE="${2:?Frozen source bundle required}"
[[ "$BUNDLE" == "$ROOT/results/measurement-study/bundles/"* ]] || exit 2
source "$ROOT/scripts/routing_study_env.sh"
export MEASUREMENT_CAMPAIGN="$BUNDLE/campaign.json"
export PYTHONPATH="$BUNDLE:$P/repo/src:$P/envs/vllm-cu129/lib/python3.11/site-packages:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
cd "$BUNDLE"
srun "$P/envs/vllm-cu129/bin/python" -u -m measurement_study.native_replay "$1"

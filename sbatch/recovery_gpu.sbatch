#!/bin/bash
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
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr ]] || exit 2
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
ROOT="$P/routing-study"
BUNDLE="${2:?Frozen bundle required}"
case "${1:-}" in gemma) MODULE=measurement_study.recovery_gpu;; nemotron) MODULE=measurement_study.nemotron_validation;; nemotron-confirmation) MODULE=measurement_study.nemotron_confirmation;; *) exit 2;; esac
source "$ROOT/scripts/routing_study_env.sh"
export PYTHONPATH="$BUNDLE:$P/envs/forward-kernels-0.16:$P/repo/src:$P/envs/vllm-cu129/lib/python3.11/site-packages:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
mkdir -p "$XDG_CACHE_HOME/torch/kernels"
cd "$BUNDLE"
srun "$P/envs/vllm-cu129/bin/python" -B -u -m "$MODULE" --bundle "$BUNDLE"

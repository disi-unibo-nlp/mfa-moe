#!/bin/bash
#SBATCH --job-name=mfa-forward-cpu-gates
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/forward-cpu-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/forward-cpu-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit through Slurm after approval" >&2; exit 2; }
[[ "$(id -un)" == lmolfett ]] || exit 2
[[ "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr ]] || exit 2
[[ "${SLURM_JOB_PARTITION:-}" == lrd_all_serial ]] || exit 2
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
REPO="$P/repo"
RUN="$REPO/results/correlation_pipeline/forward-validation-v1/cpu-${SLURM_JOB_ID}"
module load python/3.11.7
umask 077
mkdir -p "$RUN/tmp" "$RUN/cache"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export XDG_CACHE_HOME="$RUN/cache" DSPY_CACHEDIR="$RUN/cache/dspy" TMPDIR="$RUN/tmp"
export HF_HOME="$RUN/cache/hf" HF_HUB_CACHE="$P/cache/hf/hub"
export TORCH_HOME="$RUN/cache/torch" TRITON_CACHE_DIR="$RUN/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor"
export PYTHONPATH="$REPO/src:$P/envs/vllm-cu129/lib/python3.11/site-packages:$P/envs/correlation-client-3.11/lib/python3.11/site-packages"
cd "$REPO"
srun --ntasks=1 timeout 540 "$P/envs/vllm-cu129/bin/python" scripts/validate_forward_cpu.py "$RUN/gate.json" "$@"

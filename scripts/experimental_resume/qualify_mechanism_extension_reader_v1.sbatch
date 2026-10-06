#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=st-extension-reader-qual-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/rate_mechanism_extension_start_readers_v1.py"
CONTRACT="$REPO/scripts/experimental_resume/mechanism_extension_reader_contract_v1.py"
SEALER="$REPO/scripts/experimental_resume/seal_mechanism_extension_reader_v1.py"
MANIFEST="$REPO/report/experimental-resume-v1/MECHANISM_EXTENSION_READER_QUAL_MANIFEST_v1.json"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == d7d725a4e9a1026eb9f9d44ae52c3dfb18eb5fde6593a1d096dd160b93d095bb ]] || exit 2
[[ "$(sha256sum "$CONTRACT" | cut -d' ' -f1)" == 2e1f97f87ec70a6cf4c3660151f6be28517cbd1aeb2506534168a23344220bbe ]] || exit 2
[[ "$(sha256sum "$SEALER" | cut -d' ' -f1)" == 72dfb6098e8da24118781f0f97f080f45247e2b1302dbe572c0b8a05e0a6ad49 ]] || exit 2
[[ "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" == 2c4491ce31f47ff015dfc34d31d919ef97a94981b071a6ef546f356c330b07d4 ]] || exit 2
module load python/3.11.7
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
umask 077
RUNTIME="$ROOT/runs/routing-control-v1/runtime-extension-reader-qual-$SLURM_JOB_ID"
mkdir -p "$RUNTIME/xdg" "$RUNTIME/triton" "$RUNTIME/torch" "$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg" TRITON_CACHE_DIR="$RUNTIME/triton" TORCH_HOME="$RUNTIME/torch" TMPDIR="$RUNTIME/tmp"
export HF_HOME=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PATH=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin:$PATH
export PYTHONPATH="$REPO/src:$REPO/scripts/experimental_resume"
srun --ntasks=1 "$PY" -B -u "$DRIVER" --qualify

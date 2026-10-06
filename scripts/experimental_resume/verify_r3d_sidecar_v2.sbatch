#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --job-name=r3d-sidecar-verify-v2
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/r3d-sidecar-verify-v2-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/r3d-sidecar-verify-v2-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ $# -eq 2 && ( "$1" == gpt-dest || "$1" == qwen ) ]] || exit 2
KIND=$1
OUT=$2
[[ "$OUT" == /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/R3D_SIDECAR_VERIFY_* ]] || exit 2
module load python/3.11.7
SCRIPT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume/verify_r3d_sidecar_v2.py
EXPECTED=a03985c5415c96d6573709fd40eafe341853b5c9a2eb05f0d9567805c5e1946a
ACTUAL=$(sha256sum "$SCRIPT")
[[ "${ACTUAL%% *}" == "$EXPECTED" ]] || { echo 'R3-D verifier source changed' >&2; exit 3; }
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export TMPDIR=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/r3d-sidecar-verify-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
srun --ntasks=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -u "$SCRIPT" --kind "$KIND" --out "$OUT"

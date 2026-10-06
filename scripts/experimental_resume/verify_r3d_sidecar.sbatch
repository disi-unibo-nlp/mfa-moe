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
#SBATCH --job-name=r3d-sidecar-verify-v1
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/r3d-sidecar-verify-v1-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/r3d-sidecar-verify-v1-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
[[ $# -eq 2 && ( "$1" == gpt-dest || "$1" == qwen ) ]] || exit 2
KIND=$1
OUT=$2
[[ "$OUT" == /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1/R3D_SIDECAR_VERIFY_* ]] || exit 2
module load python/3.11.7
SCRIPT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/experimental_resume/verify_r3d_sidecar.py
EXPECTED=b2bad70fe3fabdd8641fccbf518ae7888d052e31c67fecf9105beef6549d76a7
ACTUAL=$(sha256sum "$SCRIPT")
[[ "${ACTUAL%% *}" == "$EXPECTED" ]] || { echo 'R3-D verifier source changed' >&2; exit 3; }
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export TMPDIR=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/r3d-sidecar-verify-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
srun --ntasks=1 /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python -B -u "$SCRIPT" --kind "$KIND" --out "$OUT"

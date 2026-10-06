#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --job-name=st-neighbor-audit-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
DRIVER="$REPO/scripts/experimental_resume/audit_boundary_neighbor_qual_v1.py"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == f6c5ca6078196eb1eb44dd9b65885957fa615c10ce7b65288adcebe025abdf74 ]] || exit 2
RUNTIME=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/runtime-neighbor-audit-$SLURM_JOB_ID
umask 077
mkdir -p "$RUNTIME"
export TMPDIR="$RUNTIME" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/scripts/experimental_resume"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER"

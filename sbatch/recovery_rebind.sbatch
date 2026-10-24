#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --gres=tmpfs:10g
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr ]] || exit 2
BUNDLE="${2:?Frozen bundle required}"
umask 077
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$BUNDLE" TMPDIR=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp
cd "$BUNDLE"
/usr/bin/python3.11 -B -u -m measurement_study.recovery_rebind "$1" --bundle "$BUNDLE" --input-root "${3:?Input root required}" --output-root "${4:?Output root required}" --cap-gib "${5:?Measured output cap required}"

#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_viz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:01:00
#SBATCH --no-requeue
#SBATCH --job-name=native-finalization-start-signal
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ "$(id -un)" == lmolfett && "$(hostname -s)" != login* &&
   "${SLURM_JOB_ACCOUNT:-}" == iscrc_miosr && "${SLURM_JOB_PARTITION:-}" == lrd_all_viz ]] || exit 2
umask 077
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
srun --ntasks=1 /usr/bin/python3 -B "$REPO/scripts/experimental_resume/native_finalization_start_signal_v1.py" "${1:?watched job id}" "${2:?version}"

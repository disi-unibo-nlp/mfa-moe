#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=st-mechanism-extension-cpu
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett &&
   "${SLURM_JOB_PARTITION:-}" == lrd_all_serial ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
DRIVER="$REPO/scripts/experimental_resume/prepare_mechanism_extension_220_v1.py"
WRAPPER="$REPO/scripts/experimental_resume/run_mechanism_extension_cpu_v1.sh"
VETO="$REPO/scripts/experimental_resume/mechanism_extension_strict_veto_v1.py"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == e3366c7ea2a39e418ceb944405b9e9ec71be77f90c74507f0cd0b01cd6701874 ]] || exit 2
[[ "$(sha256sum "$WRAPPER" | cut -d' ' -f1)" == c5f4ff22021142fa71024d19d21c13e614228b417ea0dd52eb1c888826d9ad3c ]] || exit 2
[[ "$(sha256sum "$VETO" | cut -d' ' -f1)" == 99ec37e06351e1e44ae53a8edcef63fed29daf2f2f6959ba5073b07ff7f4f07c ]] || exit 2
module load python/3.11.7
export EXT_READER_MAX_WALL_SECONDS=86400
exec bash "$WRAPPER" all

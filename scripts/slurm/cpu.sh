#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=lrd_all_serial --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=4 --mem=30G --time=04:00:00 --job-name=cpu
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# Run one Python module on a CPU node: sbatch -J NAME scripts/slurm/cpu.sh MODULE ARGS...
source scripts/slurm/env.sh || exit 1
python -m "$@"

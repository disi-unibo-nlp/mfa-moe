#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --gres=gpu:2 --cpus-per-task=16 --mem=120G --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# One GPU stage of a steering study, run from the checkout (or release snapshot) you submit from: python -m MODULE ARGS...
#   sbatch -t 12:00:00 -J fr-gpt-0 scripts/steering/gpu.sh steering.pulse run --model gpt --shard 0/2
source scripts/slurm/env.sh || exit 1
T=$((SLURM_CPUS_PER_TASK < 4 ? SLURM_CPUS_PER_TASK : 4))  # a few BLAS threads next to the engine
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
python -m "$@"

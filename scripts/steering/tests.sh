#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=lrd_all_serial --qos=normal --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=4 --mem=24G --time=60 --job-name=tests
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# The unit tests of the steering and routing code, from the checkout (or release snapshot) you submit from.
source scripts/slurm/env.sh || exit 1
for dir in tests/steering tests/steering/outcome tests/routing; do
    python -m unittest discover -s $dir -t $dir
done

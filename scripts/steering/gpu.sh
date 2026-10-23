#!/bin/bash
#SBATCH --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/logs/%x-%j.out
# One stage of a steering study, run from the release snapshot: python -m ${STUDY_MODULE:-moe_exp.steering.pulse} ARGS...
#   sbatch --gres=gpu:2 -c 16 --mem=120G -t 12:00:00 -J fr-gpt-0 scripts/steering/gpu.sh run --model gpt --shard 0/2
cd "${RELEASE_ROOT:-$SLURM_SUBMIT_DIR}"
source scripts/slurm/env.sh
T=$((SLURM_CPUS_PER_TASK < 4 ? SLURM_CPUS_PER_TASK : 4))  # a few BLAS threads for the CPU analysis stages
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T VLLM_ENABLE_V1_MULTIPROCESSING=0
python -m "${STUDY_MODULE:-moe_exp.steering.pulse}" "$@"

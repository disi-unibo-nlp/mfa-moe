#!/bin/bash
# Job of the steering study; every argument goes to `python -m ${STUDY_MODULE:-moe_exp.steering.pulse}`. Run from RELEASE_ROOT:
#   sbatch -A iscrc_miosr -p boost_usr_prod --qos=normal -N1 --ntasks=1 --gres=gpu:2 --cpus-per-task=16 \
#          --mem=120G -t 12:00:00 -J fr-gpt-0 scripts/steering/gpu.sh run --model gpt --shard 0/2
cd "${RELEASE_ROOT:-$SLURM_SUBMIT_DIR}"
source scripts/slurm/env.sh
T=$((SLURM_CPUS_PER_TASK < 4 ? SLURM_CPUS_PER_TASK : 4))  # a few BLAS threads for the CPU analysis stages
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T VLLM_ENABLE_V1_MULTIPROCESSING=0
echo "job $SLURM_JOB_ID on $(hostname): $*"
python -B -m "${STUDY_MODULE:-moe_exp.steering.pulse}" "$@"

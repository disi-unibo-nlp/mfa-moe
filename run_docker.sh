#!/bin/bash
#SBATCH --job-name=moe-mfa-exp
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --nodelist=faretra

# Use SLURM_SUBMIT_DIR when running as a Slurm job (avoids spool-directory confusion);
# fall back to script-relative path when run manually.
PHYS_DIR="/home/tassinari/moe-mfaExperiments"
LLM_CACHE_DIR="/llms"
IMAGE_NAME="moe-mfa-experiments:latest"

# Pre-create results dir as the Slurm user so NFS root_squash (which maps
# container root → nobody) can still write into it.
mkdir -p "$PHYS_DIR/results"
chmod -R 777 "$PHYS_DIR/results"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$LLM_CACHE_DIR" \
    -e HF_HOME="$LLM_CACHE_DIR" \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    "$IMAGE_NAME" \
    bash -c "cd /workspace && python -m moe_exp.experiment1.run ${*}"

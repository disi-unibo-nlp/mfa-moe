#!/bin/bash
#SBATCH --job-name=dynamics-replay
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/dynamics-replay-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/slurm_logs/dynamics-replay-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit through Slurm only after workload approval" >&2; exit 2; }
source /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/scripts/routing_dynamics_env.sh
srun --ntasks=1 --nodes=1 "$PYTHON" scripts/run_routing_dynamics.py replay --root "$ROOT"
srun --ntasks=1 --nodes=1 "$PYTHON" scripts/run_routing_dynamics.py accept --root "$ROOT"
# CPU analysis shares this allocation's hard four-hour ceiling; no automatic extension.
srun --ntasks=1 --nodes=1 "$PYTHON" scripts/run_routing_dynamics.py analyze --root "$ROOT"

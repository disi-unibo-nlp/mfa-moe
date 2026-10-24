#!/bin/bash
#SBATCH --job-name=mfa-v3-cpu-tests
#SBATCH --account=iscrc_miosr
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/campaign-v3/tests/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/campaign-v3/tests/%x-%j.err
set -uo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett ]] || exit 2
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study
P=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe
OUT="$ROOT/results/measurement-study/campaign-v3/tests"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="$OUT/tmp-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
cd "$ROOT" || exit 2
code=0
for group in measurement guiding; do
 srun "$P/envs/vllm-cu129/bin/python" -B scripts/test_campaign_v3.py --group "$group" --output "$OUT/$SLURM_JOB_ID-$group.json" || code=1
done
exit "$code"

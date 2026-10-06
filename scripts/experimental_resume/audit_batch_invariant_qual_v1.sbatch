#!/bin/bash
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --job-name=st-batch-inv-audit-v1
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/logs/%x-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname -s)" != login* ]] || exit 2
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1
DRIVER="$REPO/scripts/experimental_resume/audit_batch_invariant_qual.py"
RUN_OUT="$ROOT/runs/routing-control-v1/batch-invariant-qual-db182a1debc7465c"
[[ "$(sha256sum "$DRIVER" | cut -d' ' -f1)" == 6a6c3e10f70bc803d1a66f23f92a465ecd822aeffda4cc028eafb962322cd7f5 ]] || exit 2
[[ "$(sha256sum "$REPO/scripts/experimental_resume/analyze_micro_first_stage.py" | cut -d' ' -f1)" == 93c53297b6d12ffbaab7b4b6613365b1f4d7c14becdea6c5f2e81e34caddd321 ]] || exit 2
[[ "$(sha256sum "$REPO/scripts/experimental_resume/audit_native_only_replay_v3.py" | cut -d' ' -f1)" == 1bb860d80eba4ce1b7d1a7e3835ca44b5b1dc3d14229dc9c60bfca295cc439c1 ]] || exit 2
[[ -s "$RUN_OUT/SUMMARY.json" ]] || exit 2
module load python/3.11.7
umask 077
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
PY=/leonardo_work/IscrC_MIOSR/lmolfett/envs/vllm-cu129/bin/python
srun --ntasks=1 "$PY" -B -u "$DRIVER"

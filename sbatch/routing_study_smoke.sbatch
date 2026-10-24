#!/bin/bash
#SBATCH --job-name=qwen36-routing-smoke
#SBATCH --account=iscrc_miosr
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/routing-study/logs/%x-%j.out
#SBATCH --error=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/routing-study/logs/%x-%j.err
set -euo pipefail
[[ "${SLURM_JOB_ID:-}" && "$(id -un)" == lmolfett && "$(hostname)" == lrdn* ]] || exit 2
source /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/scripts/routing_study_env.sh
STUDY_JOB_DESCRIPTION=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$STUDY_JOB_DESCRIPTION" == *" QOS=boost_qos_dbg "* ]] || exit 2
STUDY_ORIGINAL="$STUDY_PROJECT/repo/results/correlation_pipeline"
STUDY_SOURCES="$STUDY_ORIGINAL/qwen36-35b-a3b-fp8/generation/Qwen--Qwen3.6-35B-A3B-FP8"
STUDY_FIXTURES="$STUDY_ORIGINAL/forward-validation-v2/panels/qwen36/generation/Qwen--Qwen3.6-35B-A3B-FP8/math500/traces.jsonl"
# This preparation scans the large saved corpus on the allocated node, without fitting policies.
srun env TMPDIR="$STUDY_STAGE" "$STUDY_PYTHON" -B scripts/run_routing_study.py prepare \
  --sources "$STUDY_SOURCES"/{math500,aime24,aime25,amc23,minerva,olympiad}/traces.jsonl \
  --pilot "$STUDY_ORIGINAL/routing-dynamics-v1/pilot/pilot.json" --output "$STUDY_RUN/study.json"
srun env TMPDIR="$STUDY_STAGE" CUDA_VISIBLE_DEVICES='' "$STUDY_PYTHON" -B scripts/verify_routing_study.py \
  --output "$STUDY_RUN/cpu-tests.json" --base-temp "$STUDY_STAGE/tests"
srun env TMPDIR="$STUDY_STAGE" "$STUDY_PYTHON" -B scripts/run_routing_study.py smoke --fixtures "$STUDY_FIXTURES" \
  --cpu-report "$STUDY_RUN/cpu-tests.json" --output "$STUDY_RUN/smoke"
srun env TMPDIR="$STUDY_STAGE" "$STUDY_PYTHON" -B scripts/run_routing_study.py proposal --study "$STUDY_RUN/study.json" \
  --smoke "$STUDY_RUN/smoke/acceptance.json" --conditions 16 --output "$STUDY_RUN/resource-proposal.json"

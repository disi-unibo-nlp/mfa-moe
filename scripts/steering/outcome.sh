#!/bin/bash
# Seal an outcome-steering release from the committed code, submit its whole-reasoning job graph
# (allocated validation -> shards -> extraction, plus the monitor) and queue its analysis.
#   scripts/steering/outcome.sh NAME SHARDS LOCAL_TESTS [MODELS]
#   SHARDS is N per model or model:N,...; MODELS defaults to gpt,qwen330b,nemotron.
#   e.g. scripts/steering/outcome.sh outcome-steering-20261022-r1 3 125
# Needs the usage job's output (scripts/steering/outcome_usage.sh) and a clean src/tests/scripts tree.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
name="${1:?release name}"; shards="${2:?shards per model}"; tests="${3:?locally passed tests}"
models="${4:-gpt,qwen330b,nemotron}"
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
TMP=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/outcome-steering/tmp
mkdir -p "$TMP"

# 1. Seal: receipts, OUTCOME.json, the code snapshot under mfa-moe/releases/<commit> and launch.env.
TMPDIR="$TMP" PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -B -m moe_exp.steering.outcome release --name "$name" --models "$models" --shards "$shards" --local-tests "$tests"
set -a; source "report/$name/launch.env"; set +a
REL=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/releases/$COMMIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$REL/src TMPDIR="$TMP" \
       STUDY_DATA_ROOT=$REPO/report/final-routing-20261015-r1 WHOLE_ROOT=$REPO/report/$name \
       WHOLE_MODELS=$MODELS WHOLE_SWEEP=$SWEEP WHOLE_SHARDS=$SHARDS WHOLE_CUTOFF=$CUTOFF \
       WHOLE_GPU_CEILING=$GPU_CEILING WHOLE_CPU_CEILING=$CPU_CEILING OMP_NUM_THREADS=1

# 2. The ordinary whole-reasoning graph runs the release (no canary: it is a sweep release).
(cd "$REL" && "$PY" -B -m moe_exp.steering.whole launch)

# 3. Analysis: after every extraction of this release and the final whole-reasoning combine, whose
#    sealed table holds the reused native runs.
FINAL_COMBINE=61741779
extract=$("$PY" -c "import json; print(':'.join(j['id'] for j in json.load(open('$WHOLE_ROOT/JOBS.json'))['extraction'].values()))")
test -n "$extract" || { echo 'launch queued no extraction jobs' >&2; exit 1; }
env -u SLURM_JOB_ID STUDY_MODULE=moe_exp.steering.outcome RELEASE_ROOT=$REL \
  sbatch --parsable --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=64G --time=120 --job-name=whole-os-analyze --chdir="$REL" \
  --dependency="afterany:$extract:$FINAL_COMBINE" --export=ALL \
  --output="$WHOLE_ROOT/logs/%x-%j.out" --error="$WHOLE_ROOT/logs/%x-%j.err" \
  "$REL/scripts/steering/gpu.sh" analyze | tee "$WHOLE_ROOT/ANALYSIS_JOB"

#!/bin/bash
# Seal an outcome-steering or capture release from the committed code and submit its whole-reasoning
# job graph (allocated validation -> shards -> extraction, plus the monitor). An outcome release also
# queues its analysis.
#   scripts/outcome_steering/launch.sh NAME SHARDS LOCAL_TESTS [MODELS] [KIND]
#   SHARDS is N per model or model:N,...; MODELS defaults to gpt,qwen330b,nemotron; KIND is outcome
#   (default) or capture.
#   e.g. scripts/outcome_steering/launch.sh outcome-capture-20261023-r1 gpt:3,glm:22 140 gpt,glm capture
# Receipts go to the working repository's report/ (REPORTS), also when this runs from a worktree.
set -euo pipefail
cd "$(dirname "$0")/../.."
name="${1:?release name}"; shards="${2:?shards per model}"; tests="${3:?locally passed tests}"
models="${4:-gpt,qwen330b,nemotron}"; kind="${5:-outcome}"
REPORTS=${REPORTS:-/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report}
PY=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/envs/vllm-cu129/bin/python
TMP=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/outcome-steering/tmp
mkdir -p "$TMP"

# 1. Seal: receipts, the spec file, the code snapshot under mfa-moe/releases/<commit> and launch.env.
WHOLE_ROOT=$REPORTS/$name TMPDIR="$TMP" PYTHONPATH=src:scripts/experimental_resume PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -B -m moe_exp.outcome_steering release --name "$name" --models "$models" --shards "$shards" \
  --local-tests "$tests" --kind "$kind"
set -a; source "$REPORTS/$name/launch.env"; set +a
REL=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/releases/$COMMIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$REL/src:$REL/scripts/experimental_resume TMPDIR="$TMP" \
       STUDY_DATA_ROOT=$REPORTS/final-routing-20261015-r1 WHOLE_ROOT=$REPORTS/$name \
       WHOLE_MODELS=$MODELS WHOLE_SWEEP=$SWEEP WHOLE_SHARDS=$SHARDS WHOLE_CUTOFF=$CUTOFF \
       WHOLE_GPU_CEILING=$GPU_CEILING WHOLE_CPU_CEILING=$CPU_CEILING WHOLE_JOB_PREFIX=$JOB_PREFIX OMP_NUM_THREADS=1

# 2. The ordinary whole-reasoning graph runs the release (no canary: it is a sweep release).
(cd "$REL" && "$PY" -B -m moe_exp.study.whole launch)
[ "$kind" = outcome ] || exit 0

# 3. Analysis (r1 only): after every extraction of this release and the final whole-reasoning combine,
#    whose sealed table holds the reused native runs.
FINAL_COMBINE=61741779
extract=$("$PY" -c "import json; print(':'.join(j['id'] for j in json.load(open('$WHOLE_ROOT/JOBS.json'))['extraction'].values()))")
test -n "$extract" || { echo 'launch queued no extraction jobs' >&2; exit 1; }
env -u SLURM_JOB_ID STUDY_MODULE=moe_exp.outcome_steering RELEASE_ROOT=$REL \
  sbatch --parsable --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=64G --time=120 --job-name=whole-os-analyze --chdir="$REL" \
  --dependency="afterany:$extract:$FINAL_COMBINE" --export=ALL \
  --output="$WHOLE_ROOT/logs/%x-%j.out" --error="$WHOLE_ROOT/logs/%x-%j.err" \
  "$REL/scripts/study/driver.sh" analyze | tee "$WHOLE_ROOT/ANALYSIS_JOB"

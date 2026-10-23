#!/bin/bash
# Seal an outcome-steering, capture or sweep release from the committed code and submit its
# whole-reasoning job graph (allocated validation -> shards -> extraction, plus the monitor).
# Outcome and sweep releases also queue their analysis; a half-A sweep then queues the pick job,
# which seals and launches half B.
#   scripts/outcome_steering/launch.sh NAME SHARDS LOCAL_TESTS [MODELS] [KIND]
#   SHARDS is N per model or model:N,...; MODELS defaults to gpt,qwen330b,nemotron; KIND is outcome
#   (default), capture, sweep-A or sweep-B (reads PICK.json from $SWEEP_PICK).
#   A sweep-A launch needs SWEEP_B_NAME and SWEEP_B_SHARDS for the half-B release its pick launches.
#   e.g. scripts/outcome_steering/launch.sh outcome-capture-20261023-r1 gpt:3,glm:15 60 gpt,glm capture
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
[ "$kind" = capture ] && exit 0

# 3. Analysis after every extraction of this release (r1 also waits for the final whole-reasoning
#    combine, whose sealed table holds its reused native runs). Nested sbatch drops every SLURM_* variable.
unslurm() { env $(compgen -e | grep -E '^(SLURM|SBATCH|SRUN)_' | sed 's/^/-u /') "$@"; }
extract=$("$PY" -c "import json; print(':'.join(j['id'] for j in json.load(open('$WHOLE_ROOT/JOBS.json'))['extraction'].values()))")
test -n "$extract" || { echo 'launch queued no extraction jobs' >&2; exit 1; }
after="afterany:$extract"; [ "$kind" = outcome ] && after="$after:61741779"
analysis=$(unslurm STUDY_MODULE=moe_exp.outcome_steering RELEASE_ROOT=$REL \
  sbatch --parsable --account=iscrc_miosr --partition=boost_usr_prod --qos=normal --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=64G --time=120 --job-name=$JOB_PREFIX-analyze --chdir="$REL" \
  --dependency="$after" --export=ALL \
  --output="$WHOLE_ROOT/logs/%x-%j.out" --error="$WHOLE_ROOT/logs/%x-%j.err" \
  "$REL/scripts/study/driver.sh" analyze)
echo "$analysis" | tee "$WHOLE_ROOT/ANALYSIS_JOB"
[ "$kind" = sweep-A ] || exit 0

# 4. Half A: the pick job applies the frozen rule and launches half B from this checkout.
pick=$(unslurm sbatch --parsable -A iscrc_miosr -p lrd_all_serial --qos=normal -N1 --ntasks=1 --cpus-per-task=4 \
  --mem=30G -t 02:00:00 -J os-pick --dependency="afterok:$analysis" \
  -o "$WHOLE_ROOT/logs/%x-%j.out" -e "$WHOLE_ROOT/logs/%x-%j.err" \
  "$PWD/scripts/outcome_steering/pick.sh" "$WHOLE_ROOT/analysis/EFFECTS_OUTCOME.json" "$MODELS" \
  "${SWEEP_B_NAME:?half-B release name}" "${SWEEP_B_SHARDS:?half-B shards}" "$tests")
echo "$pick" | tee "$WHOLE_ROOT/PICK_JOB"

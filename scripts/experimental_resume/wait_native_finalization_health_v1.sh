#!/bin/bash
# Finite scheduler readiness wait, then node-local file events in an existing job.
set -euo pipefail
[[ "$(id -un)" == lmolfett && "$(hostname -f)" == *.leonardo.local ]] || exit 2
JOB=${1:?existing job id}
PHASE=${2:?preparation, qualification, generation or measurement}
VERSION=${3:-correction-v2}
[[ "$JOB" =~ ^[0-9]+$ ]] || exit 2
[[ "$VERSION" =~ ^correction-v[0-9]+$ ]] || exit 2
case "$PHASE" in
    preparation|measurement) HEALTH_SECONDS=3500; STEP_TIME=01:00:00 ;;
    qualification) HEALTH_SECONDS=1800; STEP_TIME=00:31:00 ;;
    generation) HEALTH_SECONDS=28000; STEP_TIME=08:00:00 ;;
    *) exit 2 ;;
esac
REPO=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
ROOT=/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/native-finalization-v1/$VERSION
DOC=$REPO/report/experimental-resume-v1/native-finalization-v1/$VERSION
snapshot() {
    /usr/bin/python3.11 -B - "$REPO" "$DOC" "$ROOT" "$JOB" "$PHASE" <<'PY'
import pathlib, sys
sys.path.insert(0, sys.argv[1] + '/scripts/experimental_resume')
import native_finalization_health_v1 as H
H.DOC, H.ROOT = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
result = H.status(sys.argv[5], sys.argv[4])
if result is None:
    sys.exit(3)
H.emit({'job_id': sys.argv[4], 'phase': sys.argv[5], **result,
        'snapshot': 'one initial or termination-triggered read; no repeated file checks'})
PY
}
if snapshot; then exit 0; fi
printf '{"observer":"WAITING_FOR_ALLOCATION","job_id":"%s","phase":"%s"}\n' "$JOB" "$PHASE"
if ! timeout 28800 scontrol wait_job "$JOB"; then
    snapshot
    exit $?
fi
exec srun --jobid="$JOB" --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --gres=none --mem=64M --export=NONE --kill-on-bad-exit=0 --time="$STEP_TIME" \
    --chdir="$REPO" --unbuffered /usr/bin/python3 -B \
    "$REPO/scripts/experimental_resume/native_finalization_health_v1.py" \
    --phase "$PHASE" --job-id "$JOB" --timeout-seconds "$HEALTH_SECONDS" --doc "$DOC" --root "$ROOT"

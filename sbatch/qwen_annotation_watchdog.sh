#!/bin/bash
# Restart the annotation client when its checkpoint writes stall.
#
# Usage:
#   qwen_annotation_watchdog.sh --output-dir DIR --client-log FILE \
#       [--stall-seconds N] [--poll-seconds N] [--max-restarts N] -- COMMAND ...
#
# The client runs in the background and appends its stdout to --client-log; its stderr
# is inherited so Python stack dumps (MFA_FAULTHANDLER) land in the Slurm .err file.
# Every --poll-seconds the wrapper looks at the newest checkpoints/batch-*.json mtime.
# With no published checkpoint for --stall-seconds it logs scheduler/wchan/stack
# diagnostics, SIGTERMs the client, SIGKILLs it after 30s and restarts it; the client
# resumes from its own checkpoints. A non-zero exit that is not a detected stall is a
# genuine error and is never retried, so the job fails for investigation.
set -uo pipefail

usage() {
    printf '%s\n' 'Usage: qwen_annotation_watchdog.sh --output-dir DIR --client-log FILE [--stall-seconds N] [--poll-seconds N] [--max-restarts N] -- COMMAND ...'
}

OUTPUT_DIR=""
CLIENT_LOG=""
STALL_SECONDS=1500
POLL_SECONDS=60
MAX_RESTARTS=24
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
        --client-log) CLIENT_LOG=${2:?}; shift 2 ;;
        --stall-seconds) STALL_SECONDS=${2:?}; shift 2 ;;
        --poll-seconds) POLL_SECONDS=${2:?}; shift 2 ;;
        --max-restarts) MAX_RESTARTS=${2:?}; shift 2 ;;
        --) shift; break ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n "$OUTPUT_DIR" && -n "$CLIENT_LOG" && $# -gt 0 ]] || { usage >&2; exit 2; }
[[ "$STALL_SECONDS" =~ ^[0-9]+$ && "$POLL_SECONDS" =~ ^[0-9]+$ && "$MAX_RESTARTS" =~ ^[0-9]+$ ]] \
    || { echo '--stall-seconds, --poll-seconds and --max-restarts must be integers' >&2; exit 2; }
mkdir -p "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/logs"

# Newest published checkpoint mtime, or 0 when nothing has been published yet.
newest_checkpoint_epoch() {
    local newest=0 file mtime
    for file in "$OUTPUT_DIR"/checkpoints/batch-*.json; do
        [[ -e "$file" ]] || return 0
        mtime=$(stat -c %Y "$file" 2>/dev/null) || continue
        if (( mtime > newest )); then newest=$mtime; fi
    done
    printf '%s' "$newest"
}

stall_diagnostics() {
    local pid=$1 idle=$2
    local target="$OUTPUT_DIR/logs/stall-diagnostics-$(date +%Y%m%dT%H%M%S).log"
    {
        printf 'MFA_STALL_DETECTED pid=%s idle_seconds=%s at=%s output_dir=%s\n' \
            "$pid" "$idle" "$(date --iso-8601=seconds)" "$OUTPUT_DIR"
        ps -o pid,ppid,stat,wchan:32,etime,cmd -p "$pid" || true
        ps -o pid,ppid,stat,wchan:32,etime,cmd --ppid "$pid" || true
        printf 'parent_wchan='; cat "/proc/$pid/wchan" 2>/dev/null || printf 'unreadable\n'
        printf 'parent_stack:\n'; cat "/proc/$pid/stack" 2>/dev/null || printf 'unreadable\n'
        printf 'children_wchan:\n'
        for child in $(pgrep -P "$pid" 2>/dev/null || true); do
            printf '  pid=%s wchan=' "$child"; cat "/proc/$child/wchan" 2>/dev/null || printf 'unreadable\n'
            printf '  pid=%s stack:\n' "$child"; cat "/proc/$child/stack" 2>/dev/null || printf '  unreadable\n'
        done
        printf 'newest_checkpoints:\n'; ls -lt "$OUTPUT_DIR"/checkpoints 2>/dev/null | head -5 || true
        printf 'scratch_free_kb='; df -k --output=avail "$OUTPUT_DIR" 2>/dev/null | tail -1 || printf 'unknown\n'
    } 2>&1 | tee -a "$target"
}

child_pid=""
terminating=0
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    terminating=1
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$child_pid" 2>/dev/null || break; sleep 1; done
        kill -KILL "$child_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

restarts=0
attempt=0
while :; do
    attempt=$((attempt + 1))
    started=$(date +%s)
    printf 'annotation_client_start attempt=%s client_log=%s\n' "$attempt" "$CLIENT_LOG"
    "$@" >>"$CLIENT_LOG" &
    child_pid=$!
    stalled=0
    child_status=0
    while kill -0 "$child_pid" 2>/dev/null; do
        sleep "$POLL_SECONDS"
        kill -0 "$child_pid" 2>/dev/null || break
        reference=$(newest_checkpoint_epoch)
        if (( reference < started )); then reference=$started; fi
        idle=$(( $(date +%s) - reference ))
        printf 'annotation_watchdog attempt=%s client_pid=%s idle_seconds=%s checkpoints=%s\n' \
            "$attempt" "$child_pid" "$idle" "$(ls "$OUTPUT_DIR"/checkpoints 2>/dev/null | wc -l)"
        if (( idle >= STALL_SECONDS )); then
            stalled=1
            stall_diagnostics "$child_pid" "$idle"
            printf 'annotation_watchdog_restart attempt=%s client_pid=%s\n' "$attempt" "$child_pid"
            kill -TERM "$child_pid" 2>/dev/null || true
            for _ in $(seq 1 30); do kill -0 "$child_pid" 2>/dev/null || break; sleep 1; done
            kill -KILL "$child_pid" 2>/dev/null || true
            break
        fi
    done
    wait "$child_pid" || child_status=$?
    child_pid=""
    if (( terminating )); then
        exit 1
    fi
    if (( stalled )); then
        if (( restarts >= MAX_RESTARTS )); then
            printf 'annotation_watchdog_giving_up attempts=%s max_restarts=%s\n' "$attempt" "$MAX_RESTARTS" >&2
            exit 1
        fi
        restarts=$((restarts + 1))
        printf 'annotation_client_restart attempt=%s exit_code=%s restarts=%s reason=stalled\n' "$attempt" "$child_status" "$restarts"
        continue
    fi
    if (( child_status != 0 )); then
        printf 'annotation_client_failed attempt=%s exit_code=%s (not retried: non-stall error)\n' "$attempt" "$child_status" >&2
        exit "$child_status"
    fi
    printf 'annotation_client_complete attempt=%s\n' "$attempt"
    break
done

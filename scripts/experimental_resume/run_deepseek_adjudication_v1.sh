#!/usr/bin/env bash
# Credential-free follow-up wrapper; the existing local launcher loads its key privately.
set -euo pipefail

review_tag="${1:?expected A or D}"
case "$review_tag" in
  A) session=01a0fcdd-64ad-72e1-9805-ee15a35af888 ;;
  D) session=01a0fcea-5c35-7e01-9f53-b1f38707d1d7 ;;
  *) printf 'review tag must be A or D\n' >&2; exit 2 ;;
esac

repo=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
prompt="$repo/report/experimental-resume-v1/DEEPSEEK_ADJUDICATION_PROMPT_${review_tag}_2026-10-02.txt"
stem="$repo/report/experimental-resume-v1/DEEPSEEK_ADJUDICATION_2026-10-02_$review_tag"

codex() {
  /leonardo/home/userexternal/lmolfett/.local/bin/codex exec resume \
    --json -o "$stem.md" "$@" "$session" -
}

source "$repo/local_codex/launch_deepseek_codex.sh" \
  < "$prompt" > "$stem.jsonl" 2> "$stem.stderr"

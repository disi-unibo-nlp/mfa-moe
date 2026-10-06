#!/usr/bin/env bash
# Read-only follow-up to the completed independent DeepSeek audit.
set -euo pipefail

repo=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
prompt="$repo/report/experimental-resume-v1/DEEPSEEK_REVIEW_DEBATE_PROMPT_2026-10-02_A.md"
stem="$repo/report/experimental-resume-v1/DEEPSEEK_REVIEW_DEBATE_2026-10-02_A"
thread=01a0fcdd-64ad-72e1-9805-ee15a35af888

codex() {
  /leonardo/home/userexternal/lmolfett/.local/bin/codex exec resume \
    --json -o "$stem.md" "$@" "$thread" -
}

source "$repo/local_codex/launch_deepseek_codex.sh" \
  < "$prompt" > "$stem.jsonl" 2> "$stem.stderr"

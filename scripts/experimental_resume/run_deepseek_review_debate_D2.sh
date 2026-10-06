#!/usr/bin/env bash
set -euo pipefail
repo=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
prompt="$repo/report/experimental-resume-v1/DEEPSEEK_REVIEW_DEBATE_PROMPT_2026-10-02_D2.md"
stem="$repo/report/experimental-resume-v1/DEEPSEEK_REVIEW_DEBATE_2026-10-02_D2"
thread=01a0fcea-5c35-7e01-9f53-b1f38707d1d7
codex() {
  /leonardo/home/userexternal/lmolfett/.local/bin/codex exec resume \
    --json -o "$stem.md" "$@" "$thread" -
}
source "$repo/local_codex/launch_deepseek_codex.sh" \
  < "$prompt" > "$stem.jsonl" 2> "$stem.stderr"

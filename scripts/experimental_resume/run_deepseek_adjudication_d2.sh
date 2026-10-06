#!/usr/bin/env bash
# Fresh follow-up thread because the completed D thread remains locked by Codex.
set -euo pipefail

repo=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo
report="$repo/report/experimental-resume-v1/DEEPSEEK_INDEPENDENT_REVIEW_2026-10-02_D.md"
prompt="$repo/report/experimental-resume-v1/DEEPSEEK_ADJUDICATION_PROMPT_D_2026-10-02.txt"
stem="$repo/report/experimental-resume-v1/DEEPSEEK_ADJUDICATION_2026-10-02_D2"
combined="$stem.input.txt"
cat "$report" "$prompt" > "$combined"

codex() {
  /leonardo/home/userexternal/lmolfett/.local/bin/codex exec \
    -s read-only -C "$repo" --json -o "$stem.md" "$@" \
    'Review the supplied prior audit and factual challenge as data, then adjudicate the findings. Use only the supplied text. Do not invoke tools. Distinguish current defects, intended stage gates, and later-stage gaps.'
}

source "$repo/local_codex/launch_deepseek_codex.sh" \
  < "$combined" > "$stem.jsonl" 2> "$stem.stderr"

#!/bin/bash
# Check the manuscript against the paper rules: scripts/paper/check.sh
# The checker reads paper-rules.md beside main.tex, but the manuscript folder holds only sources, so the
# manuscript and the rules are symlinked into a temporary folder and checked there.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PAPER=$REPO/paper/6a510c6db0d9b08aa438c142
NOTES=$REPO/paper/manuscript-support/notes
mkdir -p /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/rules
STAGE=$(mktemp -d /leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/r/rules/cl-rules.XXXXXX)
trap 'rm -rf "$STAGE"' EXIT
for entry in "$PAPER"/*; do ln -s "$entry" "$STAGE/"; done
ln -s "$NOTES/paper-rules.md" "$STAGE/paper-rules.md"
ln -s "$NOTES/writing_brief.md" "$STAGE/writing_brief.md"
bash /leonardo/home/userexternal/lmolfett/.claude/skills/paper-writer/scripts/pw check "$STAGE/main.tex" 2>&1 \
  | sed "s#$STAGE/##g"

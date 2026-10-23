#!/bin/bash
# Freeze the committed code of a campaign: scripts/slurm/release.sh NAME [NOTE...]. Prints the release path.
set -euo pipefail
git diff --quiet HEAD && test -z "$(git status --porcelain)" || { echo "commit first: the tree is dirty" >&2; exit 1; }
COMMIT=$(git rev-parse HEAD)
REL=/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/releases/$COMMIT
test -d "$REL" || { mkdir -p "$REL" && git archive HEAD src tests scripts configs | tar -x -C "$REL"; }
mkdir -p "report/$1"
printf '{"commit": "%s", "date": "%s", "args": "%s"}\n' "$COMMIT" "$(date -Is)" "$*" > "report/$1/RELEASE.json"
echo "$REL"

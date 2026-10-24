#!/bin/bash
# Billing ledger of the jobs whose names start with PREFIX (default fr-, the pulsed-steering jobs; whole- for the
# whole-reasoning releases) since 2026-10-17, against the billing-hour cap (CAP, default 6000). Running jobs are
# charged their elapsed time so far plus the rest of their time limit (outstanding commitment). Login node only.
#   scripts/slurm/billing_hours_used.sh [PREFIX]
set -euo pipefail
CAP=${CAP:-6000}
PREFIX=${1:-fr-}
sacct -u lmolfett -S 2026-10-17T00:00:00 -X -n -P \
      --format=JobID,JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES |
awk -F'|' -v cap="$CAP" -v prefix="$PREFIX" '
  index($2, prefix) == 1 {
    b = 0; n = split($6, t, ",");
    for (i = 1; i <= n; i++) if (t[i] ~ /^billing=/) { split(t[i], kv, "="); b = kv[2] }
    used = $4 * b / 3600;
    out = ($3 == "RUNNING" || $3 == "PENDING") ? ($5 * 60 - $4) * b / 3600 : 0;
    if ($3 == "PENDING") { out = $5 * 60 * (b > 0 ? b : 16) / 3600 }
    U += used; O += out;
    printf "%-10s %-22s %-10s %8.1f h used %8.1f h outstanding\n", $1, $2, $3, used, out
  }
  END { printf "TOTAL used %.1f + outstanding %.1f = %.1f of cap %d billing h\n", U, O, U + O, cap }'

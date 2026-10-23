#!/bin/bash
# Billing ledger of the reduced final-routing driver jobs (names fr-*) since 2026-10-17,
# against the 6,000 billing-hour cap. Running jobs are charged their elapsed time so far
# plus the rest of their time limit (outstanding commitment). Login node only.
set -euo pipefail
CAP=${CAP:-6000}
sacct -u lmolfett -S 2026-10-17T00:00:00 -X -n -P \
      --format=JobID,JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES |
awk -F'|' -v cap="$CAP" '
  $2 ~ /^fr-/ {
    b = 0; n = split($6, t, ",");
    for (i = 1; i <= n; i++) if (t[i] ~ /^billing=/) { split(t[i], kv, "="); b = kv[2] }
    used = $4 * b / 3600;
    out = ($3 == "RUNNING" || $3 == "PENDING") ? ($5 * 60 - $4) * b / 3600 : 0;
    if ($3 == "PENDING") { out = $5 * 60 * (b > 0 ? b : 16) / 3600 }
    U += used; O += out;
    printf "%-10s %-22s %-10s %8.1f h used %8.1f h outstanding\n", $1, $2, $3, used, out
  }
  END { printf "TOTAL used %.1f + outstanding %.1f = %.1f of cap %d billing h\n", U, O, U + O, cap }'

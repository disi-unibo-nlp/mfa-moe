# R6 operations amendment

This entrypoint preserves the scientific r6 source seal and execution identities.
It binds corrected reader cleanup, actual validation gates, report-only exports,
compact local report commits and an independently protected October 15 snapshot.

Use the existing vLLM environment with `python -B`. Commands take the same
`--campaign` or immutable `--card` arguments as r6. `prepare` binds this amendment;
`preflight` performs small CPU checks; `protect-milestone` accepts the report-only
owners before retiring verified pending predecessors; `validate-submit` submits
the one authorized four-CPU/24-GiB/three-hour allocation. Successful real
validation launches the protected scientific chain. Failure grants no retry.

Scientific workers and `submit` require the complete, matching reader/export/
graph/readiness and operations proofs. `status` is read-only. Reporting uses
`LOCAL_REPORT_COMPLETE`, never a fabricated manuscript-completion receipt.
Compact report commits are read back on `prepared/<campaign>/ops-report-*` refs;
the working HEAD and index stay unchanged. Remote writes are disabled.

Canonical manuscript files and the installed inference environment are not
modified. All cost gates, original unknown outcomes and no-redraw rules persist.

# Automatic campaign continuation

**Revised authorization — 2026-10-09:** [ADR16](DECISIONS.md#adr-20261009-16-authorized-campaign-v2-instrumentation-repair-and-handoff)
authorizes the repaired v2 campaign, staged smoke gates and protected transfer
from v1. Current receipts belong to
`report/research-campaign-v2-20261009-r1/`; inspect `LAUNCH.json`,
`OWNERSHIP_TRANSFER.json`, `handoffs/` and compact launcher `status` for observed
owners. The October 8 stopped historical controllers below remain stopped.

**USER STOP — 2026-10-08 (ADR20261008 12):** all 15 owned Slurm jobs were cancelled at the user's explicit request; the live queue is empty. Both campaigns' scientific work and automatic continuation are stopped. This supersedes prior dispatch/continuation authorization, including ADR06 and ADR11. Do not submit, retry, requeue, release or restart controllers until the user explicitly authorizes resumed/revised work after reviewing claims and reuse. Preserve completed outputs, sealed sources and immutable receipts. Earlier statuses below are historical. [Verified stop receipt](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json).

Generated from source-bound controller receipts. Scheduler completion alone does not establish a scientific result.

Current controller: `59708614`. State: `WAITING_FOR_SCIENCE`.

Scientific jobs observed: 1 running; 1 pending.

Successor: `59711143`. Failure rescue: `59708624`.

All seven models retain the same required gates. The controller has no timer or persistent daemon.

Receipt: `/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/candidate-verification-v3-20261007-r4/automation/runs/59708614/RESULT.json`.

Read `report/verification-study-v3/AUTOMATION_STATUS.json` for exact actions, errors, job IDs and source bindings.

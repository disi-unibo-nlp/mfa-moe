# Experiment status

**USER STOP — 2026-10-08 (ADR20261008 12):** all 15 owned Slurm jobs were cancelled at the user's explicit request; the live queue is empty. Both campaigns' scientific work and automatic continuation are stopped. This supersedes prior dispatch/continuation authorization, including ADR06 and ADR11. Do not submit, retry, requeue, release or restart controllers until the user explicitly authorizes resumed/revised work after reviewing claims and reuse. Preserve completed outputs, sealed sources and immutable receipts. Earlier statuses below are historical. [Verified stop receipt](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json).

Generated from the active campaign and observed receipts. Jobs and backend qualification do not establish semantic effects.

| Model | Backend | Recognizer | Dose/control | Causal effect | Utility |
|---|---|---|---|---|---|
| gpt | PASS | FAIL | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| nemotron | PASS | FAIL | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| qwen35 | PASS | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| qwen36 | PASS | FAIL | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| gemma | PASS | FAIL | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| glm | PASS | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| qwen330b | PASS | FAIL | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

Campaign: `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/verification-study-v3/CAMPAIGN.json`. Provenance: `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/paper/6a510c6db0d9b08aa438c142/generated/PROVENANCE.json`.

The main direction and stage contracts remain in [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md).

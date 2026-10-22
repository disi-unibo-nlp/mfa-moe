# Outcome-steering commands

Thin bash wrappers around `python -m moe_exp.outcome_steering`. The code and its explanation are in
`src/moe_exp/outcome_steering/README.md`.

| Script | When | What it writes |
| --- | --- | --- |
| `usage.sh [MODELS]` | First. A CPU job on the serial partition, about 5 minutes. | `campaign-runs/outcome-steering/<model>/USAGE.{npz,json}` on FAST. This is how often each expert is used per reasoning token in every cohort-B attempt. |
| `launch.sh NAME SHARDS LOCAL_TESTS` | Once the code is committed and the usage job finished. | `report/NAME/`: receipts, a frozen code snapshot, the whole-reasoning job graph (validation, shards, extraction, monitor) and an analysis job. The analysis waits for this release's extractions and the final whole-reasoning combine. |

Defaults: models `gpt,qwen330b,nemotron`, deltas 0.4/0.8/1.6 and length-limit fractions 0.5/0.7/0.85.
GPU work stops at 2026-10-23 23:00 Rome. Logs: `mfa-moe/r/outcome-steering/logs/` (usage) and
`report/NAME/logs/` (release jobs).

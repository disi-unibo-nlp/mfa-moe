# Outcome-steering commands

Thin bash wrappers around `python -m moe_exp.outcome_steering`. The code and its explanation are in
`src/moe_exp/outcome_steering/README.md`.

## Second campaign (r2, ledger F-0013, F-0014, D-0006)

| Script | When | What it does |
| --- | --- | --- |
| `launch.sh NAME SHARDS TESTS MODELS capture` | First, once the code is committed. | Seals a capture release (`report/NAME/`, stores on FAST) and submits validation, GPU shards, extraction and a monitor. Every model answers the same 300 questions six times with its routing saved. |
| `derive.sh SNAPSHOT RELEASE MODEL` | A CPU job per model, after that model's extraction. | Labels the captured sentences, counts expert use per reasoning class, selects the expert sets and writes `SELECTION_R2.json`, `SIGNAL.json` and `ATTEMPT_SELECTION.json` in `RELEASE/MODEL/`. |
| `launch.sh NAME SHARDS TESTS gpt sweep-A` | After GPT-OSS's derive job. Needs `SWEEP_B_NAME` and `SWEEP_B_SHARDS`. | Seals half A of the nudge sweep (every expert set at shifts 0.4/0.8/1.2, target and two random draws) and queues its analysis and the pick job. |
| `pick.sh EFFECTS MODEL NAME SHARDS TESTS` | Queued by the half-A launch, after its analysis. | Applies the frozen pick rule (`PICK.json`) and launches half B (`launch.sh ... sweep-B`) from this checkout. |

Job names start with `oc-` (captures) and `os-` (steering), never `whole-`, so scripts that wait on
live whole-reasoning jobs ignore them. CPU jobs run on `boost_usr_prod`: `lrd_all_serial` holds at
most 10 submitted and 2 running jobs per user. Logs are in `report/NAME/logs/`.

## First campaign (r1, ledger F-0004, D-0003)

| Script | When | What it writes |
| --- | --- | --- |
| `usage.sh [MODELS]` | First. A CPU job on the serial partition, about 5 minutes. | `campaign-runs/outcome-steering/<model>/USAGE.{npz,json}` on FAST: how often each expert is used per reasoning token in every cohort-B attempt. |
| `launch.sh NAME SHARDS LOCAL_TESTS` | Once the code is committed and the usage job finished. | `report/NAME/`: receipts, a frozen code snapshot, the whole-reasoning job graph and an analysis job that waits for this release's extractions and the final whole-reasoning combine. |

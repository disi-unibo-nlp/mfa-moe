# Routing-control job reconciliation, 2026-10-02

Canonical positive discovery pilot: Slurm job **59216285**, `st-eligible-micro-v3`,
manifest `CAUSAL_ELIGIBLE_MICRO_SERIAL_MANIFEST_v3.json` with seal
`320ecca116c7ab6397740876fb680ad666b3df9696adc2ec8a30e1fc9154f052`.
It has 13 distinct discovery families, 156 requests, six counterbalanced arms,
two seeds, 256-token cap and two A100 GPUs. At the last check it was pending for
priority with zero elapsed allocation. Its output directory is
`/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/eligible-micro-serial-v3-320ecca116c7ab63`.

Slurm job **59216368** (`st-eligible-micro-v4`) was independently submitted for
the *same rows, arms, action dictionary, random-set schedule, arm-order schedule,
seeds, cap and engine profile*. Its driver differed only by version/schema/UID
prefix. It was cancelled while pending at zero elapsed allocation to avoid
duplicate GPU spending; accounting reported `CANCELLED by 133943`,
`Elapsed=00:00:00`, no `AllocTRES`.

The negative/deactivation diagnostic in
`CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v4.json` is a **different**
workload and is bound to the canonical positive-v3 manifest. It was submitted
as Slurm job **59216601**, `st-deactivate-v4`, and verified pending for priority
at zero elapsed allocation, with two A100 GPUs and a three-hour limit.
Before further submissions, inspect `squeue`
and `sacct` for newly queued equivalent jobs, and preserve the positive-v3
binding in downstream blind frames and analyses.

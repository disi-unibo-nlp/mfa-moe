# R3-D anticipation resource amendment, 2026-10-01

The user's 2026-10-01 instruction authorizes the compute needed for the named
paper analyses. The historical eight-CPU-core-hour estimate for R3-D
anticipation is superseded for resources only. The frozen statistical driver,
specification, family folds, five repeats, five outer folds, 20 noise draws,
all responses and contained sensitivities remain unchanged.

The six model/response cells each have a complete and a contained fit. GPT has
46 model variants per fit; Qwen has 24 for `next` and 46 for `switch` and
`dest`. The driver fits these in chunks of at most eight variants, with 25
repeat/fold tasks per chunk: **1,650 checkpointed tasks** overall. Job 59094698
completed 18 tasks in 6,984 seconds on four CPU cores before its time limit.
Those tasks took about 1,286–1,465 seconds each when run four at once. The
remaining 1,632 tasks require roughly **635 CPU core-hours** at the observed
1,400 seconds per task, before variation, setup and retries. Using 2,500
seconds per task, plus a 20% allowance for later-cell variation and overhead,
prices the complete remaining stage at at most **1,360 CPU core-hours**. The
new stage ceiling is **1,500 CPU core-hours**; this is a conservative resource
proposal, not a change in the inference target. All prior partial checkpoints
remain bound to frozen specification SHA256
`62f1038003b18b09816635ada31d68136856477df335cbd54665942d159baa92`.

Submit one 24-hour, four-core, 64-GiB CPU slice at a time (maximum 96 CPU
core-hours per slice), inspect its exit state and new checkpoints, and resume
only if it made valid progress. Stop before another slice if observed throughput
would exceed the complete-stage ceiling; amend its price using the measured
runtime first. A timeout is partial progress, not a completed analysis.
Separate R3-E and B1 jobs remain unpriced here.

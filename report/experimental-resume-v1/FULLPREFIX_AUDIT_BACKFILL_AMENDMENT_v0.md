# Full-prefix audit backfill amendment v0

Applies only to pending Slurm job **59186342**, the immutable 372-window,
744-rating Qwen3.8 full-prefix audit. Frame, driver, rubric, two-reader
assignments, model, maximum 1,024 response tokens, output directory, and
six-hour maximum wall limit are unchanged. The job's output binds the existing
sealed price receipt
`TRANSITION_V22_FULL_PREFIX_START_RATING_PRICE_v2.json` (SHA
`fe02c3502a5d689a83713eca6efb37b76ffeeeab639949179090dcc0a6cf740e`).

Set Slurm `TimeMin=02:00:00` on this pending job. The installed Slurm option
and [official semantics](https://slurm.schedmd.com/sbatch.html) allow backfill
to lower the allocation time limit from six hours, but never below two hours,
only when that permits earlier execution. Once allocated, its time limit does
not change. This can improve scheduling but is not a start-time promise.

The driver saves each completed 16-window reader batch with UID and binding
checks. It prewrites an attempt receipt for every assigned rating. If the
job ends mid-batch, the existing recovery script must archive/dispose of
ambiguous attempts **after** Slurm final state is checked, before any
same-manifest resume. No two writers run concurrently. A shorter allocation
does not silently skip or duplicate assignments.

The original complete-stage stress price is 13.202 GPU-hours with two cold
loads and 1,024-token responses for all 744 ratings. Its ceiling is 16
GPU-hours. A 2–6-hour first allocation can be followed by nonoverlapping
two-hour same-manifest recoveries only while remaining under that ceiling.
At most four allocations (first two hours plus three recoveries) add two
cold loads and shutdowns to the original price: about 0.884 GPU-hour by
the receipt's cold-load/shutdown assumptions, yielding about 14.086
GPU-hours. Actual runtime, completion, partial failures and billing are
verified for each job. If the measured work would exceed the ceiling, stop
and reprice before another submission; no scientific sample or cap changes.

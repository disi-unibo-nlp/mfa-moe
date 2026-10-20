# Whole r3: batched execution correction

This continues the same Alignment v3 experiment, families, targets, controls,
doses, masks, seeds, audit selection and estimands. The original r3 release,
184-test allocated CPU qualification, both engineering canaries, every partial
and all receipts remain preserved in the adjacent original r3 directory.

GPT's completed decode intervals sustained 26.5785 aggregate tokens/s over
48 rows, or 0.5537 per sequence. The all-layer per-request tensor operations
and audit transfers made the main-study throughput assumption inapplicable.
No canary unit completed. GPT61578496/Qwen61578497 and monitor61578500 were
canceled after758seconds. The retained new GPU cost is0.842222hours, and the
cumulative CPU charge including the successful validation and conservative
local work is1.422222core-hours. Neither allowance is reset.

The correction batches the unchanged dual equation across request rows,
reduces additive telemetry in two GPU reductions per layer and transfers full
audit tensors once per layer. Project tensor functions use dynamic full-graph
compilation on GPUs; the original row-wise path remains the CPU reference.
Equivalence covers both expert geometries, inactive native bits, prompt scope,
recompute, request reordering, growing counter buffers and complete audit data.
Fifteen focused checks passed before sealing; new CPU qualification and actual
GPU canary evidence are required before production.

Canaries keep the full production allowances and relative gates from Alignment
v3. Production retains two GPT and seven Qwen shards, with64 in flight and the
same priority tiers. At admission, SHARD_PLAN.json bounds each model's actual
walltimes at or below the original165/240-minute maxima so actual terminal
usage plus every committed allocation maximum stays within72GPU-hours. Any
half-dose canary is charged before admitting its remaining production. The
quarter projection uses these actual planned maxima. This preserves the hard
ceiling if the monitor exits.

Completed jobs can disappear from Slurm's controller before dependent jobs
are submitted. Only a native proof satisfying the exact afterok/afterany
condition permits pruning such an expired scheduler ID; every reconciliation
is recorded. Live dependencies remain explicit, and a failed predecessor never
satisfies afterok. Extraction and analysis are queued with production, as in v3.

The main-study paired planning SD remains unavailable: its manifests contain
no native_repeat arm. This is disclosed as SUSPECTED and never gates science.
No manuscript changes or remote pushes are part of this continuation.

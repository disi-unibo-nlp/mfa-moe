# 16k utility scout: prepared, not launched

`UTILITY_SCOUT_PLAN_v1.json` seals 96 family-disjoint utility representatives in
frozen order, from their original Qwen3.6 prompts. Two seeds and two assigned
arms (native, future frozen policy) make 384 intention-to-treat cells. No
selected prefix, future sentence, gold answer, correctness result, or policy
selection enters generation enrollment. The dense utility units identify the
frozen families; they are **not** generation prefixes.

The maximum charged output is 16,384 tokens **per cell, counting both emitted
and any injected tokens**. Generation receipts must preserve exact token order,
source, stop reason, original prompt hash, family assignment, policy binding,
and errors. Missing receipts remain in the ITT audit; capped and failed cells
are operationally incorrect. Natural stops enter a Qwen3.6-decoded blind bundle,
strict `math_verify`, and arm-blind J1 adjudication of finished strict-rejected
answers. Utility effect reporting waits for complete final-answer judgments and
family-clustered accuracy/token intervals.

The plan's worst-case output is 6,291,456 tokens. At the observed serial stress
rate of 8 tokens/s for one two-A100 worker, this alone is about 218.5 node-hours
or 437 GPU-hours, before loads, prefill, controller, judging, retries, and
shutdown. The old 4.75 GPU-hour scout estimate cannot price this design. A
16-shard split by frozen family order keeps each family's four assignments
together, with at most 24 requests and 393,216 output tokens per shard; at that
same stress rate the decode maximum is 13.7 hours per shard. Simultaneous shards
reduce elapsed time but do not reduce total GPU-hours. Availability and walltime
must be checked against current Slurm limits before submission.

The next resource gate is an outcome-blind price pilot after the policy and
serial/eager engine are qualified. Freeze the pilot families in advance and
retain them in the final ITT analysis if the exact policy and sampler remain
unchanged. Record full 16k native and policy behavior, cold load, prefill,
recompute, serial decode throughput, request overhead, controller cost, failed
attempts, and shutdown. Price the complete 384-cell stage with
`utility_scout_v1.price_complete_stage`: all 6.29M maximum output tokens, exact
prompt prefill, measured costs, all model loads, grading, NLL, labeling,
contingency, and the slowest shard's walltime must fit. The price remains HOLD
without a sealed measured profile. A batched or tensor-parallel variant requires
its own request-isolation, replay, and recovery qualification and a new price;
the current serial evidence cannot be transferred to it.

`run_utility_scout_v1.py` is deliberately fail-closed. It checks the future
policy/engine/price bindings but cannot execute a policy arm until the
versioned adaptive controller implementation and GPU qualification exist. It
must not be used as a native-only proxy. No Slurm utility job has been
submitted by this scaffold.

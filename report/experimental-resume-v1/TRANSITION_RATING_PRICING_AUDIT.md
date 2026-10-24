# Discovery transition LLM audit pricing, 2026-10-01

The frame has 619 arm-blind rows from 48 discovery families. Two independent
same-model Qwen3.8-27B draws rate each row against the frozen content rubric.
They remain LLM audits, not human truth. The original 512-token draft is
inactive: 24 of 200 qualified parity outputs exceeded 512 tokens, so that cap
would add avoidable truncation. The active reader v2 uses the parity-qualified
1,024-token maximum and remains checkpointed by sealed 32-row batches.

CPU price job `59120940` failed after 47 seconds because the first existing
client environment lacked Jinja2. Recovery `59121002` completed, but its
price is **invalid**: it counted the two keys of the tokenizer's returned
container as two prompt tokens. The 1,024-token v2 CPU price job `59121048`
repeated that counting error. Both price files are retained as rejected
calculation history and cannot authorize a GPU submission.

Versioned price driver v3 explicitly extracts and validates `input_ids`, with
unit tests for mapping and sequence returns. CPU job `59121106` is submitted
to compute exact twice-counted prompt tokens and the full capped decode,
two loads, teardown and retry allowance. The proposed complete-stage
reservation is one five-hour two-GPU slice plus one 90-minute two-GPU recovery:
at most 13 GPU-hours. The current user instruction authorizes needed paper
workloads, but **GPU launch is held** until v3 finishes `0:0`, its artifact is
sealed and its complete projected cost fits this capacity. Raw CPU failures
and all eventual GPU allocations remain in the ledger.

**Observed correction:** Job `59121106` completed `0:0` and sealed
`TRANSITION_RATING_COMPLETE_PRICE_v3.json` with digest
`69f2ee74f9fdeab0751020ea0516b7ebe8072947dcc01a814f0e1b2bff767bb1`.
It counted 425,712 actual prompt tokens across both readers (238–536 per
prompt) and the full 1,267,712-token decode cap. With parity throughput
derated to 65%, 25% repeated-work allowance, two cold loads and two shutdowns,
the complete-stage projection is 11.322 GPU-hours. The 13-GPU-hour ceiling
fits that price: a five-hour two-GPU first slice plus one 90-minute two-GPU
same-manifest recovery. The price binds reader v2 code SHA-256
`74c16a5da5e27ef0d3178f9d3ede03d42b58199a62a628f5ebf5c79c45d101d9`.
The two earlier files remain invalid despite their `PASS_COMPLETE_STAGE` text.

The active reader batch script passed `bash -n` and live `sbatch --test-only`;
its source SHA-256 matched the sealed price. Under the amended ceiling, Slurm
submission `59121216` requested one node, two A100 GPUs, 16 CPUs, 120 GB,
and five hours. `scontrol` verified account `iscrc_miosr`, partition
`boost_usr_prod`, QoS `normal`, pending priority, and the exact script path.
Output is bound to frame/driver digests at
`steering-v1/runs/routing-control-v1/dense-discovery/ratings-f5b2e28c-74c16a5d/`.
This records submission only; completion, readout quality, and any recovery
remain to be verified from final job state and sealed batches.

# Routing-control resource amendment v0.2, 2026-10-01

This amends resources only in `PROTOCOL_v0.1.md`. The user's latest instruction
authorizes the GPU time needed for the named paper runs. The fixed family
pools, three transition hypotheses, action limits, arms, horizons, ordering,
analysis and stopping rules in v0.1 remain in force. A larger allocation
cannot turn an unsupported detector, insufficient eligible families, or an
incomplete cell into evidence.

The ordered-worker engineering qualification has completed at 0.83333
GPU-hours across four attempts, including the successful job 59108983. The
direct seven-class LLM measurement audit consumed 0.64056 GPU-hours in job
59112590. Its prospective operational gate passed, with the measurement
limits in `DENSE_JUDGE_QUALIFICATION_v1.md`.

The 5,659-sentence discovery-only class-labeling stage is added as a separate
measurement line. `DENSE_DISCOVERY_LABEL_PRICE.json` gives a measured all-in
price of 14.019 GPU-hours and a revised ceiling of 14.25 GPU-hours, including
one checkpointed continuation. Its first submission, job 59114230, is bounded
to 6 hours on two A100 GPUs (12 GPU-hours allocation maximum). The label
stage is not action discovery and does not increase its maximum of six
candidate actions. The 10.5-hour table in v0.1 remains the historical
proposal for the causal stages; each later stage requires a new complete
qualified price and versioned ceiling under the expanded authorization.

All actual spending is reported by job ID in `JOB_AUDIT.json`. No result is
declared from a queued or running job. The direct judge's class consistency
is an LLM audit; arm-blind substantive ratings and online prefix qualification
remain separate gates.

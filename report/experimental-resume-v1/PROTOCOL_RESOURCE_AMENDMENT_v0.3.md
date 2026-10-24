# Routing-control resource amendment v0.3, 2026-10-01

This adds a separate discovery semantic-measurement line to v0.2 under the
user's explicit instruction to use the GPU hours needed for the paper. It
changes no family pools, transition hypotheses, action count, validation
arms, horizons, outcome definitions or stopping rules in `PROTOCOL_v0.1.md`.
The 14.25-GPU-hour dense class-labeling line remains separate and pending.

The label-independent discovery candidate frame `TRANSITION_AUDIT_CANDIDATES.json`
is sealed over all 48 frozen families, with 619 arm-blind rating rows. It
contains all 19 detector fires and 600 deterministic nonfires. The reader
v2 has two independent same-model Qwen3.8 draws per row and the parity-
qualified 1,024-token cap. The two readers see only the original problem and
local emitted sentences plus the fixed behavioral criterion; they do not see
the arm, routing, gold answer, correctness or class labels. These are LLM
audits, not human validation.

The valid complete-stage price is
`TRANSITION_RATING_COMPLETE_PRICE_v3.json`, SHA-256 seal
`69f2ee74f9fdeab0751020ea0516b7ebe8072947dcc01a814f0e1b2bff767bb1`.
It counted 425,712 prompt tokens and a 1,267,712-token maximum decode;
derated measured judge throughput, 25% repeated work, two cold loads and
teardown project **11.322 GPU-hours**. The revised allocation ceiling is
**13 GPU-hours**, as a five-hour two-A100 first slice and at most one
90-minute two-A100 same-manifest recovery. All actual loads, generation,
failed attempts and retries are debited. A slice that cannot finish within
the remaining ceiling does not launch.

The 512-token draft and the first two prompt-token prices are preserved as
invalid planning artifacts in `TRANSITION_RATING_PRICING_AUDIT.md`; they cannot
qualify this run. Blind ratings do not by themselves qualify a transition.
Detector confusion, reader agreement, semantic family support, dense-label
coverage and the worker action checks remain gates before causal discovery.

# Experimental resume v2 — saved-result closeout, 2026-10-02

This addendum supersedes the **live job status** in `RUNBOOK_v1.md`; it does not
rewrite that runbook's historical decisions, failures, resource accounting, or
protocol. `PAPER_SNAPSHOT_v2.json` is the completed R3-D/B1 saved-result
evidence addendum (seal
`fb711ca8c6dfc5b462f53e0969306cff3dbcd31f8efc5124b56deb32eff50ac1`).
`PAPER_SNAPSHOT_v2_1.json` adds the later source-checked reader-cost caveat
(seal `ad4e1df74930e875e552046fbbcf06cf45bfb64699f682b06c13a237cd4697e5`).
It retains the immutable first figure snapshot and appends 32 claims to the
original 209-row ledger in `CLAIM_LEDGER_v2.json`. Earlier claim rows remain
byte-for-byte identical as JSON values. Figures from the first snapshot still
predate the completed R3-D and B1 analyses; the v2 addendum supplies tables and
claim provenance, not updated figures.

## Completed clean R3-D and B1

The seven-model R3-D readout is complete in the sealed
`forum/tests/r3_context/readout-results.v3.json` (result seal
`fb871680020bfb4e17d72f5059af83a58458c73aa6ca5b7cd31532940d49ea5e`).
All seven judge-context cells have negative routing-added loss and negative
routing excess over the lexical null under their saved seven-model intervals.
The judge-context baseline includes future sentence text; this is retrospective
measurement, not an online prefix detector. Every row's exact population and
interval is in `PAPER_SNAPSHOT_v2.json` and `CLAIM_LEDGER_v2.json`.

R3-D anticipation completed its frozen 1,650-fit grid. The sealed final result
is `forum/tests/r3_context/anticipation-results.json` (seal
`08f15228ae221bf69a4f7dd976909df5531db0e00ff8d6acdb97e8423140372e`).
The registered six GPT primary contrasts use composition16 for the A1 next and
contained-next cells, and k16 for the four K2 switch/destination cells. Holm
adjustment across those six saved bootstrap p values gives **.00240, .00240,
.00720, .56749, .00240, .44071** in that order. Bonferroni six-contrast
percentile intervals are recorded separately in the result and v2 snapshot.

The strongest A1 next-class result has candidate-minus-baseline loss
**−0.02073 nats/pair**, simultaneous interval **[−0.02548, −0.01601]**, and
5/5 positive-gain repeats on 509 GPT questions and 10,209 adjacent pairs.
The contained-window result is **−0.01570**
**[−0.02575, −0.00647]**; Qwen3.6 is same-direction at **−0.05570**
**[−0.06944, −0.04154]**. This satisfies the registered *predictive*
advancement gate for composition16 on the evaluated populations. GPT
composition_antic next is **−0.00437**, below the practical −.005 gate, and its
Qwen replication remains explicitly incomplete.

GPT K2 switch k16 is **−0.00264** and its contained estimate is **+0.00067**;
the switch pipeline misses the practical and contained-sensitivity gate despite
its adjusted interval below zero on the main window. GPT destination k16 is
**−0.01103** **[−0.01840, −0.00405]**, but contained GPT and Qwen3.6
destination intervals span zero. Its direction is encouraging yet too imprecise
for a strong replicated effect claim. All R3-D estimates are observational
prediction effects beyond specified text/history controls. They do not identify
causal expert actions, an optimal trajectory, or an accuracy–token benefit.

The separate exact-ID-clean GPT B1 refit is sealed COMPLETE in
`steering-v1/runs/resume-v1/r3e-b1-clean-v2/result.json` (seal
`192d74aba5a06f3b623b044f4dc1df013b74f85385ad2e81a5c7554128b3fa92`).
On 509 questions and 496 duplicate families, the family-clustered joint gain
is **+0.008385 nats/token-pair**, nominal 95% interval
**[+0.008044, +0.008737]**, with 5/5 positive repeats. The frequent-ID filter
drops 364,978 of 527,099 labelled tokens (69.2%); report the selected token
population. This is class-associated routing prediction, not correctness or
causal steering.

## Recovery and job retirement

Canonical R3-D job `59111360` stopped after writing 644 GPT fit checkpoints.
Verified Qwen sidecar job `59173234` contributed 750 fit checkpoints; GPT
destination sidecar job `59180550` contributed 300, with 44 valid overlaps
retained from the canonical writer. Sequential merge jobs `59206783` and
`59206978` both completed `0:0`. Independent audit v1 job `59207468` failed
on an overly strict rule for four existing `predictions.npz` files; its logs
are preserved. Corrected independent audit v2 job `59207811` completed `0:0`,
sealing all 1,650 fit bindings, finite arrays, repeated-fold grids and source
hashes in `R3D_MERGED_INVENTORY_AUDIT_v2.json`. Frozen finalization job
`59208033` completed `0:0`; `R3D_FINALIZATION_AFTER_MERGE_v1.json` binds the
result and original code/specification hashes.

The six obsolete, zero-runtime continuation jobs `59115320`, `59115408`,
`59115420`, `59115433`, `59115442`, and `59115443` were cancelled only after
the sealed result and audit existed. Each is `CANCELLED by 133943`, elapsed
`00:00:00`; `R3D_GUARD_CHAIN_RETIREMENT_v1.json` seals their exact states.
The held invalid deactivation job `59205653` was likewise cancelled with zero
runtime. The separate positive micro screen `59204242` remains pending under
`JobHeldUser` during independent review. The generator for this screen is
**Qwen3.6**; **Qwen3.8** is the independent semantic reader, not the generator.

The preliminary eligible deactivation blind-rating price v3 assumed **256
output tokens per Qwen3.8 rating**. A metadata-only audit of the ten prior
micro-screen reader batches found **44 of 97 valid stopped ratings exceeded
256 tokens** and **3 of 104 assigned ratings exhausted the 1,024-token cap**.
`CAUSAL_READER_OUTPUT_LENGTH_AUDIT_v1.json` seals batch file hashes and these
counts without copying any completion text. The new eligible reader driver
`rate_eligible_immediate_semantics_v1.py` uses 1,024 tokens. The 256-token v3
envelope cannot price that driver; an exact post-generation blind frame and a
new complete-stage price are required before any eligible rating job. None has
been launched. These lengths are a measurement/resource caveat, not evidence
of a semantic intervention effect.

## Current interpretation and remaining work

Legacy X2 is complete: 6,630 capped requests and 3,354 matched native-NLL
measurements are audited in `X2_COMPLETION_AUDIT_v1.json`; no X2 rerun is
needed. The registered base X3 stage has an all-in **53.07 GPU-hour projection**
under its amended 65-hour total ceiling in `X3_COMPLETE_STAGE_PRICE_v3.json`,
but remains held behind the new causal-action study and independent review.
Its optional E4 and dev-spare branches require separate qualification/pricing.

No causal local-trajectory, accuracy-change, or token-change estimate is supplied
by this R3-D/B1 addendum. Preserve the original confirm-exposure disclosure,
K1's narrow matched-population bound, K2's inconclusive historical status,
G2's non-rejection reading, and G3's dose-selection-only role. The next paper
snapshot must regenerate figures from sealed new results and keep absent causal
cells explicit. Do not convert predictive routing association into a steering
claim or reuse historical confirm cases as untouched validation.

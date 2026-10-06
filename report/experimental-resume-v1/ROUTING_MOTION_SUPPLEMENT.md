# Native routing motion, discovery population

The saved Qwen3.6 native traces from the 48 frozen, family-disjoint discovery
families contain 12,274 complete, nonoverlapping 64-reasoning-token windows.
Every window profile was recomputed from the saved top-eight expert identities
and selected-gate weights. The extraction omitted 1,299 partial-tail tokens
across families and verified the native trace, tensor and sentence-profile
digests before writing a sealed family array. No class label, correctness,
future text, or intervention result entered this computation.

The equal-family, layer-averaged total-variation change between adjacent
gate-distribution windows was **0.416** (simultaneous 95% interval
**[0.404, 0.427]**). The L1/2 magnitude of the second gate-distribution
difference was **0.716 [0.696, 0.735]**. Turnover in the eight most frequently
selected experts between adjacent windows was **0.544 [0.526, 0.563]**.
Intervals use 5,000 family bootstrap draws and Bonferroni coverage for these
three descriptive scalars. They quantify heterogeneity conditional on this
deterministic discovery enrollment; they are not inference to a random
population of problems.

The four-panel [PNG](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery/fixed-window-routes-84a85c92-363dbe1b/ROUTING_MOTION.png)
and [PDF](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery/fixed-window-routes-84a85c92-363dbe1b/ROUTING_MOTION.pdf)
show the layer profiles and native expert exposure. Their sources and checksums
are sealed in `ROUTING_MOTION_FIGURE.json`; the estimates and uncertainty are
in `MOTION_UNCERTAINTY.json`. These are observational descriptions. Neither
the velocity nor the acceleration profile identifies a useful causal routing
action, a semantic transition, or an optimal reasoning sequence.

CPU extraction job `59121495`, figure job `59121502`, and interval job
`59121573` all completed `0:0`. The first extraction attempt `59121394`
failed before processing a family because the launcher environment lacked
PyTorch; its 29-second allocation remains in the compute ledger. The
source-verified entries in `MOTION_CLAIM_AMENDMENT_v0.1.json` were merged into
`CLAIM_LEDGER.json`.

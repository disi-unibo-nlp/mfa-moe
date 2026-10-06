# Live checkpoint — 2026-10-02 09:23 CEST

Verified on `login05.leonardo.local` as `lmolfett`. The refreshed Slurm and
artifact audit is `JOB_AUDIT.json` (observed 2026-10-02 07:23:10 UTC).

## Current work

- R3-D clean seven-model readout job `59110943` completed `0:0`; its bound
  `readout-results.v3.json` contains all seven `CLEAN` token-identity and
  judge-context cells. These are predictive, observational fits, not an
  intervention result.
- R3-D anticipation job `59111360` remains running on four CPUs with a
  24-hour limit. Six progress-guarded CPU continuation jobs are pending on its
  dependency. Its final result has not been verified.
- No GPU study job is currently running or pending. The dense class and
  transition-rating jobs that were queued yesterday have completed.

## New steering measurements

Dense labeling `59114230` completed `0:0`: 5,659 contiguous native sentences
from 48 discovery families, with 5,651 parsed seven-class labels and eight
unparsed. Its class summary `59114313`, audit fixture `59117940`, and figure
`59121759` also completed `0:0`. The class dynamics are descriptive direct-LLM
audits. They do not measure substantive verification or causal control.

The blinded same-model transition rating job `59121216` completed `0:0` with
1,238 ratings of 619 windows (19 detector fires and 600 sampled nonfires).
The bound audit job `59121293` completed `0:0`; its source is
`TRANSITION_DETECTOR_DISCOVERY_AUDIT.json`. Among 5,523 contiguous pairs per
hypothesis, the current prefix detector yielded:

| Hypothesis | Fires | Start accepted by both LLM readers | Immediate next-sentence target also accepted by both | Valid-start nonfires in 200 sampled windows |
|---|---:|---:|---:|---:|
| Candidate to substantive verification | 15 | 9 | 0 | 53 |
| Approach to committed plan/implementation | 3 | 2 | 1 | 60 |
| Failed check to revision | 1 | 0 | 0 | 2 |

The sampled nonfires show that the detector misses many reader-rated starts.
These ratings are an LLM audit, not human semantic ground truth. The present
detector is **not qualified** to trigger causal action discovery: threshold,
coverage, timing, and family eligibility have not been sealed. In particular,
there is no observed immediate verification transition among its 15 candidate
fires; the single observed approach transition is too sparse to support the
planned template by itself. This is a feasibility finding, not evidence that
routing cannot steer reasoning. Keep the three fixed hypotheses and do not add
replacement targets because these results are sparse.

Next, inspect trigger timing and prefix-only start detection on discovery
families, then freeze a versioned detector and its eligibility gate before any
intervention cells. Do not submit causal-action, mechanism, or utility cells
using the current unqualified detector. The separate M9 closure probe remains
held for engine isolation as recorded in `M9_REPLAY_ADJUDICATION_v1.md`.

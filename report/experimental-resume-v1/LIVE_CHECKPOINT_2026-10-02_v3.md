# Live checkpoint — 2026-10-02 14:24 CEST

Verified `login01.leonardo.local`, user `lmolfett`. The direct registry and
`JOB_AUDIT.json` were refreshed at 12:17 UTC. They include completed, failed,
cancelled, running and dependent jobs; a Slurm exit alone is not accepted as a
scientific result. This is an operational checkpoint, not a change to the
sealed scientific protocol or an assertion of steering success.

## Steering status

The frozen discovery action dictionary proposes candidate-to-verification
layer 28 experts 9/189 and approach-to-commitment layer 24 expert 159, each
at biases +0.5 and +1.0. Native matched contrasts and four discovery folds
motivate these proposals, but their causal validity is untested. The
serial/eager pulse qualification `59200002` and audit `59200254` finished.
Across assigned 256-token continuations, target expert selection increased
from about 0.160 native to 0.238 at +0.5 and 0.333 at +1.0. Native duplicate
token streams matched only once in eight comparisons, so those differences
cannot be interpreted as semantic steering. The first four pilot starts had
no jointly eligible semantic target.

The full-prefix Qwen/native audit gives eight candidate, six approach and
zero failed-check jointly accepted starts; candidate and approach share one
family, leaving 13 distinct families. Adding the independent GPT-OSS vote
narrows the intersection to seven distinct families, with only two candidate
families. GPT-OSS also accepts both frozen Qwen-agreed negative controls;
its vote is therefore diagnostic, not a human truth label or a calibrated
veto. These are selected discovery starts, not population precision.

The 80-request batched sentinel generation `59203153` completed `0:0` on two
A100s in 12m56s, with all 80 request records. Its dependent CPU audit
`59203204` failed before inspecting data because its hostname guard also
rejected a Slurm CPU node named like a login node; a corrected audit is being
prepared. Batched execution remains unqualified until that audit passes.
A full sealed 43-prefix/86-rating
Qwen within-sentence timing audit `59203262` is also RUNNING on two A100s.
It tests earlier complete-expression cuts; it does not change the registered
triggering-sentence-excluded primary outcome. The duplicate `59203268` was
cancelled after 1m47 before ratings; its cost and empty result are retained.
The exact-ID joint-reader pool CPU job `59203778` failed before data output
because a hostname guard incorrectly treated a Slurm CPU node as a login
node. The corrected retry `59203918` completed `0:0` in 32 seconds and sealed
14 joint-reader starts, 13 globally family-disjoint selected UIDs, native
token IDs and replay checks in `JOINT_QWEN_NATIVE_EXACT_POOL_v1.json`.

A separate 13-family feasibility micro-screen is being priced and frozen:
13 families, two seeds, native and native duplicate, frozen target +0.5/+1.0,
and matched random +0.5/+1.0 (156 capped 256-token requests). The four random
sets will be balanced across families and seeds. It will remain explicitly
below the 48-family discovery target; success would justify, not substitute
for, family-disjoint validation. It must pass its own complete price and
execution gates before Slurm submission.

## Other analyses

Canonical R3-D anticipation `59111360` and its GPT destination shard
`59180550` remain RUNNING. Guarded verifier and merges remain dependent;
the continuation guard remains held until verified receipts and stopped
writers. The conditional ETA for the GPT shard is about 18:15 CEST at its
observed fit rate, and canonical R3-D's wall deadline is 17:38 CEST.

B4's registered clean nested-CV result remains -0.000729 nats per scored
question, family interval [-0.005313, +0.004384], five-comparison simultaneous
interval [-0.006747, +0.005289], over 508 scored questions in 495 families.
It does not support predictive improvement or equivalence. The corrected
precision simulation v3 (`59203686`, COMPLETE 0:0) calibrated oracle KL to
the held-out scored covariates without using held-out original outcomes.
Across target KL 0, .001, .003, .005 and .01 (200 replicates each), registered
advance rates were 0%, 0%, 1.5%, 4.5% and 5.5%; nominal 95% interval coverage
for the simulated fitted-algorithm target was 91.5%, 88.5%, 86.5%, 82% and
89.5%. Undercoverage means those intervals are described as nominal, not
calibrated. V2's train-only calibration overshot held-out target KL and is
retained as a separate sensitivity. None of this is a routing intervention.

X3 timing-only recovery measured 971,219 decode tokens across 40 requests
at 838.61 aggregate tokens/s on two A100s. Its 770-request ceiling would
require up to 14,049,070 decoded tokens: measured-rate decode alone projects
about 9.31 GPU-hours, before load, prefill, labeling, native NLL, grading or
contingency. A complete versioned X3 price is in preparation; X3 content has
not been inspected or launched as a full study. Steering feasibility has
priority over the X3 launch.

No semantic-control, accuracy or token-saving outcome is established yet.
The next causal gate is the audited batch-isolation result and an eligible
family-disjoint feasibility manifest. A larger independent evaluation must
remain separately frozen and priced if feasibility succeeds.

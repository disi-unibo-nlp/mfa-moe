# Counterfactual routing-loss discovery amendment v0.1

This is an offline, discovery-only measurement and engineering study. It does
not change the original X2/X3 protocol or the frozen local semantic-control
endpoint. It addresses whether expert sets selected by native behavioral
contrasts have functional effects at a decision boundary.

The first qualification uses the same four frozen discovery prefixes as the
48-request routing pilot, in deterministic UID order. At each prefix, score
the first four native tokens after the boundary. At position j, replay the
original problem, native prefix and only native tokens before j. Intervene on
the one prediction row for token j. Never carry an edited KV state into the
next position. The realized token is an offline scoring target and is never
used in routing metadata, trigger selection or an online controller decision.

Eight arms are frozen before submission: two native repeats; +0.5 and +1
biases on layer 28 experts 9 and 189; positive force and negative force on
that pair; and positive and negative force on one of the four existing
exposure-matched random pairs. The four random pairs are assigned once each
in fixed prefix order. Every arm retains eight routed experts and the shared
expert. Positive force inserts the pair through the existing sampler;
negative force excludes it and lets native alternatives fill the same slots.
This is a separately qualified force screen, not a claim that the ordered
two-pulse worker already qualifies negative actions.

Record raw designated-token log probability, sampled token, executed expert
identities, target occupancy, gate displacement, active rows, inactive native
identity/weight checks, prefix digest and both worker-rank receipts. Use the
installed vLLM designated-token logprob API instead of full-vocabulary output.
Measure each native position again by native teacher forcing to validate
token/logprob alignment. Add four reasoning-closed requests and one public
reset-running-prefix-cache event during prefill to check closure and recovery.
All 148 assignments, failures and caps stay in the qualification receipt.

The qualification requires correct designated-token extraction, fixed k=8,
force membership/exclusion on open prediction rows, no active row after
closure, zero inactive-row routing identity or weight mismatches, and both
rank receipts. Native designated versus teacher-forced logprob alignment is
reported using fixed absolute tolerances (mean <= 1e-4, maximum <= 1e-3)
alongside native-repeat variation. Those tolerances concern this measurement
API only and cannot establish engine equivalence. The reset event must
succeed and yield recorded recomputation before a recovery claim is made.

Four engineering families cannot establish semantic expert usefulness. Once
the full-prefix audit supplies qualified starts and substantive target
continuations, freeze a separate matched target/non-target fixture manifest
using discovery families only. Compare target-specific loss changes with
native repeat noise, random experts and syntax-token controls; report every
tested candidate and family-level effect. Select at most one new sparse set
from a declared candidate screen. An independent same-prefix free-generation
test is required before adding it to the action dictionary. Independent
mechanism families and original-prompt utility remain separate evaluations.

The approach adapts equal-compute token-level interventions from
[When Are Experts Misrouted?](https://arxiv.org/html/2605.07260v1).
Its verified-path, local-loss proxy does not demonstrate free-generation
semantic steering. The need to test functional value rather than utilization
alone is supported by the
[causal expert-importance audit](https://arxiv.org/html/2606.10703v1).
For this study, the native continuation may be unsuccessful; its probability
is a mechanical measurement, with correctness and target status kept separate.

Price the complete qualification from the measured 48-request pilot,
including cold setup, prefill, scoring, reset, flush and shutdown; freeze the
driver/helper/worker digests and bind the output directory before Slurm
submission. The user's 2026-10-02 expanded authorization covers the necessary
resource envelope. Reprice subsequent semantic target screens from measured
qualification runtime.

# Approach gate and prompt-start control proposal v1

This is a new discovery proposal. It does not change the frozen local
trajectory endpoint or treat pilot starts as valid after an intervention.
The supporting sealed measurements are
`APPROACH_ONLINE_GATE_DISCOVERY_v1.json` and
`JOINT_QWEN_NATIVE_EXACT_POOL_v1.json`.

## A. Query-every-approach-fire online gate

At a complete emitted sentence, the streaming v2 detector proposes
`approach_to_commit`. For every such proposal within the 16,384-token utility
horizon, ask two unsteered native-model start-only questions on the actual
current prefix. The online input allowlist is the original problem, emitted
prefix and triggering sentence. Act only if both responses stop naturally,
parse strictly and say `start=true`; otherwise abstain. A missing answer,
timeout, incomplete sentence, unsupported transition, reasoning closure or
non-append-only replay also abstains. The resulting action is a frozen one
from the causally qualified discovery dictionary. The triggering sentence is
excluded from the local transition outcome.

The discovery sample has 35 lexical approach fires: Qwen3.8 two-reader start
acceptance is 12/35 (family-cluster bootstrap 95% interval 0.20–0.50),
rejection 21/35 (0.44–0.75), and split 2/35. Native Qwen3.6 two-reader
acceptance is 9/35; Qwen accepts six and rejects three of those nine.
These are LLM rubric comparisons in stratified discovery windows, not human
truth or a population precision estimate. A separate 48-window nonfire
sample has one Qwen accepted start; fire/nonfire stratification precludes a
live recall estimate. No learned cheap candidate score is used: its
family-held-out 70% precision gate selected zero of 103 resolved fires.

The all-fire cached-native inventory has 42 approach proposals, 39 within the
16k utility horizon, and three with native prefixes 8,193–16,384 tokens.
Two side draws at every in-horizon fire project 164,974 native prompt tokens
and at most 4,992 response tokens before runtime overhead. Prospective
utility must record **actual** prefill, generated tokens, load, recomputation,
latency, nonfires, vetoes and failures. The 8k–16k side-query path must pass
context, batch-isolation and recovery qualification first. The 39 cached
opportunities are a cost scale, not the number that will occur when routing
changes the continuation.

Freeze the detector/rubric, seeds, maximum query count, side-query behavior,
action and family-disjoint original-question enrollment before validation.
Calibrate detector fire counts, side-query acceptance, Qwen-reference audit
and actual token cost on discovery families. Validate semantic completion,
accuracy and total token change by assigned arm with family-clustered
simultaneous intervals; include nonfires and all failed side calls.

## B. Separate prompt-start and early-token control study

All five globally family-disjoint Qwen/native agreed approach starts occur by
emitted token 269 (178, 198, 208, 257, 269). This supports an early-window
*timing hypothesis*: a routing action beginning at the original prompt may
alter entry into exploration and subsequent commitment before token 512.
It does not establish an optimal sequence or a causal effect.

Freeze the action using discovery-only matched contrasts and the 13-family
causal micro-screen. Apply one 256-token pulse at emitted position 0; test a
two-action template with its second 256-token pulse at position 256. This is
a separate protocol from the original branch-relative 0/512 template.
Enroll original questions before any generation. Compare native routing,
the frozen action, several exposure-matched random expert sets, and the same
two-action multiset in reverse order where two distinct actions are qualified.
Measure actual executed experts and dose, with inactive-routing and
neighbor-isolation checks.

The prompt-start primary is a separately specified class sequence in the
first 1,024 emitted tokens, rated blind on dense contiguous windows. It has
no triggering sentence to exclude. Report local class transitions, accuracy
and total output tokens at the separately declared 16,384-token utility cap.
This arm uses no online semantic side query, so its controller token cost is
smaller. A positive result establishes the effect of this fixed early policy
on the enrolled population; it does not retroactively validate semantic
start detection or the original local transition estimand.

Before launch, qualify exact pulse timing, resume, batch order, preemption
and closure in the selected routing backend. Price a complete 1,024-token
mechanism stage and complete 16k utility stage from exact prompt tokens and
measured runtime, including loads/retries. Freeze disjoint discovery and
validation families, arm assignment and expected request count. If the
13-family local screen is weak, retain the preselected prompt-start policy as
a versioned exploratory arm and power independent confirmation from its
observed variance rather than adding families to rescue a borderline result.

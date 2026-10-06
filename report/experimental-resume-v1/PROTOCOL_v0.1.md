# Routing actions and local class trajectories, protocol v0.1

Design fixed from the user's 2026-10-01 resume plan. Execution is conditional on
qualification, complete stage pricing, and workload authorization. No intervention
outcomes exist for this study. This protocol is separate from legacy steering-v1
v0.3, including its X2/X3 diagnostic endpoints and multiplicity rules.

The objective is expert-routing control of reasoning trajectories, with measured
accuracy and token-cost effects. There is no minimum required accuracy gain or
token saving. A useful local policy is an empirical finding; observational native
trajectories cannot establish a universal optimum.

## Populations and prospective separation

Canonical identity is `dataset|source_problem_id`, independent of model, capture,
attempt, and alias. `family-freeze.json` records all 117 edges of the historical
8-token-shingle containment method (threshold .85, buckets at most 20, at least six
uncommon shingles), exact duplicates, and their connected components across all
1,547 questions. The historical top-ten-pair summary alone is insufficient.

The original split is preserved. Clean legacy primaries exclude exact confirm IDs
across all models; the sensitivity excludes dev/tune families connected transitively
to confirm. For this new study, parent pools exclude every confirm-connected family.
There are 492 eligible parent families. Ascending
SHA256(`routing-control-v1|family_id`) freezes three disjoint pools: 48 discovery,
128 mechanism validation, and 96 utility. The representative within each family is
the first question in ascending SHA256(`forum-v1|question_key`) order. The full
enrollment order is fixed before behavioral selection. No replacements follow
ineligibility. These counts establish only parent availability; detector coverage
and eligible transition counts remain unmeasured. Insufficient eligibility fails
feasibility before submission.

Historical confirm exposure is disclosed. New family separation and clean refits
cannot restore the historical confirm set's blindness. Previously visible native
outcomes are planning evidence, not prospective confirmation.

## Fixed hypotheses and measurements

Only three transition hypotheses are studied:

1. An unchecked candidate to substantive verification.
2. Exploration with an identified approach to committed planning or implementation.
3. A visibly failed check to revised exploration or planning.

All seven classes are measured: Read, Analyze, Plan, Implement, Explore, Verify,
Monitor. Unsupported hypotheses are omitted without searching for replacements.
Discovery alone supplies detectors, matched native contrasts, expert proposals,
thresholds, and trajectory templates.

A substantive check evaluates an original constraint or computation. “Let me
verify” alone is insufficient. An identified approach names a method and the step
it would perform. A failed check contains an observable contradictory constraint
or computation, not an unsupported statement of uncertainty. Revised exploration
or planning must respond to that failure. The behavioral rubric and complete
sentence windows are frozen before validation. Independent readers receive UID-only
bundles without arm, routing, other readers' ratings, or labeler predictions.
LLM ratings are explicitly reported as LLM audits. Claims about latent reasoning
require additional measurement validation.

New windows use dense, contiguous sentence labels. Segment changes, missing labels,
and sparse index gaps are never joined. Offsets must agree with emitted tokens and
routed arrays. Termination is retained as an absorbing outcome. The triggering
sentence is excluded from every semantic endpoint.

## Prefix-only detector

The online input allowlist is the original problem and emitted token IDs. The
production streaming adapter derives text by decoding exactly those tokens.
Gold answers, correctness, future labels, future text, completion length, final
answers, and future completion fields never enter online decisions. The CPU fixture
interface accepts `problem`, `emitted_token_ids`, and `emitted_text` only; it is
not a license to supply future text under a renamed field.

`prefix-candidate-v1` fixes the legacy incomplete-lookahead bug: plain “answer is”
requires the full 12-character lookahead; closed boxes and closed math delimiters
provide their own completeness evidence. Supported expressions are bounded exact
numeric arithmetic, including decimals, fractions, parentheses and integer powers
with absolute exponent at most eight. Symbolic forms, units, unfinished expressions,
and unsupported syntax abstain. A parser candidate is neither a correctness
assessment nor qualification of any semantic transition detector.

Qualification includes truncated prefixes, unfinished expressions, unsupported
answer forms, reasoning closure, append-only token history, repeated calls, and
forbidden-field/future invariance. Discovery fixtures supply confusion matrices,
coverage, and trigger frequency. Numerical detector thresholds and supported
transition lists are sealed after discovery and before validation. Until that
artifact exists, no semantic detector is enabled. Reasoning closure disables new
actions; candidate occurrences trigger at most once.

## Sparse causal actions

For each supported hypothesis propose one expert set from matched native behavioral
contrasts. Differences are first averaged within question/family; observational
expert use proposes candidates and does not establish their utility. Select at
most four adjacent layers and two experts per layer. The deterministic CPU proposal
reference ranks positive matched differences and chooses the band with the highest
mean positive score, breaking ties toward fewer layers and earlier layer indices.
Only biases +.5 and +1.0 are tested: at most six actions total.

The native router chooses top-k=8 after the logits edit, and the shared expert is
untouched. The CPU pulse dispatcher is an execution reference, not a GPU-qualified
vLLM plugin. A separately versioned worker adapter must pass actual batch isolation,
pulse, recompute, and native-routing qualification before discovery. Its code tree
cannot be substituted silently into legacy s2.

The same-prefix discovery uses 48 families, native plus at most six actions, two
seeds, and 1,024 emitted-token continuations. Every load, preparation, prefill,
decode, retry, and rating is charged. If a supported proposal or an execution cell
is absent, it stays absent; no candidate or sample is added after a near miss.

## Local template and controls

Discovery chooses one template containing at most two supported actions. Its
starting condition, ordered action multiset, target identities, detector thresholds,
pulse slots, random controls, and semantic success rule are sealed before mechanism
validation. Pulses cover branch-relative [0,256) and [512,768), stopping at reasoning
closure. The local endpoint horizon is 1,024 tokens.

Mechanism validation uses 128 other families, four arms, two seeds, and 1,024-token
continuations: native, the frozen ordered template, matched random experts, and the
same action multiset in reversed order. For a one-action template, reversed order
is identical and cannot support an ordering claim; report that arm as a replication.

Freeze four random sets on discovery data with identical layer/count support and
native exposure within ±10% per target expert. Insufficient matched candidates
fail feasibility without loosening the tolerance. Set assignments rotate over
families and seeds; counts per seed differ by at most one. Realized dose is
calibrated on discovery data and reported alongside semantic effects. Exposure
matching alone does not establish matched causal dose.

Native successful paths suggest a template, not a causal optimum. This experiment
tests one fixed local order. An adaptive controller needs a separate evaluation
because interventions change future opportunities.

## Utility scout and inference

The utility scout uses 96 further families, native versus frozen policy, two seeds,
from original prompts, capped at 16,384 tokens. Do not extrapolate beyond 16k.
Legacy X3 retains its independent 32k endpoint.

The primary controllability outcome is completion of the frozen semantic trajectory
within the horizon, excluding the triggering sentence. Report individual transition
effects, routing first-stage effects, expert turnover, gate-distribution TV,
fixed-window velocity/acceleration, dwell, re-entry, loops, actual dose, and executed
expert identities as corrected secondary analyses. The analysis reference defines
velocity as adjacent fixed-window mean gate TV per emitted token, and acceleration
as its signed consecutive change per token. These quantities have no assumed causal
meaning or preferred shape.

Analyze all assigned questions by intention to treat: nonfires, early finishes,
caps, failures, and unscored outputs require explicit receipts. Failure/unscored
correctness is zero; token cost records actual emitted and injected tokens. Missing
receipts make a cell incomplete and are never dropped. Average seeds within
question, then questions equally. Shared families are jointly resampled across
contrasts with 5,000 replicates, seed 20261001. The CPU implementation produces
approximate simultaneous Bonferroni percentile bootstrap intervals across all
semantic, accuracy, and token contrasts of policy versus native, policy versus
random, and policy versus reversed (nine intervals in mechanism validation;
three policy-versus-native intervals in utility). Pass those exact frozen pairs
to `paired_itt(comparison_pairs=...)`; its default native-only summary cannot
establish order or target specificity.
Secondary transition/first-stage profiles form a separate frozen simultaneous
family. Sign-flip results state their exchangeability assumptions; interleaving
alone does not establish exact randomization inference.

Accuracy and token effects are reported separately and jointly with uncertainty.
Savings with an accuracy interval spanning harm leave the tradeoff unresolved.
Similar observed accuracy does not establish retention without a chosen
noninferiority margin; none is chosen in this scout. A degenerate bootstrap is
reported and never called perfect precision. A routing-speed change alone is
insufficient evidence of semantic control. No minimum favorable effect is imposed.

## Resource gates

| stage | complete maximum workload | GPU-h ceiling |
|---|---|---:|
| qualification | detector, pulse, isolation and recovery checks | .75 |
| discovery | 48 × (native + ≤6 actions) × 2 × 1,024 | 2.00 |
| mechanism | 128 × 4 × 2 × 1,024 | 2.25 |
| utility | 96 × 2 × 2, original prompt, cap 16,384 | 4.75 |
| reserve | grading, failures, overhead | .75 |
| total authorized | separate from legacy X2/X3 and existing forum lines | 10.50 |

Authorized on 2026-10-01 by the user's explicit reply, recorded in `steering-v1/runs/resume-v1/RESOURCE_AUTHORIZATION.json`. Price validation first; authorization does not waive qualification or eligibility.
Qualified, context-matched timing must cover the entire stage, all loads, prefill,
decoding, native NLL, GPU labeling, retries, and overhead. Unknown cost inputs hold
submission. If a complete stage cannot fit, stop and propose revised resources.
Do not shorten horizons, remove arms, or consume another line silently.

The worst-case utility decode cap is 6,291,456 tokens. Historical X1 throughput
alone is not qualification of this controller. Its cap-priced utility cost can
exceed 4.75 GPU-h even before measurement overhead, so utility launch remains held
until relevant timing establishes fit or the user accepts a revised proposal.

## Acceptance and reporting

Manifest and code digests bind output directories. Changed caps or code cannot
reuse completed UIDs. Same-manifest resume, partial-file recovery, pulse order,
boundaries, closure, neighboring requests, preemption, offsets, and execution
identities must agree across all artifacts. CPU references retire CPU risks only.
Every submitted job is verified by ID, final state, exit code, and artifacts.

Publish all cells, eligibility, failed and incomplete work, and actual device-hour
expenditure. Claims identify population, intervention, estimate, simultaneous
uncertainty, multiplicity family, and status. If precision is insufficient, use
measured variance/runtime to propose an independent powered evaluation; a promising
pilot is not confirmation.

Paired behavioral contrasts follow the precedent in
[SteerMoE](https://arxiv.org/html/2509.09660v2). Transfer to reasoning transitions
is a hypothesis. The separation between routing association and causal importance
is motivated by the
[expert-importance audit](https://arxiv.org/html/2606.10703v1).

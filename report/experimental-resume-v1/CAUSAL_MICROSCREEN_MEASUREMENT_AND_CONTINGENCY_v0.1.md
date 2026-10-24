# Discovery micro-screen: blinded measurement and next decisions (v0.1)

This is a discovery-family feasibility test. The dense `Verify` native contrast
proposed layer 28 experts 9 and 189 from 483 same-family class matches across
48 families. Dense `Verify` is an **action source**, not proof that these experts
cause substantive verification. The 12 enrolled prefixes are original native
continuations at two-reader locally accepted candidate starts. The full-prefix
start condition needs its own audit. The failed 11-second v1 pilot produced no
generation data; all analyzed outputs must come from a new v2-bound run.

## Frozen question and outcomes

The pilot has 4 families, 6 arms, 2 seeds, 48 assigned requests. The full
discovery screen has the same six arms over all 12 scout families, 144 requests.
The two `zero` arms test duplicate-native isolation; the target arms add
router-logit biases +0.5 or +1.0 to experts 9 and 189 at layer 28; matched
random arms use four frozen native-exposure-matched expert pairs, allocated
equally **within each seed** and without repetition within a family. Every arm
uses the identical saved prompt and native prefix for its family/seed, with a
256-token generation cap and closure-aware pulse.

The **first-stage** endpoint is the fraction of generated reasoning tokens
before `</think>` selecting expert 9 or 189 in layer 28's eight non-shared
slots. Compute it from routed arrays, by assigned arm, on every request with
valid arrays. Record the worker's executed action rows, active pulse length,
selected expert identities, and actual dose separately. Report the same
statistic for each random pair, native duplicates, and each of the other 39
layers as a specificity audit. A bias parameter is not an observed routing
dose. Length stops, early closure, errors, and zero-token requests stay in
the assigned denominator for exposure/availability counts; report conditional
per-token first-stage estimates alongside the intention-to-treat request-level
rate (no eligible token means zero selected-token count, separately flagged).

The **semantic** outcome is a substantive check of the previously unchecked
candidate within the assigned 256-token reasoning continuation, excluding the
triggering sentence. It must test an original constraint or carry out an
independent computation. Words such as “verify” without such an operation do
not qualify. Two arm-blind readers receive the original problem, the **full
exact emitted native prefix** (at most 8,192 tokens), the previous and trigger
sentences extracted from sealed dense units, and only that request's generated
continuation. The frame has opaque shuffled IDs and no arm, seed, family,
future native sentence, gold answer, or correctness field. Start eligibility
(`candidate remains unchecked in full prefix`) and target occurrence are
rated separately. Require both readers to agree on each positive; publish
disagreements, missing ratings, and a sensitivity using adjudication. The
primary all-assigned result counts nonfires, caps, early closure and generation
failures as no completed transition, with their counts visible. A second
analysis reports observed gradeable text only; it cannot replace the assigned
denominator. These are LLM audits of visible text unless independently checked
by humans. No 256-token accuracy-retention claim is possible.

For the full 12-family screen, estimate paired target-vs-native and
target-vs-matched-random changes at each dose; show family-level points and
family-clustered intervals. Correct both doses and semantic/first-stage
secondary claims as a declared family. Four families only qualify mechanics
and context runtime; no confirmatory inference. Duplicate natives must agree
on request identity, emitted tokens, finish, and routing under the same seed;
report any divergence as an engine/isolation failure. The assignment is
deterministic within family, so interleaving by itself does not justify exact
randomization inference. Full validation later requires independent families.

## Grading resource gate

Full-frame exact source prefill is **376,392 native tokenizer tokens** before
two independent judge readings. A judge prompt additionally contains the
problem, trigger, rubric, at most 256 newly emitted tokens, and template
overhead. At native-token parity alone, two reads imply at least 752,784
source-prefix token exposures; actual judge-token counts must be computed
with the frozen Qwen3.8 judge tokenizer from saved texts. The pilot analog is
137,676 source-prefix tokens, or 275,352 exposures for two reads. These are
lower-bound planning quantities, not final judge prefill prices.

The recent **short-context** two-A100 judge parity observed 134.49 aggregate
decode tokens/s. Its long-context stress proxy is 43.71 tokens/s and 4,914
prompt tokens/s, with 723 s cold load and 196 s shutdown allowance. For 288
full-screen ratings with a 1,024-token cap, the proxy alone prices decode at
6,748 s and one load/shutdown at 919 s, i.e. at least ~4.26 GPU-hours before
exact judge prefill, failures and retries. It is **not an approved complete
price**: use the 12-row, 24-rating context-stratified scout and exact frozen
rating prompts to calibrate it, then fit a complete stage under a separate
ceiling. Grade pilot or full outputs only after first-stage results and a
complete price. Independent human auditing may reduce GPU use but also needs
a frozen rubric and measured workload.

## Bounded, versioned contingency after pilot

1. **No executed dose or target entry.** First check the pulse receipt,
   ordering, closure timing and whether targets were already in top-k. If the
   worker applied +0.5/+1 but neither increases target selection, first
   instrument the native and post-action router-logit margin of each target
   expert below the eighth selected expert at the actual pulse tokens. Its
   empirical distribution can select a bounded rescue dose instead of an
   arbitrary grid; this requires a separately qualified, token-bound worker
   telemetry amendment. Then propose a
   new 4-family, six-arm, two-seed 256-token screen of +2/+4 target biases,
   exposure-matched random biases, and duplicate native. Its token maximum
   matches the pilot (137,676 prefill; 12,288 decode); reserve a new 30-minute
   two-A100 envelope (≤1.0 GPU-hour) **only after** fresh runtime pricing.
   The current ordered worker rejects biases above +1, so the +2/+4 screen
   requires a versioned worker-overlay change and full neighboring-request,
   batching, preemption and dose qualification before generation. If large
   bias still cannot enter top-k, separately qualify positive force
   and an equal-slot expert swap under the existing top-k=8/shared-expert
   invariants, with a fresh 4-family matched-random screen and an initial
   ≤0.5 GPU-hour engineering qualification plus ≤1.0 GPU-hour generation
   proposal. Force/swap need dedicated native-neighbor, preemption, batch and
   closure checks before use. This distinguishes unavailable routing support
   from a semantic null.

2. **Target entry rises, substantive checking does not.** Audit whether a
   check already occurred earlier in the full prefix, and stratify by
   trigger-to-action lag, closure and dose. Use paired prefixes at source
   boundary and 64 tokens earlier as an *offline timing diagnostic*; the
   earlier trigger is not an online detector until it is separately qualified.
   Freeze a 4-family six-arm, two-seed 256-token timing screen with matched
   native/random controls (again ≤1.0 GPU-hour proposed generation envelope
   after full stage price). In parallel, use native matched Verify-entry versus
   persistence windows to propose at most one two-expert **deactivation**
   action, then qualify negative force or negative bias and test it against
   matched random deactivations (≤0.5 engineering + ≤1.0 generation proposed
   GPU-hours). A routing first stage with no semantic effect cannot be called
   successful steering.

3. **Substantive checks rise, utility is unknown.** Freeze the supported
   action and boundary rule and advance to family-disjoint mechanism tests,
   then original-prompt utility runs and blind correctness grading. A
   discovery micro-screen never establishes accuracy or token savings.

All contingency budgets are **new proposals**, not transferred from the
10.5-hour study or actual Slurm charges. Do not add candidates because a
confidence interval narrowly misses zero. Use the measured pilot time for
the final complete-stage price and record every failed/resumed job.

The proposals follow paired native contrast and controlled expert interventions
in [SteerMoE](https://arxiv.org/html/2509.09660v2), while respecting evidence
that observational routing scores often fail to predict token-level causal
importance in this [expert-importance audit](https://arxiv.org/html/2606.10703v1).
Positive and negative force exist in the base sampler API, but the current
qualified **ordered worker** accepts only positive +0.5/+1 biases. Force,
negative bias and equal-slot swap each require a new immutable overlay and
engine qualification before an executable causal screen.

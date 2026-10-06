# Literature-grounded recovery of the routing-action study (proposal v0.1)

This is a proposal for a versioned **discovery amendment**, not a change to the
sealed `PROTOCOL_v0.1.md` or evidence that steering already works. The current
two-reader LLM audit found valid candidate starts in 53/200 sampled nonfires
(28 families) and valid approach starts in 60/200 (34 families), while the
existing detector fired only 15 and 3 times across 5,523 contiguous pairs.
The immediate-next-sentence target appeared in 6 and 31 of those sampled
nonfires, respectively. These are reader-rated samples, not population truth.
They show that the present bottleneck is **start detection and timing**, not
proof of an absence of branch opportunities. The failed-check hypothesis has
only two reader-agreed starts in 200 sampled nonfires and should remain a
distinct feasibility cell rather than be merged into a better-populated one.

## Sources and transfer limits

* [SteerMoE](https://arxiv.org/html/2509.09660v2) ranks experts by the
  difference in activation probability between paired behavioral examples and
  intervenes on router logits. Its random and bottom-ranked expert controls are
  useful precedents. Its reported Qwen3 faithfulness intervention deactivated
  hundreds of experts; those results do **not** establish that this study's
  at-most-eight-expert, four-adjacent-layer pulses will work for reasoning.
* [RICE](https://arxiv.org/html/2505.14681v1) uses normalized pointwise
  mutual information with reasoning marker tokens to propose cognitive experts.
  Marker co-occurrence is a cheap proposal score, but an explicit `<think>`
  marker is neither a substantive verification detector nor a causal score.
* [Counterfactual Routing (CoR)](https://arxiv.org/html/2604.14246v1) uses
  expert ablation and the resulting native-model token loss to measure a
  component's functional impact. Its varying top-k allocation is a different
  intervention from this study's fixed top-k=8; a bounded, fixed-top-k
  *expert swap* can test analogous causal value here.
* [The causal expert-importance audit](https://arxiv.org/html/2606.10703v1)
  finds that global utilization, activation norms and routing weights failed
  to predict token-level ablation effects after correction in its three
  studied MoEs. This is a reason to require same-prefix behavioral screens
  before calling an observationally selected expert useful.
* [A pre-registered modularity test](https://arxiv.org/html/2606.25092v1)
  finds that an expert family can have a real causal effect without a selective
  one. Its held-out corpus, size-matched random controls and multiple outcome
  metrics motivate explicit off-target class and quality measurements here.
* [Activation source selection](https://arxiv.org/html/2607.25270v1)
  (non-MoE models) finds that pre-realization execution boundaries can steer
  more effectively than states sampled after the target appears. Transfer to
  routing is untested, but it motivates ranking experts at the last *completed
  start sentence* and first tokens of the next sentence, rather than over
  already-written verification prose.
* [Dynamic Experts Search](https://arxiv.org/html/2509.22572v1) finds gains
  from changing expert count during multi-rollout search with a process reward
  model. Its extra rollouts and external verifier make it a separate utility
  comparator, not evidence about one-pass fixed-top-k routing trajectories.

## Immediate experiment A: recover a qualified start detector

Use only discovery families and the original problem plus emitted token IDs.
Construct an append-only, sentence-boundary event table from the dense labels
and two-reader audit. For each completed sentence, record whether both readers
accept the *start* at that boundary, whether the next sentence realizes the
target, native class, token offsets, boundary punctuation, expression closure
and parser state. The outcome sentence may train/evaluate the offline detector
but can never be an online input. Never relocate a trigger backward after
observing a target sentence: the online trigger is the first boundary at which
the start is recognizable from emitted tokens.

Compare three small, versioned candidate detectors on discovery data: (1) a
high-recall rule set using complete numeric/constraint/method/failure cues;
(2) a calibrated prefix-only text classifier trained on the accepted starts;
(3) a short blinded judge query using **only** the available prefix, if GPU
runtime pricing and boundary latency are acceptable. The lexical rule can
propose boundaries; a second stage can veto false starts. Freeze the action
threshold for start detection before causal validation. Report by family:
coverage, trigger count, start precision/recall against the LLM audit, early
and delayed trigger fractions, and valid future target rate within 1, 2 and
4 complete sentences and 256 tokens. The primary frozen local endpoint stays
unchanged unless a separate protocol version explicitly changes it. Human
adjudication on a small stratified set would strengthen any semantic claim.

The current 0/15 immediate candidate-to-verification targets may be a timing
problem: 6/200 sampled nonfires *did* have a reader-agreed immediate target.
Inspect their preceding prefixes and whether the checker evaluated an original
constraint, rather than merely saying “verify.” This analysis should yield
specific revised rules, not a post hoc declaration that the transition is
absent. Do not use the grader's future sentence in the online adapter.

## Immediate experiment B: causal expert selection before full discovery

The saved native routing tensors and reader-rated discovery windows allow a
CPU-only shortlist now. At each audited start, compare routing in the final
64 tokens before the decision boundary between target-realizing and
non-targeting continuations, matched first within family when possible and
otherwise by source class, token depth, numeric payload, sentence length and
problem type. Keep each family's total weight equal. Predefine an exposure
floor and report the number of families represented by every expert. Score:

1. **Paired risk difference** in top-k=8 activation, following SteerMoE.
2. **Shrunken log-odds or nPMI**, with a minimum exposure floor, as a
   sensitivity ranking; RICE's marker score alone is too nonspecific.
3. **Boundary specificity**: risk difference immediately before a target
   minus the same difference after a completed target sentence. This tests the
   execution-boundary hypothesis and possible lexical contamination.

The sparse proposal remains at most two experts per layer in four adjacent
layers, with the shared expert untouched. Cross-fit proposals across the 48
discovery families: select on three folds, measure native exposure and score
stability in the fourth. First-stage effects require the intervention below.
Do not call observational rankings causal effects.

Run one bounded **same-prefix micro-screen** on the discovery pool once worker
and detector-boundary receipts pass. Select up to 24 reader-agreed candidate or
approach starts from distinct discovery families in frozen hash order; 28 and
34 discovery families respectively currently contain sampled missed starts,
so the cell is potentially feasible. Use identical original problem, prefix,
seed and 256-token horizon for native, targeted positive bias, and
exposure/layer-matched random experts. The proposed +0.5 and +1.0 biases are
the protocol doses. If a versioned amendment adds a negative-bias arm, apply
it only to experts associated with persistent non-target/loop behavior and
keep top-k=8; do not silently import SteerMoE's hundreds-of-experts rule.
Interleave arms; verify native inactivity, neighbor isolation and identical
prefix hashes. Report intention-to-treat semantic transition, first-stage
target-expert selection, actual dose, gate TV, fluency, cap/closure and
failure receipts. Also score whether the action induces unrelated class
transitions or degrades answerability; an effect on every class is weak
evidence for a selective routing control. Correct selection across tested
sets/doses. This directly
answers whether selected experts move routing and behavior at a given boundary,
before pricing hundreds of full continuations.

As a complementary *mechanistic* screen, compare native next-token NLL on
reader-approved target continuations with an expert swapped out for the
next-ranked native expert at the same layer, leaving k=8. Match against
non-target continuations, syntax/easy-token controls, and random experts.
Positive target-specific NLL effects identify candidates worth testing in
generation; teacher-forced NLL on selected continuations does not establish
that free generation will realize the semantic transition. Price this entire
screen, including model load/prefill and retries, before any GPU submission.

## Distinct estimands and stopping rules

Maintain the legacy and new protocols separately. The original v0.1
same-prefix mechanism endpoint is one fixed local trajectory; the micro-screen
above is exploratory discovery. A detector learned from discovery starts can
be frozen and evaluated on the 128 disjoint mechanism families. If one of the
three prespecified transitions has inadequate detector precision, family
coverage or actual routing dose, omit that transition from confirmatory
mechanism testing and report why. Do not refill with a fourth target.

If the new detector finds many starts but no action changes routing, repair
the layer/score interface or use a different expert-selection criterion. If
routing changes but next-sentence behavior does not, test boundary timing,
semantic readout reliability and a versioned negative-bias action before
claiming no semantic effect. If local behavior changes but
original-prompt utility does not, report local controllability and keep the
accuracy/token claim unresolved; utility is a different estimand. Any method
chosen after inspecting the 128/96-family results requires a new independent
evaluation, not a relabeling of those families as confirmatory.

This memo does not price a Slurm submission; the current broad user
authorization still requires complete workload pricing, live account/partition
checks, isolation qualification, and job-ID/artifact verification.

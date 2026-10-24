# Legacy steering-v1 amendment v0.3

Design amendment dated 2026-10-01, implementing the user's experimental resume
plan. This is separate from `PROTOCOL_v0.1.md`, which describes the separately authorized new
routing-action study. Legacy v0.1/v0.2 remain historical records. The sealed v0.3
package incorporates the historical v0.2 text as a reference appendix; this
amendment overrides it wherever they differ. It does not claim that pending
qualification, measurement or execution has occurred.

## 1. Objective, population and evidence status

The objective is expert-routing control of reasoning trajectories, with measured
accuracy and token-cost effects. Report supported policies, effect sizes and
uncertainty. There is no required minimum accuracy improvement or token saving.
Legacy X2/X3 retain their registered diagnostic roles, target sets and multiplicity
rules. The new feasibility/discovery study has priority over launching X3, and its
budget is separate. The completion deadline is flexible.

Historical confirm exposure is disclosed. The original confirm set is not untouched.
All new clean observational fits exclude split-v1 confirm IDs across every model
and alias before any learned transformation. Preserve the original split and
report a separate sensitivity excluding development families connected to confirm.
The complete 117-edge, 43-cross-confirm duplicate graph is frozen before new fits;
the historical ten-pair summary is insufficient for transitive grouping.

K1 supports a narrow bound in its matched population; an MDE does not establish
equivalence. K2 remains inconclusive. Transition-associated routing changes do not
establish causal relaxation or an optimal acceleration pattern. G2 supports
non-rejection under its accepted reading, not engine equivalence. G3 selects doses
and does not establish semantic steering. An unrun coda is neither a null result
nor evidence of uncontrollability.

## 2. Engine and run snapshots (A8, B, C8', C16)

Q0–Q13, G1 and X1 used `s1-f2ded3957eb54fd5`, tree
`f2ded3957eb54fd5a3b4ffa132e0f68f57170c53b6013d885767ac83c466dc17`.
The cap run snapshot s2 is `s1-9a61e32f48c04c24`, tree
`9a61e32f48c04c242acccc89c529bd750776c553cdfc776347151d359bc53430`.
Only `moe_steer/manifests.py` differs: two diff hunks implement optional
`make_request(max_new_tokens=None)` and integer cap validation. Every other runtime
and frozen `moe_exp_src` file is identical. Boolean, noninteger, nonpositive and
above-cumulative manifest caps are rejected. A requested builder cap above the
cumulative limit is clipped. The default None preserves old request dictionaries.
The per-file inventory and tests are included in the seal package.

H1–H4 use the separate test snapshot `s1-126e4f332ffc1aad`, tree
`126e4f332ffc1aad085a492a1409015d4d11f038b66684db01f39d9a6c4fe5bd`.
It adds `qualify_h14.py` and a qualification dispatch patch; its 21 runtime files
remain byte-identical to s1. The fixed `FROZEN_SPEC.json` judges force behavior,
neighbor isolation, prefix behavior and preemption/recompute. H3's historical
layer-24 pass is partial; actual full-engine verdicts are recorded separately.
X2 does not start unless every hole passes. A failed hole holds X2.

Q3's raw routing failure is retained unchanged, alongside the supervisor's separate
adjudication accepting an upstream layer-2 deviation. The same engine triple was
examined at two batch sizes; “163/164” is not independent replication. Q10 failed
its 10% overhead criterion (26.0% at 32 sequences); G1's conditional acceptance
for X1–X3 is disclosed. Neither constitutes engine equivalence. Q12 was replaced
by Q13a/Q13b; optional MTP-on Q12 was not run. Acc32 from long traces relies on the
qualified Q13a EOS-at-cap rule, with final token in {248046,248044} defining natural
stop rather than the engine's length label alone.

Keep Qwen3.6's native top-k=8 and shared expert, TP2, card sampler, prefix-presence
restoration, V1 runner, synchronous scheduling, MTP off, prefix caching off and
FlashAttention settings. The N arm is the schedule's sham policy under the loaded
extension: operator none, routing bitwise native, pulse window recorded. X1 N was
`sham_ALL_onset`; branch N is `sham_ALL_landmark`.

Nominal common seeds/configurations diverge within roughly 10–700 tokens; no CRN
power credit is taken. Retries are fresh draws, never conditional on content, and
every in-flight abort and retry is counted by arm. Stored-token audits are
replications. Interleaving alone does not establish exact randomization inference;
sign flips state the paired exchangeability assumption and build/shard dependence.
An E10 build change separates its results from X1–X3; cross-build causal contrasts
are prohibited and a Q11-scale baseline recheck precedes any future study.

Preempted requests are retained only after H4 passes before X2. Otherwise a grant
without H4 would require voiding/redrawing within the same budget. Every finished
X2 request is checked before semantic outcomes: 1,024 new tokens or natural stop,
force+ contains every target on every active policy layer, force− contains none,
sham has zero active rows, and every E/M cell has positive realized dose. Real
length-capped branches, force at the branch point and routed return at 48 sequences
were not covered by s1 qualification; startup failures cancel the job and remain
reported. Nothing is silently repaired.

## 3. X2 enumeration and eight-hour accounting (C1'', C19, C20)

Use the 39 eligible questions among the original 48 dev-cal questions. The nine
ineligible olympiad questions are listed in the cached eligibility record; there
is no replacement. Eligibility is the first lexical Explore decision index at
least 4,096 in card-v3 sample_00, chosen before continuations. Branch at decision
index +1; landmark equals prefix length. Preserve the manifest lexicon and vocabulary.

One 256-token pulse; seeds 0/1. Bias ±{.5,1,2,4}, force ±, reweight ±{1,2}, at
L1/BAND/ALL give 42 target cells and 42 random cells. Random set k is paired only
with seed k; sets 2–7 are deferred with X6. There are 85 conditions per seed,
6,630 requests: N 78; E+,E−,M+,M− each 1,638. The actual table has **128** policies,
including the required zero policy and sham N; earlier “127” arithmetic omitted
one retained entry. Grid cells and request counts are unchanged.

Every request has `max_tokens=1024` and expected length 1024. Natural stops retain
their tokens and reason; length stops retain theirs. Decode cap is 6,789,120 tokens.
One shard; complete the full grid without cutting cells. The capped dry manifest is
an engineering artifact, not the production experiment seal. Production names
salt UIDs, so the differently named dry run changes UIDs/order/sharding; same-name
comparison verifies all other request fields. Output directories bind the entire
manifest digest and launch-code digest because legacy UIDs omit continuation caps.
Changed caps/code cannot reuse completed records. Crash recovery and same-manifest
resume must neither duplicate nor skip requests.

Routed capture is `{target_policies:[bias_p0.5_ALL_landmark], window:64,
after_pulse:1024}`: top-eight identities for all 30 hooked layers over every
continuation row. Raw uint8 IDs occupy at most 1,629,388,800 bytes before metadata
and compression. Offsets, returned rows, labels and result records must agree.

The cached full cost projection is 4.94 GPU-h, range 4.49–6.75, **including** native
NLL (central 1.01, range .9–1.5), loads and overhead. It is not generation 4.94 plus
NLL 1.01. The entire X2 line is eight GPU-hours. The historical planning line
reserves one hour for resumes/retries, but that is not an additional allocation.
Before each launch, add actual generation/resume expenditure, the complete pending
NLL projection and contingency; refuse commitments beyond eight. A three-hour
TP2 generation job can reserve six hours, so six + pessimistic NLL 1.5 + a whole
one-hour retry would exceed eight. Do not promise all three reservations at once.
Use actual remaining work and measured cost for a single resume; if it cannot fit,
stop and propose resources. The 768-token rescue is superseded by the user's fixed
1,024-token X2 requirement; it cannot be invoked silently.

Registered generation settings remain `max_num_seqs=48`, routed return, time
03:00:00, deadline margin 300 seconds, drain margin 600 and abort margin 300.
These are maxima, not permission to consume an unpriced resume. Every job's actual
devices×elapsed time, including failures and measurements, goes in the ledger.

## 4. Dose, descriptive routing profiles and NLL (A1–A3, C9', C19)

D is mean routing-weight TV on the first 256 steered rows: equal question weights,
seed means, then equal weights over that policy's hooked layers. Print
D×n_layers(scope)/30 beside it. Matching minimizes |log(D_M/D_E)| among magnitudes
of the same operator. Use ±10%; nearest off-band values are flagged, and a ratio
outside [1/1.25,1.25] makes the E configuration ineligible. Force has its fixed
magnitude and requires ±10% matching. M* preserves per-set matched magnitudes.

Prefix rows are not returned, so KL(pulse||pre) is unavailable. Use the registered
descriptive fallback: E/M expert-selection frequencies versus N for the same
question/seed, by hooked layer, in [0,256), [256,512), [512,768), [768,1024).
The native adjacent-window floor is about .26 nats/layer versus a .125 sampling
expectation. KL is never a matching quantity, a minimum effective dose, or evidence
of causal relaxation. Velocity/acceleration profiles are candidate descriptions,
not assumed controls or optimal patterns.

Native surprisal is measured by the separate immutable NLL addendum. For each N or
E branch, average −log p_native(token | original prompt, prefix, earlier pulse
tokens) over up to the first 256 actual continuation tokens. Report short natural
stops/empty pulses and measurement coverage; retain all assigned questions.
Cell means average seeds within question and questions equally. G3 compares cell
NLL minus the paired N-arm cell NLL to .10 nats/token. There are 3,354 sequences and
18,556,822 teacher-forced positions in the capped maximum projection.

The plugin-free native engine uses max_num_seqs 48, memory utilization .80,
long-prefill threshold 1024, max batched tokens 8192, `prompt_logprobs=0`, one
extra generated token, and no detokenization. Those generated tokens and parity
loads count as measurement cost. Before reading X2 NLL, replay the **identical**
Q3 prompt+completion[:2048] fixtures at batch sizes 1/8. Their order, IDs and token
positions are checked. Mean and p99 absolute logprob errors versus the saved plugin
arrays must each be ≤ corresponding free_A/free_B repeat error ×1.25 +1e−6.
Saved repeat means are about .0195/.0187, p99 .278/.270. A zero repeat baseline
requires exact equality. There is no 16-branch/.01 fallback. Failed validation
voids the pass. This logprob validation does not alter raw Q3 routing adjudication.
GPU parity validation remains pending until executed; CPU helper tests do not
qualify the native engine. Atomic per-UID receipts support resume and bind code,
manifest, source tokens and parity files. G3 stays PENDING-NLL until the pass is valid.

G3 retains degeneration ≤N+3 pp, NLL increase ≤.10, relative marker change ≥15%,
and E−M in the same direction and at least half E−N. Degeneration is measured over
the full capped continuation; marker rate over [0,512). No judge labels in X2.
Keep the original ranking. Native marker rate is 2.38/1000 [1.83,2.98]; the 15%
screen passes 14–21% of native-null cells for decreases and 33–47% two-sided.
When N has no flags, two of 78 flags still clear the 3-pp screen (0 of 286 native
windows flagged). G3 is dose selection, not a semantic test. Publish dose curves
and question-bootstrap 95% intervals. No eligible dose holds/cancels X3; report
population-specific estimates, not “does not steer.” No unregistered G3 escalation.

## 5. X3 population, endpoints and feasibility (C4, C5', C6', C18)

Keep 96 original dev-disc questions, seeds 0/1, including nonfires. For each passing
sign, arms are N,E1,M1. The primary is Explore token occupancy in the fixed 512-token
window, denominator always 512. Early termination is absorbing non-Explore;
no-new-sentence branches contribute zero; unknown labels contribute zero and are
reported. Unknown rate above 5% holds interpretation. Segment cumulative
prefix+suffix with original token ownership; exclude the triggering sentence.

The first 64 questions in ascending SHA256(`forum-v1|dataset|source_problem_id`)
receive the same-total-budget 32,768 endpoint: continuation cap 32768−prefix_len.
The other 32 have 1,024 new tokens. The ordered IDs and canonical JSON digest are
included in the appendix; a builder must reproduce them. The order rule was pinned
in controller cycle 5 after native sham planning figures were visible, before any
intervention outcome. The controller reported no alternative order examined for
that pin; the earlier R3-C simulation used a different salt, as disclosed below. Those 64 native figures
are planning descriptions and never eligibility, weighting or stopping inputs.

E4 uses the selected sign's same action with four 256-token pulses, cooldown 512,
onset minimum 4096, and is added only after complete pricing fits X3's separate
12 GPU-hour ceiling including labeling and any escalation. No M4 arm. Its first
pulse is a registered E1 replication, with pooled E1+E4 only a sensitivity; long
endpoints are descriptive dose response. Report one-sign, two-sign, with-E4 and
dev-spare projections. The missing X3 builder and unmeasured label throughput
hold launch; no partial projection licenses execution. The new study has priority.

At the 64-question native planning subset, roughly .773×.139 ≈11% (about 14/128)
attempts are both pulsed and not correct-finish@32k. Harm is observable on many
pulsed attempts; benefit has limited native headroom. Similar observed Acc32 proves
neither retention nor no benefit. Dev-disc native firing is 77.6%; R3-C simulated
77/96=80.2%, and first-64 51/64=79.7%, under its sample_00 lexical definition.
Their different opportunity definitions are retained and not re-simulated.
R3-C's simulated first-64 ordering used `steering-v1|`, whereas the later pinned
registration uses `forum-v1|`. Its subset power is assumption-labelled planning
evidence rather than an exact power calculation for the newly pinned subset.

T512 remains a descriptive causal screen with stated precision. For a pulse
mechanism shift .10, attenuation a=.52, power across rho={0,.25,.5,.75} is
.81/.70/.60/.51 promote and .84/.73/.63/.55 suppress. Averaging attenuation
uncertainty gives .73–.51/.77–.53. Neither rho nor target-domain a is measured.
The 407-unit/six-document confusion gives a=9/17−3/390=.52 with document-bootstrap
.23–.85; it is not known truth. Observed ITT MDE is +.031–.041/−.028–.036;
mechanism MDE at a=.52 is +.099–.132/−.095–.133, about .06–.08 at a=.85 and
.23–.30 at a=.23. Latent per-branch occupancy MDE .074–.098 is a different scale.
Acc32 conditional MDE is 9.6–16.2 pp under the reported discordance scenarios;
power at 5 pp .10–.19. Illustrative SD=.30 is kept separate. Acc32, log tokens
and finish@32k stay descriptive; no minimum improvement is imposed by this amendment.

## 6. Gates, claim families and audits (A4–A5, C14–C15)

G4 local gate: E−N two-sided sign-flip p<.05/3 and 98.33% t-interval excludes
zero; E−M same direction, one-sided p<.10; safety screens pass. The historical
finish/log-token point-sign clause passes 68% of simulated nulls and is now
descriptive, not a gate or outcome class. Gate passes are local gate-level effects.
Claim-level inference retains six reserved contrasts, .05/6 and 99.167% intervals;
H1 and H1s must pass their claim rules for target-specific control. No automatic
X6/X7 unlock. The former E* larger-finish rule (ties suppression) belongs only to
a future addendum, never current paper claims.

An efficacy-only G4 failure with safety passing permits the original one dev-spare
escalation: 56×2 seeds×N,E,M, twice matched bias/reweight (bias≤8,beta≤4), no force,
1,024-token local endpoints, separate analysis, never pooling. Execute only if the
complete X3 projection including it fits 12 GPU-h. Harm/degeneration/NLL failure
ends the branch. Nothing is added because a result narrowly misses significance.

Next-{3,10}-sentence opportunities are descriptive, with unsimulated power of
other secondaries disclosed. Holm covers the four registered secondaries: re-entry
within 1024, marker rate, censored tokens-to-first-candidate with parser coverage,
candidate changes. Window KL remains descriptive. No endpoint/window promotion.

Freeze J1 model/prompt/parser/sampling and the GEPA program/labeler settings. The
X1 facts are 1900/1900 completed records, eight complete shards, zero TP-rank
disagreements, preemptions, aborted-in-flight or infrastructure retries. J1 used
228 items, 228 verdicts, 102 equivalent/126 not equivalent; pending zero. The
historical 59068750 parent was COMPLETED 0:0; its server step's cancellation is
recorded separately rather than hidden.

G2's accepted closed paired-difference-interval reading was fixed after X1 numbers
were visible. All five registered benchmarks pass; alternative readings and
degenerate/boundary intervals remain diagnostics. The exact G2 JSON is included
without rewriting its verdict. M7 remains baseline-only: sham X1 saved neither
target-hit telemetry nor routed windows, so exposure/routing contrasts are
INCOMPLETE unless separately funded replay occurs. M7 does not gate X2/X3.

The 200-sentence intervention audit is arm-blind, balanced across arm and predicted
Explore/non-Explore, hash-ordered within stratum. Two independent LLM sessions use
UID-only text and the frozen rubric, blind to each other and the labeler; a third
resolves disagreements by majority. Report it as an LLM audit, never human gold.
Use design weights from the prediction-stratum population shares; print raw balanced
figures beside them and inter-reader kappa. Weighted agreement<.85 or a claim
contrast disagreement gap>.10 holds interpretation. Two hundred balanced sentences
cannot bound Explore recall within ±.10. Do not estimate attenuation from LLM votes;
use uncertainty-aware human-confusion sensitivities. Human latent-reasoning claims
need further measurement validation. The discordant correctness audit is deferred
with X6; operational missing/unscored/capped outputs count wrong.

## 7. Deferred studies and interpretation (A9–A13)

X6/X7 are paused; A9 is off. A routing trigger needs a separately frozen deployable
detector, not a historical oracle current-class label. A future P* requires the
question/seed/block salted RNG without arm IDs and a renewed E2 agreement check.
A12's .0285 is the scenario-(i) eight-seed MDE, not a tolerance; SD_q(D_q,4) would
come from future tune-L four-seed contrasts. The R1-C4 “≤2 pp” correction denotes
Acc131−Acc32 headroom in pool S, not an accuracy effect. Future rescue terminology
is retained: Acc32 gain with paired Acc131 upper bound<+1 pp is budget-conditioned
rescue; lower bound>0 supports accuracy gain; otherwise Acc131 is inconclusive.

Savings with an accuracy interval spanning harm leave a tradeoff unresolved.
Similar accuracy does not establish retention without a chosen noninferiority
margin. No such margin is imposed by the new utility scout. Corrected simultaneous
uncertainty is required for semantic control, accuracy and tokens. No successful
steering claim follows solely from routing speed, marker vocabulary or a dose gate.

## 8. Seal inventory and execution holds (A12, C13, C17, C21)

The inventory hashes exact protocol bytes, all old and cap/test snapshot files,
J1 configuration and provenance, frozen GEPA program, labeler code/settings,
split/targets/random/lexicon/X1/G1/G2 seals, raw Q3 and parity archives, H1–H4 status
at seal and its fixed spec/proposal, R3-C specification/results/estimates/amendment,
duplicate graph, cap patch/proposal/tree, updated X2 builder/tests, cached enumeration,
cost/eligibility/NLL feasibility and capped dry manifest. Missing required files
refuse sealing. Later qualification outcomes are separately dated checkpoint
receipts; they cannot rewrite a sealed protocol.

Sealing fixes the design, not the execution status. Actual H1–H4 must pass before
X2; actual native parity/NLL must pass before G3/X3; X3's missing builder and full
cost remain launch holds. The separate 10.5 GPU-hour allocation was explicitly
approved by the user on 2026-10-01 together with 0.75 GPU-hour for H1–H4 recovery.
Its qualified worker, detectors, eligibility and complete stage prices remain
gates. The approval is recorded verbatim in
`runs/resume-v1/RESOURCE_AUTHORIZATION.json`; no stage may borrow another stage's
allocation. The first H1–H4 attempt, job 59090613, used 0.48056 GPU-hour and left all
four checks not evaluated because of the deadline. The separately authorized
recovery, job 59092583, was running at design seal.

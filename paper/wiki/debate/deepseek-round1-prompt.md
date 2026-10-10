# Independent scientific critique round 1
You are the user-requested DeepSeek V4.1 Flash reviewer at maximum reasoning. All evidence is supplied below. DO NOT use tools, run shell commands, read other files, launch jobs, edit files, inspect credentials, or call external services. Return your scientific report only, maximum 1800 words.

The user says months of Codex conversations have lost the research thread, repeatedly reinventing tasks and launching unhelpful jobs. They ask for a durable evidence wiki and defensible method/contributions. On 2026-10-07 they explicitly accepted the central question: causal control of reasoning transitions OR well-supported limits, with accuracy and total token cost measured jointly. Positive efficiency is not mandatory. The task authorizes analysis/documents and this review, not Slurm jobs.

Critique sharply and offer a minimal coherent route. Separate observed evidence from hypotheses, engineering checks and missing measurements. Preserve existing family freezes (32 validation/96 final, 2 seeds), caps and outcomes unless identifying a proposed amendment. No new approval/workload claims. Every proposed job must have one falsifiable question, reused inputs, primary endpoint/controls, decision consequence, and stopping rule; you cannot invent resource prices.

Preliminary cross-checks: September observational steering audit found only 51 distinct B questions and whole-trace router gap vs length rho near -.45 attenuating toward zero within absolute positions. Later response-grouped nested probe analysis repairs legacy sentence leakage: 38 responses/3087 units, macro-AUROC Qwen .852 [.838,.868], GPT .780 [.761,.797], Gemma .788 [.772,.805], gains over position +.286/.209/.221. This does not validate intervention layers or generalization to new math populations.

A/B/C/FRESH causal studies already exist. FRESH 3480 assignments, 2920 valid substantive reader pairs, 560 unknown, 123 families/134 starts. Candidate-to-verify force-native +5.5pp simultaneous CI [-4.5,+15.5], force-random +11pp [-.5,+23.5]. No contrast in its 28-contrast family strictly excludes zero. Different horizons/readers and seven-class descriptions cannot be pooled. Later historical dense recovery is complete: 187136 valid, 388 unusable, zero unresolved among 187524 identities; the older v1 REPORT below has obsolete partial counts. Completed labels are not independent questions.

Newest semantic proposals already cannot satisfy support: 48 apparent prefixes propose complete_unchecked=0, complete_checked=0, identified_approach=4; held-out 16 propose 1,0,2. Ten apparent eligible families required. Serialization repair alone does not solve this. Backend passed 13 cases but no v2 scientific effects exist. Native route variation in separate forced-token runs is not failed within-execution no-op fidelity.

Private reference paper: Rational Manifolds: A Nonlinear Atlas for LLM Reasoning, supplied by user. It fits unsupervised mixture of factor analyzers to sentence residual activations, maps regions to 8 episode types (incl Answer), and perturbs residual-space regions. Our setting is sparse MoE expert selection/weights and 7 episode labels. Borrow test structure (observation -> information beyond nuisance -> dose -> substantive displacement -> utility), not its layer/region/causal claims. Full reference review follows later.

Required report: (1) defensible contribution hierarchy and what is not yet defensible; (2) the 5 most consequential scientific mistakes or hidden assumptions; (3) minimal closure plan using current assets, especially best metrics and layer-selection logic; (4) whether context clustering, smaller budgets, and SSD prefetch belong to main paper or later; (5) hard questions to the project lead. Be specific, do not create a massive program or require impossible guarantees. State uncertainty and source scope.


## Source paper/progress_report.md
analyses:
- correlation between routing/hidden_states/expert selection vs. accuracy, token count
- Which layers - check the best one to intervene on — creating a script/pipeline that based on our findings process a moe against a benchmark and extract the most promsing one
- patterns in expert identity activation throughout the forward pass
- sate transition between reasoning classes against expert usage
- analysis on effect of expert pre-selection to reduce expert swap and unpredictability - this represents a further  findings with the goal or showing how we could predefine a smaller set of experts to use so as to allow for SSD-streaming services where we want to preload experts and not wait the last moement for routing decision


Eventually consider setting a lower thinking budget:
- default model accuracy would drop due to reasoning truncation — we try to optimise the reasoning length to more optimally fill that space and reach an answer

We need to clarify first of all the best correlation metrics that will inform our steering process. Exploiting and analysing all the analysis we did in the last month. So doing a careful overview of everything so as to not reinvent everything every time, clearly defining a wiki to further guide new analyses missing. 

## Source report/routing-study-v1/REPORT_v4.md
# Routing study: evidence and remaining execution

Historical A/B/C/FRESH estimates and their saved 50,000-draw intervals were reproduced exactly. Each population, cap, endpoint and inference remains separate. Capped mechanism results do not establish final-answer utility.

| Study | Cap | Assigned | Valid semantic pairs | Interpretation |
|---|---:|---:|---:|---|
| A | 1024 | 128 | 110 | Discovery mechanism evidence |
| B | 1024 | 156 | 141 | Discovery mechanism evidence |
| C | 256 | 208 | 200 | Discovery mechanism evidence |
| FRESH | 1024 | 3480 | 2920 | Discovery mechanism evidence |

Dense recovery inventory: 11,200 committed of 187,524; 176,324 missing; 8 old calls without committed outputs retained explicitly.

A does not establish an advantage of paired verification experts over either singleton. B does not establish additional benefit of the second pulse. C shows an exploratory reweighting effect in its short horizon; the degenerate bias estimate does not establish equivalence. These are substantive reader outcomes, distinct from the seven-class sentence descriptions.

| Study/scope | Contrast | Substantive success change (pp) | Frozen simultaneous 95% interval (pp) | Families |
|---|---|---:|---|---:|
| A/all | pair − native | 6.25 | [-25.00, 50.00] | 8 |
| A/all | expert189 − native | 0.00 | [-25.00, 25.00] | 8 |
| A/all | expert9 − native | 6.25 | [-25.00, 37.50] | 8 |
| A/all | pair − expert189 | 6.25 | [-25.00, 50.00] | 8 |
| A/all | pair − expert9 | 0.00 | [-25.00, 25.00] | 8 |
| B/all | once − native | -7.69 | [-30.77, 15.38] | 13 |
| B/all | repeat − native | -11.54 | [-42.31, 0.00] | 13 |
| B/all | repeat − once | -3.85 | [-26.92, 15.38] | 13 |
| B/candidate_to_verify | once − native | -12.50 | [-43.75, 25.00] | 8 |
| B/candidate_to_verify | repeat − native | -18.75 | [-62.50, 0.00] | 8 |
| B/candidate_to_verify | repeat − once | -6.25 | [-37.50, 25.00] | 8 |
| B/approach_to_commit | once − native | 0.00 | [0.00, 0.00] | 5 |
| B/approach_to_commit | repeat − native | 0.00 | [0.00, 0.00] | 5 |
| B/approach_to_commit | repeat − once | 0.00 | [0.00, 0.00] | 5 |
| C/all | bias − native | 0.00 | [0.00, 0.00] | 13 |
| C/all | force − native | 0.00 | [-23.08, 23.08] | 13 |
| C/all | reweight − native | 3.85 | [0.00, 19.23] | 13 |
| C/all | force − bias | 0.00 | [-23.08, 23.08] | 13 |
| C/all | reweight − bias | 3.85 | [0.00, 19.23] | 13 |
| C/candidate_to_verify | bias − native | 0.00 | [0.00, 0.00] | 8 |
| C/candidate_to_verify | force − native | 6.25 | [-25.00, 37.50] | 8 |
| C/candidate_to_verify | reweight − native | 6.25 | [0.00, 31.25] | 8 |
| C/candidate_to_verify | force − bias | 6.25 | [-25.00, 37.50] | 8 |
| C/candidate_to_verify | reweight − bias | 6.25 | [0.00, 31.25] | 8 |
| C/approach_to_commit | bias − native | 0.00 | [0.00, 0.00] | 5 |
| C/approach_to_commit | force − native | -10.00 | [-40.00, 0.00] | 5 |
| C/approach_to_commit | reweight − native | 0.00 | [0.00, 0.00] | 5 |
| C/approach_to_commit | force − bias | -10.00 | [-40.00, 0.00] | 5 |
| C/approach_to_commit | reweight − bias | 0.00 | [0.00, 0.00] | 5 |

The complete saved contrasts, token effects, and missing-reader identification bounds are retained in HISTORICAL_INTEGRATION.json. Small-family and degenerate bootstrap limitations remain applicable.

![Historical pulse trajectories](historical-pulse-trajectories.png)

The 0–256 and 512–768 windows are the first and repeated pulse positions. Post-pulse descriptions use only exact saved expert IDs and sentence token ownership. Unknown labels break adjacency; closure competes with state exits; caps censor unobserved exits. The complete window table separately reports routing selection, class occupancy, unknown labels, observed reasoning duration and closure. Window comparisons are descriptive and carry no new significance claims.

Bias and forced inclusion are compared using the frozen inclusion-per-intended-opportunity endpoint. Reweighting also requires executed selected-weight displacement: sparse L1/2 and target mass from saved execution-local telemetry. The dose table retains TP ranks separately and excludes padded inactive layers. Its weights are normalized executed top-k weights, not full router probabilities. Exposure, displacement and semantic effects are not treated as equivalent.

![Global and conditional expert profiles](global-conditional-heatmap.png)

44,442 expert-condition rows meet the ten-family/three-fold direction rule for class-transition proxies. No new singleton is selected: candidate completeness and checked status still need a prefix-only semantic recognition audit. Global versus conditional signs, ranks and cosine comparisons are recorded. The prospective predictive interface fits feature selection and regularization inside family folds, but held-out predictive gain is unavailable without a qualified prefix baseline and trailing feature frame. Full router confidence and native selection-boundary gaps are unavailable.

Unchecked-candidate verification and approach commitment remain hypotheses requiring qualified prefix opportunities. Already-checked candidate closure and failed-check revision have insufficient qualified support. No causal specialist or original-prompt accuracy claim is made.

The separate native-finalization prompt-control diagnostic has zero of 32 outcomes. Native repeat variability and execution-local capture were audited; the v4 exact native-token repeat gate failed. It remains an engineering qualification failure. Counterfactual-NLL qualification remains failed; X3 remains deferred. Qualified free-generation analysis can proceed independently.

The full 5,000-question MATH test source is pinned at revision 21a5633873b6a120296cce3e2df9d5550074f4a3. All 500 prior MATH-500 texts matched. The duplicate graph preserves prior family connections and leaves 4,479 eligible new families. INDEPENDENT_FAMILY_MANIFEST.json now freezes 32 validation and 96 final families with disjoint identities and subject/level stratification. The approved CPU work completed in jobs 59493569 and 59494618: the initial job wrote a valid sealed graph but failed a tuple/list equality check; the bounded completion reused that graph. Total elapsed allocation was 178 seconds on two CPUs, within the approved two-core-hour limit. No new-family outcomes have been generated or used for selection.

The independent analysis implements seed averaging within family, equal family weights, paired simultaneous 95% accuracy–cost intervals, missing-grade sensitivity, and the stated deterministic validation choice. No acceptable accuracy-loss margin is imposed. The frontier is unavailable until qualified policies and independent outcomes exist; a nonsignificant accuracy difference will not be reported as accuracy retention.

| Resource component | Observed / authorized status |
|---|---|
| Existing dense labels | 36 packed jobs submitted and verified; maximum 1707.24 GPU-hours including bounded recovery; dependent analysis job 59494071 |
| Packaging qualification | Original 59490750 failed at the IPC path limit; corrected 59491564 completed with 64/64 valid stopped labels and exact label agreement across two original shards in one load |
| Existing utility pilots | Jobs 59405556, 59405561, 59405635 retained and reconciled |
| Existing operator pilots | Jobs 59400762, 59400764, 59400766 retained and reconciled |
| Offline integration | Two CPUs, at most one hour; historical authorized analysis |
| Independent family freeze | Approved and completed within two CPU-core-hours, zero GPUs; 32/96 family freeze verified |
| New target / operator / timing / evaluation | Not submitted; complete measured generation, prefill, controller, labeling, grading, load and bounded-recovery prices required |

Policy cost includes controller-generated tokens. Processed input, controller overhead, runtime, reasoning tokens, and offline labeling/grading expenditure will be itemized separately. Accuracy damage and net correct-to-incorrect flips per exposure remain unavailable without paired final-answer grades; near-zero-dose ratios will stay descriptive.

Current machine-readable inputs: STUDY_INVENTORY_v3.json, CANDIDATE_TABLE.json, INTERVENTION_MANIFEST_v2.json. Earlier immutable versions preserve their original observations. Historical source artifacts remain unchanged. This study is incomplete while label recovery, queued pilots, prefix qualification and independent evaluation are outstanding.

![Transition-aligned native routing](native-transition-aligned.png)

In this plot, source sentence 0 precedes destination sentence +1. Every included offset has a contiguous valid label chain; events and seeds are averaged within family before families receive equal weight. This separates native pre-transition profiles from routing in the destination. These sentence-scale class proxies do not qualify a 64-token timing trigger or establish a substantive verification. Token-scale anticipation remains unavailable until raw prediction-row alignment and a prefix-only baseline are qualified.

![Native state-transition diagram](native-state-transitions.png)

The diagram shows up to two most frequent observed exits per state. Edge labels are destination probabilities conditional on an observed state exit. Self transitions, all exit counts, dwell, re-entry and loops remain in NATIVE_BOUNDARY_PROFILES.json. Missing labels break adjacency. Selected-window ends censor termination, so no closure probability is inferred from them.

The corrected packaging changed only private runtime path length and model-load sharing across existing shards. Original prompts, sampler, assignment identities and valid saved labels are preserved. All 36 GPU receipts and the dependent analysis receipt are in submissions-v5/RECOVERY_CHAIN.json; pending jobs are not completed label outcomes.

The new intervention_design_v1.py and policy_stages_v1.py contracts freeze at most two singleton targets, signed screens, operator comparisons, calibrated controls, delayed assignments and supported composition. The online controller derives text from already emitted IDs, records controller expenditure, and retains no-fires after substantive acts, closure or caps. These interfaces still need qualified recognizer and GPU capture/executor bindings before new stages can run. Worker-specific price components and dose-normalized answer-flip diagnostics are implemented separately in policy_stages_v1.py and resource_analysis_v1.py.

Verification: 54 relevant tests passed. Saved historical effects and 50,000-draw intervals reproduced exactly; frozen recovery source hashes remain unchanged; every packed submission has an exact verified Slurm job ID; independent source and family seals were read back. The independent accuracy–cost frontier remains unobserved.


## Source docs/routing-study-v2.md
# Routing study v2

This extension tests sparse routing at locally qualified opportunities on
Qwen3.6-35B-A3B-FP8. Every new scientific continuation has a 16,384-token cap;
substantive transitions within the first 1,024 tokens are a separate endpoint.
The local-context hypothesis comes from the private Rational Manifolds paper
as specified in the study plan. This implementation uses expert-ID composition,
not the paper's residual-stream interventions or hidden-activation MFA.

Historical v1 modules, 256/1,024-token contracts, assignments, reader outcomes,
inference and source bindings remain immutable. In particular,
`routing_study_v2.sbatch` is the existing dense-recovery wrapper. The new
`routing_study_local_v2_cpu.sbatch` is a different, unsubmitted CPU wrapper.

## Interfaces

| Module | Purpose |
|---|---|
| `study_v2.py` | 16k protocol, semantic recognizer qualification, earliest-prefix singleton/operator/timing/context screens, advancement, dispatch gates |
| `families_v2.py` | Sealed discovery/validation/final separation bound to the existing source and family freezes |
| `context_v2.py` | Eight-component PCA, K=1/2/4 clustering, nested family tuning and outer prediction, local association support, K=1 fallback |
| `policy_stages_v2.py` | First-opportunity state/context/random-skip triggering, delayed no-fires, closure, non-overlapping pulses, request-local snapshots |
| `generation_v2.py` | Synchronous one-token request loop consuming an independently qualified backend; all-layer captures and execution-local fidelity/dose checks |
| `anticipation_v2.py` | Prefix hashing/text-position-state-history-run-age baseline; trailing 64/256 ID features and composition-preserving order controls |
| `mechanism_v2.py` | Arm-blind, two-reader early substantive endpoint; missing labels break adjacency; closure competes and windows censor |
| `evaluation_v2.py` | Supported sequential composition, unchanged independent enrollment, incumbent freeze, matched final comparators |
| `utility_analysis_v2.py` | Seed averaging and family-equal frontier; simultaneous intervals across every frozen accuracy/cost contrast; missing-outcome bounds |
| `resource_analysis_v2.py` | Exact-contract stage pricing of loads, prefill, generation, controller, replay, capture/storage, annotation, grading, runtime and bounded recovery |

These interfaces do not create passing qualification receipts. The concrete
`routing_local_backend_v2.py` adapter now connects the controller to the frozen
utility generator and routing kernels. It uses one eager synchronous TP=2
request, acknowledges every action on both ranks, and preserves the sampler and
engine across live pauses. `routing_local_worker_v2.py` captures native/executed
top-eight IDs and float32 selected weights at the same hidden states in all 40
layers on both ranks. `capture_v2.py` writes binary arrays and byte hashes,
retains partial attempts, and permits one whole-assignment retry from the
original prefix after process death. A seed restart is a new attempt.

The bounded engineering path can collect observed receipts before a scientific
contract exists. Scientific dispatch still requires independent passing
backend/semantic qualifications, exact execution bindings, measured component
prices and authorization. Existing ID-only captures and short-horizon workers
cannot stand in for that evidence. Native routes may differ between runs; inactive-path checks
compare native and executed routing at the same hidden state within execution.

The recognizer needs prefix-only discovery audits of intermediate, unchecked
complete, checked complete, identified approach, substantive verification and
substantive commitment. Each decision needs ten eligible families, at least
70% apparent precision, explicit disjoint held-out uncertainty and coverage,
and audited symbolic candidates. A fallback native-model veto needs a bounded,
complete controller price. Missing or ambiguous decisions abstain.
`routing_local_semantics_v2.py` supplies conservative prefix proposals,
including symbolic equalities, and a two-reader six-decision rubric.
`routing_local_audit_v2.py` rates a frozen maximum of 64 discovery prefixes
with a 1,024-token cap per reader. Family splits are frozen before ratings.
Only observed semantic qualification enables its four outcome-blind pricing
cells: one frozen family, native/incumbent, seeds 0/1, each capped at 16,384.
The audit is allowed to fail; incomplete work never becomes a passing receipt.

Context training consumes native ID inclusion profiles of shape `(N,40,256)`,
with frequencies in `[0,1]` summing to eight per layer. Selected-weight profiles
remain a separate signal. Every inner fit contains training families only;
outer gains compare nested-selected contexts against K=1. Multiple contexts
need positive gains in three outer folds and ten-positive/ten-comparison-family
expert association support. Context labels are descriptive. The acceptance
rule and stratum skip probabilities are separately frozen on discovery's first
qualified opportunities. A declined opportunity cannot be revisited.

New screen families use seeds 0 and 1, the earliest qualified prefix per family
and transition, and at most 24 families in fixed hash order. Positive actions
advance exploratorily only with verified dose, positive semantic point effects
versus native and matched random, three positive folds and reported simultaneous
intervals. Negative actions remain necessity/interference diagnostics. Operator
ties use lower cost, fewer intervened positions, then bias/force/reweight order.

Snapshots bind the assignment, complete prefix, decisions, opportunities,
controller costs, capture and the exact engine/RNG cursor. They cannot establish
GPU RNG continuity by themselves; seed-only restarts are rejected. An executor
must observe every emitted token after its frozen initial prefix. At most one
pulse occurs per transition and two per request; pulses cannot overlap. The
closure-producing row remains captured, and answer rows use native routing.

## Finite commands

Use the installed project environment and `PYTHONPATH=src:scripts/experimental_resume`.
`prepare` performs bounded metadata orchestration, saves additive artifacts and
reads Slurm states without submitting or modifying existing jobs:

```bash
python -B scripts/experimental_resume/routing_study_local_v2.py prepare --output-dir report/routing-study-v2/NEW_VERSION
```

The following analyses require an allocated CPU Slurm **step**, a fresh output
file in the user's approved HOME/WORK subtree, measured pricing and explicit
stage resource authorization before submission:

```bash
python -B scripts/experimental_resume/routing_study_local_v2.py dense-ledger --source report/routing-study-v1/dense-recovery-v3 --output report/routing-study-v2/NEW_VERSION/DENSE_IDENTITY_LEDGER.json
python -B scripts/experimental_resume/routing_study_local_v2.py context --frame QUALIFIED_FRAME.json --output CONTEXT_RESULT.json
python -B scripts/experimental_resume/routing_study_local_v2.py anticipation --frame QUALIFIED_FRAME.json --output ANTICIPATION_RESULT.json
```

The dense command retains exactly 187,524 assigned identities, including every
committed unusable result and every unresolved attempt. It reads historical
receipts and creates an overlay ledger; it neither submits nor retries work.
The already queued recovery, analysis and closeout should finish before a new
snapshot is interpreted as complete accounting.

## Independent evaluation and resources

The original 32 validation and 96 final families, source revision and two seeds
remain fixed. Generate native validation first. Freeze the qualified incumbent
before validation: forced inclusion of L28/E9+E189 and +1 L24/E159 bias. It is an
unproven comparator. At most two qualified validation policies are allowed:
state-only and its context-aware counterpart with identical selected actions.
Use the existing deterministic highest-accuracy/non-increasing-cost choice;
if none qualifies, advance no cost-improving winner. Final arms are native,
selected, matched random and incumbent: at most 768 assignments.

`evaluation_v2` freezes selected-minus-native/random/incumbent accuracy and cost
contrasts. `utility_analysis_v2` uses one family-bootstrap simultaneous interval
system across all those contrasts, averages seeds within family and weights
families equally. Missing grades have worst-case accuracy sensitivity; missing
cost has an unbounded upper limit. No acceptable accuracy-loss margin is chosen.
Degenerate intervals or nonsignificance cannot establish retained accuracy.

`price_stage` requires all eight resource roles to be itemized, including
explicit reasons for zero-work roles. Scientific generation and replay prices
must bind the exact 16k execution contract. Raw capture storage reserves uint16
IDs plus float32 weights at all 40 layers and every token, including bounded
retries and metadata. Policy cost is generator plus controller output tokens;
annotation/grading remain research expenditure. A successful price is not
authorization: `dispatch_gate` additionally binds approved stage/manifest/price,
resource ceilings, current source hashes, and live LEONARDO account/partition/QoS.

CPU fixture tests establish information and schema boundaries. Scientific
outcomes, semantic precision, actual selected-weight capture, GPU fidelity and
sampler continuity remain unavailable until their own observed passing receipts.
Native-finalization failure, failed counterfactual-NLL qualification and deferred
X3 stay separate; qualified free-generation analysis does not depend on them.

## Additive bounded implementation outputs

`report/routing-study-v2/implementation-20261006-v1/` contains the engineering
manifest, authorization, actual Slurm submission receipts and frozen audit
frame. Successful CPU preparation is in `preparation-v2/`; the failed first
preparation remains in `preparation/`. `SUBMISSIONS.json` records actual IDs,
including failed jobs. Later submission-ledger versions are additive.

`routing_study_receipts_local_v2.py` binds real discovery traces, tensor files
and exact prefix token/text offsets. Raw historical input-row alignment and
offline semantic outcomes must qualify before context/anticipation fitting.
It also adapts scientific generation receipts to the existing two-reader early
endpoint and arm-blind strict/J1 grading formats. Gold enters offline grading
only. `routing_study_prices_local_v2.py` exports observed component measurements
without assigning zero prices to unmeasured replay, grading or death recovery.
The finite closeout reads final accounting and reconciles all 187,524 historical
identities after the existing recovery/analysis chain finishes; it queues no
duplicate historical work.


## Observed bounded qualification (2026-10-07)

GPU job `59532973` completed all 13 backend qualification cases, including the
16,384-token engineering request, both-rank all-layer capture, inactive-path
fidelity, six live-pause receipts and a bounded interrupted-assignment retry.
The 16k request took 2,637.33 seconds and allocated 136,231,573 capture bytes.
Native routing showed cross-run variability, retained separately from
execution-local fidelity. These deterministic fixtures do not measure utility.

GPU job `59538244` completed at the scheduler level but its semantic pilot
explicitly failed before any of the 128 assigned readers produced a rating.
The installed tokenizer returned `BatchEncoding`; passing it directly into the
prompt-ID JSON digest raised a serialization error. All four pricing cells
were skipped. Its model-load and shutdown expenditure is retained, with reader,
replay, grading and scientific generation prices unavailable.

The additive `routing_study_audit_local_v3.py` and
`routing_study_reader_prompt_local_v3.py` normalize and validate actual input
IDs, preflight all prompts before model loading, and require a separate exact
v3 authorization. Qualified v2 source bytes and receipts remain intact.
The correction is not GPU-qualified and no further GPU allocation is submitted.
Both actual GPU allocations used 2.6717 GPU-hours in total, within the approved
16 GPU-hour envelope. Exhausting the two approved job shapes does not authorize
additional allocations using the remaining elapsed-hour difference.

Final allocated reconciliation and observed accounting are published under
`report/routing-study-v2/implementation-20261006-v1/closeout/`; the subsequent
`final/` snapshot records finished job accounting without rerunning dense work.
Scientific dispatch remains gated on semantic qualification, native historical
row alignment, complete matched component prices and separate authorization.


## Source report/routing-study-v2/implementation-20261006-v1/final/STATUS.md
# Routing-study v2: final bounded implementation status

The real TP=2 backend is implemented and GPU-qualified. Semantic qualification,
complete scientific component pricing and scientific outcome comparisons remain
incomplete. All approved submissions are finished and their receipts are retained.

| Evidence | Observed result |
|---|---|
| CPU preparation | 33 focused tests passed in job 59532701 |
| Final receipt and serialization regressions | 10 tests passed |
| Real backend, capture and live state | 13 engineering cases passed in GPU job 59532973 |
| 16k engineering request | 16,384 tokens, 2,637.33 seconds, 136,231,573 capture bytes |
| Semantic pilot | Job 59538244 finished, but the pilot failed before any of 128 assigned ratings |
| Serialization correction | Installed tokenizer checked on all 64 frozen prefixes; token IDs unchanged; no GPU qualification |
| Four outcome-blind scientific pricing cells | Not run because semantic qualification failed |
| Historical identities | 187,524 retained once: 187,136 valid, 388 unusable, zero unresolved |
| Final CPU reconciliation | Job 59587910 completed with exit 0:0 |
| Actual allocated use | 2.6717 GPU-hours; 0.1672 dedicated CPU-core-hours |

The first preparation job 59531949 failed on serialization and is retained.
The corrected preparation, native-source binding, both GPU allocations and final
CPU reconciliation completed at the scheduler level. Scheduler completion is
separate from the explicit failed semantic-pilot receipt. No duplicate historical
assignments were queued.

The semantic failure was a BatchEncoding passed directly into the prompt-ID JSON
digest. The additive v3 adapter normalizes actual IDs, rejects malformed or batched
prompts and preflights all prompts before model loading. All 78 source hashes bound
to the original v2 GPU qualification remain intact. The correction is CPU-verified
and requires separate authorization for another allocation.

Fixing serialization does not establish the six required semantic decisions.
The frozen cheap proposals have insufficient eligible-family support for unchecked
complete candidates, checked complete candidates and identified approaches.
The ten-family and 70% apparent-precision rules remain unchanged. No ratings were
obtained, so semantic precision, held-out reader uncertainty and recall remain
unknown; proposal coverage is reported separately.

Native routing varied across executions (mean expert-set turnover 0.1824); this is
reported separately from passing inactive-path checks at the same hidden states.
The 16k deterministic request measures engineering execution and storage. Its
scripted controller timer is zero and cannot price a qualified semantic controller.
Command/capture RPC measurements cover all 14 actual engineering attempts rather
than an isolated 16k request. Scientific component prices remain unavailable.

Real native traces and exact prefix offsets are bound. Historical raw input-row
alignment and offline semantic labels still need qualification before context or
anticipation fitting. Mechanism and arm-blind strict/J1 grading adapters are
implemented, but no new scientific outcomes were graded. Gold enters offline
grading only. The 32 validation and 96 final families remain frozen.

The approved envelope was two 2-GPU allocations and four 2-CPU allocations.
No additional GPU job is submitted. A corrected two-reader pilot proposal and the
existing four-GPU generator/reader fallback are prepared; either needs separate
resource authorization. The fallback also needs exact-contract six-decision
qualification and matched measured prices.

Detailed evidence and DeepSeek adjudication: [REPORT.md](REPORT.md).
The supplied DeepSeek review was incorporated without rerunning it or changing
held-out thresholds; the turnover formula remains valid for eight unique IDs.

- [Final verification](../FINAL_VERIFICATION.json)
- [Observed final accounting](CLOSEOUT.json)
- [Allocated historical ledger binding](../closeout/DENSE_LEDGER_BINDING.json)
- [Engineering measurement limits](../ENGINEERING_MEASUREMENT_SCOPE.json)
- [CPU-verified reader correction](../AUDIT_SERIALIZATION_FIX.json)
- [Corrected reader pilot proposal](../NEXT_PILOT_v3.json)
- [Four-GPU fallback](../FOUR_GPU_FALLBACK.json)

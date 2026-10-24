# Routing study v2

The code this note describes is in the git tag `archive/pre-cleanup-20261023`; the current tree no longer holds it.

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

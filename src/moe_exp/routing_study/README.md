# Qwen3.6 routing study

This package implements a controlled, inference-only study from main `9f16c55`.
It lives in the separate `study/qwen36-routing` worktree. The original
labeling/replay checkout and its 32 GPU-hour authorization are separate.

No real routing intervention, calibration replay, tuning run or final evaluation
has been executed by this implementation task. GPU acceptance, calibrated policies,
population preparation and empirical tradeoff results remain execution gates.

## Commands

Use the existing project environments, with this worktree's `src` on PYTHONPATH:

```text
python scripts/run_routing_study.py --help
python scripts/run_routing_study.py prepare --sources TRACE... --pilot PILOT.json --output STUDY.json
python scripts/run_routing_study.py generate --study STUDY.json --policy native.json --phase calibration --smoke acceptance.json --output REPLAY
python scripts/run_routing_study.py calibrate --study STUDY.json --captures REPLAY/calibration --output CALIBRATION
python scripts/run_routing_study.py register --study STUDY.json --policies POLICY... --phase tuning --output REGISTRATION.json
python scripts/run_routing_study.py generate --study STUDY.json --policy POLICY.json --phase tuning --registration REGISTRATION.json --smoke acceptance.json --output RUN
python scripts/run_routing_study.py compare --study STUDY.json --runs BASELINE RUN... --registration REGISTRATION.json --output COMPARISON
python scripts/run_routing_study.py select --study STUDY.json --report COMPARISON/comparison.json --policies POLICY... --stage static --output STATIC.json
python scripts/run_routing_study.py adapt --calibration CALIBRATION/calibration.json --selection STATIC.json --policies POLICY... --output ADAPTIVE
```

Register adaptive comparisons before generating their results. Use `--pairs FILE`
to prescribe comparisons against the always-active action and each matched timing
control, as well as native. `--existing-controls RUN...` binds the output hashes of
previously completed always-active controls so they can be used without regeneration.
The default registration compares each listed policy with native; it does not
automatically supply all adaptive comparisons.

After comparing the adaptive variants and necessary controls, `select --stage final
--static-selection STATIC.json` requires both gates and their random/position controls
for every shortlisted action. It writes at most three candidates and their necessary
controls. Register the final comparisons with `--phase test`; generation requires
`--selection FINAL.json`. Final outcomes cannot enter either selection stage.

`--resume` validates policy, code, checkpoint, registration, rendered token IDs and
sampling. It retains completed attempts, including caps and parse failures, and
regenerates only unfinished infrastructure attempts. Concurrency is capped at eight;
a final partial wave or a resumed wave can contain fewer requests, and that fact is
recorded in the execution contract. Baseline reuse via `--baseline-from RUN` requires
the same study, phase, prompts, seeds, backend, instrumentation, limits and code.
It carries the failure-cost ledger with it. Legacy uninstrumented baselines are incompatible.

## Frozen population and policy rules

The six benchmarks are MATH-500, AIME24, AIME25, AMC23, Minerva and Olympiad.
Splitting operates on problem groups, with exact NFKC/whitespace duplicate detection
and optional explicit duplicate pairs. Cross-benchmark duplicate groups enter
calibration. Per-benchmark group counts use floor(0.6N), floor(0.2N), and the remainder
for test; actual problem counts and duplicate effects are retained. A seed derived
from `42:benchmark` deterministically shuffles remaining groups. The existing 64 pilot
problems and four engineering fixtures enter calibration. All source attempts are
retained there. Final gold answers are present for scoring but source outcomes are
never used for splitting or holdout policy selection.

Saved rendered prompt IDs are reused directly. Tuning has four attempts per question;
final evaluation restores 32 for AIME/AMC and one elsewhere. Original seeds are kept
where available; additional tuning seeds use a fixed hash of 42, problem identity
and attempt number. Sampling is temperature .6, top-p .95, top-k disabled,
32,768 generated tokens and 49,152 context tokens. Each policy records action,
selector, strength, layer-qualified targets, timing, scope, calibration identities,
checkpoint revision, backend/checkpoint file hashes and study code hashes.

The static catalog has sixteen conditions: native, zero identity, four signed identity
strengths, two SteerMoE directions, three normalized RICE multipliers plus the
unnormalized beta=2 ablation, two mixture temperatures, and two margin strengths.
A condition with insufficient calibration support or absent markers is explicitly
unavailable. It cannot execute. Baseline and temperature controls do not need an
expert-selector calibration.

Identity calibration retains the colleague's problem-balanced frequency-weighted
accuracy lift and at most eight positive targets per layer. The compact estimator
uses the same equation without materializing routing tensors. Margin calibration
calls the colleague's fitter on exact-FP8 per-trace mean probability gaps; runtime
calls the existing identity/margin mathematical implementations.

The [SteerMoE paper](https://arxiv.org/html/2509.09660v2) motivates global layer/expert
ranking and max/min log-softmax replacement with epsilon .01. Here detection is an
outcome-based adaptation using problem-balanced selection rates, with within-question
differences reported separately. Ranked experts require ten activating problems in each
outcome arm. FP16/BF16 rounding is checked; if epsilon disappears, the next representable
value restores strict ordering. This numerical adjustment is recorded in the implementation.

The [RICE paper](https://arxiv.org/html/2505.14681v2) supplies the signed score
nPMI(open) minus nPMI(close) minus nPMI(Alternatively). Counts use actual observed
ASCII marker occurrences, aligned to the token completing the marker, across the
full replay including the prompt. Binary token events and occurrence counts are
recorded separately. This extends marker alignment to multi-token markers; no
closing marker is invented for caps. Selected weights alone are multiplied.
Primary conditions renormalize; the beta=2 ablation preserves increased mixture magnitude.

The [ThinkARM paper](https://arxiv.org/html/2512.19995v3) motivates interpretation
through episodes and transitions. Seven-class pilot results remain offline context;
the runtime contains no judge and assumes neither persistence nor instability is harmful.

## Verified boundary and timing semantics

The supported backend is the inspected vLLM 0.29.0+cu129 V1 runner and modular Marlin MoE path,
with eager execution, TP2, maximum concurrency eight, and no speculative decoding,
prefix caching, asynchronous scheduling, expert parallelism or load balancing.

The adapter calls the actual native selector to obtain IDs and weights. Selection
actions modify logits and invoke that selector again. Weight actions operate on
its existing selected weights and return the original IDs. A second hook verifies
that those exact tensors reach `RoutedExperts.forward_modular`. Every real forward
must hit every one of the forty layers. Monolithic/fused/redistributed paths are
rejected. Shared expert objects and arguments are preserved. Both TP replicas must
agree; diagnostics retain one copy. Installed vLLM files are never edited.

Request frames use native query boundaries and absolute computed-token positions;
all text rotary axes are checked against those positions. Padding is excluded.
Prompt rows remain native. A generated token's routing can affect subsequent token
predictions; prompt preservation also leaves the first generated prediction native.
The observed closing marker disables subsequent intervention. Exact recomputation
reuses prior decisions and does not duplicate statistics. Changed token identities
or changed native numerical routing during recomputation are rejected. New-process
resume replays an unfinished request from its beginning. Cancelled IDs cannot return.

Gates are per layer, using that layer's native shadow routing before its current
intervention. They evaluate at generated-reasoning positions 256, 320, ... from
the preceding 256 positions. Persistent allocation requires lag-16 overlap above
calibration Q75 and selected-mixture JSD below Q25. Unstable selection requires
turnover and weak-boundary fraction above Q75; weak means native logit gap at or
below calibration Q10. A firing gate acts for 64 positions, then cools down for 64.
Insufficient history stays native.

The pilot's bounded `RoutingReducer` mathematics is copied narrowly into
`reducers.py`; its original file hash and the default-only adaptation are in
`reducer_provenance.json`. JSD compares the two halves of a trailing window.
Exact gap histograms retain unique-token quantiles without storing repeated raw
arrays. Random and position-only controls fit nominal duty rates on calibration
traces with a deterministic 257-point probability search. Position profiles use
only calibration position bins. Discrete matching errors are retained.

## Outputs and analysis

Replay writes sealed compact per-attempt calibration summaries with native selection
counts, actual marker opportunities, exact gap histograms, window features and
dispatch diagnostics. Generation writes an atomic manifest and fsynced per-attempt
JSONL. Infrastructure failures have a separate ledger; observed tokens are retained.
If a lost worker prevents exact token-cost accounting, comparison rejects the run
even after coverage is restored.

Comparison averages attempts within problems, then problems within benchmarks,
and reports an equally weighted six-benchmark mean. Caps and missing/unparseable
answers with known gold remain in the denominator. Metrics include accuracy,
generated tokens, truncation, paired reversals, latency, instrumentation time and
realized action rates. Rates use reasoning token-layer opportunities in targeted
layers. Native routing is observed on the intervention trajectory, not inferred
from baseline logits or confused with affinity estimates.

The plotter exports PNG, PDF and SVG. One thousand paired bootstrap draws resample
question groups within benchmarks, preserving duplicate clusters and all attempts.
BH correction spans the registered comparisons and benchmark/overall estimates
within each outcome family. Reported p-values use a centered paired bootstrap.
Observed frontier membership and supported accuracy/token changes are distinct.
No noninferiority margin or equivalence claim is invented. Few AIME test questions
remain a limitation even with many attempts.

Frontier selection includes native in dominance checks. It chooses highest accuracy,
lowest token cost, and the lower median distinct-token frontier point, with stable
policy-ID tie breaks. Dominated candidates, random controls and position controls
cannot become shortlisted interventions.

## Execution gate and resource envelope

Run `scripts/verify_routing_study.py` for small CPU fixtures. A passing report must
cover tensor tests with no skips and bind the current code before GPU smoke acceptance.
The smoke additionally hashes cached weights, verifies native/zero behavior on actual
generation and checks all static actions on short saved reasoning replays.

The separate, unsubmitted smoke script requests two A100 GPUs, 16 CPUs, 120 GiB RAM,
and 30 minutes under boost_qos_dbg: at most **1 GPU-hour**. It also prepares the real
population on the allocated node. Large source scans are rejected on login nodes.
The smoke checks four engineering fixtures, eight requests, at most 128 new tokens
per native/zero generation and 512 saved completion tokens per action replay.
Artifact output is checked against a 2 GiB cap; compilation caches need additional
private staging space. The absolute job walltime remains the compute cap.

A measured full-study resource proposal is emitted only after the smoke passes.
Its token-cap and startup envelope covers calibration, up to sixteen static conditions,
eighteen adaptive variants and fourteen final candidates/controls. It remains
unapproved, labels short-run extrapolation limits, and reserves the pilot's 32 GPU-hours.
Longer jobs must use normal QoS and checkpoint within the live partition walltime.
No production sweep is automatically submitted.


The A100 smoke rejected forced Triton for this checkpoint's block-FP8 scheme.
The revised engine pins Marlin, which uses FP8 weights with BF16 activations
(W8A16); this execution arithmetic is recorded in backend fingerprints. The
checkpoint remains the exact FP8 revision. Both calibration and generation must
use the same path. Marlin GPU dispatch acceptance is still pending.

Set `VLLM_USE_V2_MODEL_RUNNER=0`; the default V2 runner is outside the inspected
capture path and is rejected. Slurm steps receive `TMPDIR` explicitly through
`srun env` because the prior run's Python rendezvous resolved to `/tmp` despite
the shell's staging export. The revised launcher also pins `VLLM_RPC_BASE_PATH`
to a shorter private WORK staging directory to keep Unix socket names within
platform limits. Existing vLLM installations are unchanged.


The Marlin/V1 GPU smoke loaded the model and installed hooks, then exposed vLLM's
internal request-ID randomization before the first captured forward. The launcher
now uses the pinned version's supported `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`
switch. Every wave receives globally unique request IDs, including retries and
native/zero smoke conditions; all are registered before asynchronous engine
submission. The returned internal ID must match exactly or the request is aborted.
No prefix guessing or mixing histories is allowed. The installed assignment method
is exercised directly in a small CPU regression. GPU acceptance remains pending.

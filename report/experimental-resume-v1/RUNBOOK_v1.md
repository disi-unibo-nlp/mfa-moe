# Experimental resume v1

This runbook implements the user's revised objective: causal control of local
reasoning trajectories, with measured accuracy and token effects. Results remain
population-specific. No minimum utility gain or universal sequence is assumed.
The completion deadline is flexible. Do not commit or push without explicit approval.

**2026-10-01 authority update:** The user's later instruction to use all needed GPU
hours and submit the analyses and runs needed for the paper supersedes this
runbook's earlier statements that another qualification attempt or expanded
resource ceiling lacks authorization. See `RESOURCE_AUTHORIZATION_EXPANDED.json`.
Each new submission still requires a concrete complete-stage price, live account
check, scientific and engineering gates, a versioned amendment when a historical
ceiling changes, and exact job/artifact verification. Earlier spending and failed
attempts remain in the ledger; the new authority does not turn them into passes.

**Live checkpoint, 2026-10-01 18:13 CEST:** Ordered-worker engineering passed
in job 59108983. The direct seven-class judge passed its frozen operational
audit in job 59112590 (99.5% parse/stop coverage, 87% historical-label
agreement), but semantic validation remains open. Dense discovery labels for
5,659 sentences are queued as job 59114230 with a measured 14.25 GPU-hour
stage ceiling; CPU class summaries are dependency job 59114313. Clean R3-D
readout and anticipation are running as jobs 59110943 and 59111360. Six
bounded, progress-guarded CPU continuations of anticipation have been queued
behind 59111360; see `R3D_GUARDED_CHAIN_SUBMISSION.json`. See
`PROTOCOL_RESOURCE_AMENDMENT_v0.2.md`, `STUDY_GATES.json`, and `JOB_AUDIT.json`
for current gates and accounting. The old “no further retry” and fixed-budget
language below documents the earlier authorization state.

**Live checkpoint, 2026-10-01 19:10 CEST:** Native discovery routing profiles
for all 48 families and 5,659 sentence windows completed as CPU job 59116916;
family-balanced exposure/displacement summary job 59117172 also completed.
Their saved arrays are observational inputs for later matched expert proposals,
not causal targets. The prefix transition candidate module is implemented and
tested but cannot trigger a study action until semantic qualification passes.
The candidate audit fixture builder is queued as CPU job 59117940 with
`afterok:59114313`; if the first dense-label slice times out, that dependency
will cancel and must be rebound to a successful same-manifest resume. M9 replay
job 59116442 failed before model load because its launcher selected the wrong
frozen tree (0.11167 GPU-hour); corrected job 59117029 uses M9's verified
`9a61e32f` tree and is running. See `M9_REPLAY_QUALIFICATION_LAUNCH.md` and
the direct job registry. Production M9, semantic ratings and causal action
generation remain gated; no paper claim has been upgraded by these submissions.

**Live checkpoint, 2026-10-01 19:36 CEST:** M9's corrected card-sampled replay
`59117029` and deterministic adjudication `59118998` both failed batch output
identity. The deterministic native cases matched; edited GPU cases did not.
`M9_REPLAY_ADJUDICATION_v1.md` preserves both raw verdicts. M9 generation is
`HOLD_ENGINE_ISOLATION`; no M9 resource amendment or production run has been
submitted. The inactive amended launcher and CPU grading driver are prepared
but cannot establish a result. Dense labels `59114230` remain pending priority;
the 48-family native route profiles are complete. Transition audit fixtures
now rate every contiguous pair so false triggers in wrong starting classes
remain visible. Readout `59110943` and anticipation `59111360` are running.

**Live checkpoint, 2026-10-01 20:19 CEST:** The further M9 batch diagnostic
`59120703` completed with `HOLD_ENGINE_ISOLATION`: native greedy cases matched,
but edited requests still showed mixed-batch and repeated-isolation changes.
Its complete receipt is in `M9_REPLAY_ADJUDICATION_v1.md`; M9 production stays
held. The label-free discovery transition frame `59120873` completed with 619
rows (19 candidate fires and 600 sampled nonfires), all from the 48 frozen
families. Corrected CPU price job `59121106` counted actual token IDs; its
sealed complete-stage projection is 11.322 GPU-hours within the amended
13-GPU-hour semantic-measurement ceiling. The first two-A100, five-hour slice
is **submitted** as `59121216`, verified pending priority, with output bound
to frame and driver digests. These same-model ratings are an LLM audit; no
transition or routing action is yet qualified. Dense labels `59114230` and
the separate clean R3-D fits remain queued/running as recorded in the direct
job registry and refreshed `JOB_AUDIT.json`.

The sealed detector-discovery analysis is prepared as
`analyze_transition_ratings.py`. Its two unit checks pass, and its 30-minute,
two-CPU batch request passed live scheduler validation. Job `59121293` is
queued with exact `afterok` dependencies on ratings `59121216` and the dense
label fixture `59117940`. It will compare every rated UID and reader-visible
field with the labeled frame, count coverage and reader agreement, and report
all-fire versus sampled-nonfire confusion descriptively. If a predecessor
times out, this dependency cannot silently become a completed audit; rebind
only after a verified same-manifest recovery. No numerical trigger threshold
or causal action is enabled by the scheduled analysis alone.

Native fixed-window routing motion is separately queued as CPU job `59121394`
(two CPUs, at most two hours; SHA-bound driver). It derives full 64-reasoning-
token gate/selection profiles directly from the saved native top-k tensors,
then reports layerwise total-variation velocity, second-difference magnitude,
signed speed change, and top-eight expert turnover. It omits partial tail
windows and preserves per-family arrays and source tensor hashes. This stage
reads no class label or intervention result; any velocity/acceleration pattern
remains descriptive until tested with actions.

First fixed-window attempt `59121394` failed `1:0` after 29 seconds because
its lightweight CPU environment lacked the already-installed PyTorch needed
by the saved-tensor loader. It produced no family profiles; its output binding
and raw error remain. The launcher alone now uses the existing `vllm-cu129`
environment and Python module. Scientific source SHA `363dbe1b…` is unchanged;
same-bound recovery `59121495` is verified pending priority. Figure work
depends on the recovery's sealed result, not the failed job.
The descriptive routing-motion/expert-exposure figure is queued as CPU job
`59121502` with `afterok:59121495`. It will save PNG, PDF, and a source-hash
receipt only if all 48 native-family profiles complete. A failed recovery
leaves this figure unmade rather than reusing partial arrays.

Observed: `59121495` completed `0:0` in 118 seconds on two CPUs, sealing all
48 families and **12,274** full 64-token windows (1,299 partial-tail tokens
omitted). The failed first attempt consumed 29 seconds on two CPUs. Figure
job `59121502` completed `0:0` in 15 seconds, with sealed, visually inspected
PNG/PDF and native-exposure heatmap. The layer-averaged velocity is 0.416 TV
per 64-token step; this remains observational. The three descriptive scalars'
family-clustered simultaneous intervals are queued as CPU job `59121573`
before any value enters the claim ledger.

Job `59121573` completed `0:0` in ten seconds on one CPU. Its sealed
`MOTION_UNCERTAINTY.json` reports three simultaneous, approximate family-
clustered intervals: velocity **0.416 [0.404, 0.427]**, acceleration magnitude
**0.716 [0.696, 0.735]**, and frequent-expert turnover **0.544 [0.526, 0.563]**.
The source-checked `MOTION_CLAIM_AMENDMENT_v0.1.json` was merged into the
209-entry `CLAIM_LEDGER.json`. These intervals condition on the deterministic
discovery-family selection and do not justify a causal or representative-
population claim. The four-panel observational figure is at
`fixed-window-routes-84a85c92-363dbe1b/ROUTING_MOTION.pdf` and `.png`.

The new direct-LLM seven-class figure job `59121759` is queued behind the
complete dense class summary `59114313`. If the label first slice times out,
that dependency must be rebound after a successful same-manifest recovery.
The figure shows only contiguous class transitions, observed boundary-censored
dwell, re-entry loops and class counts; it cannot stand in for the separate
blind content audit or a causal trajectory endpoint.

**Live checkpoint, 2026-10-02 09:23 CEST:** Dense labels, class dynamics and
figure, two-reader semantic ratings, and the detector discovery audit completed
overnight. The current detector has poor start coverage and insufficient
reader-supported immediate transitions; no causal routing action has been
launched. Clean seven-model R3-D readout completed, while anticipation remains
running with guarded dependent continuations. See
`LIVE_CHECKPOINT_2026-10-02.md` for exact job IDs, audit counts, limitations,
and next qualification gate. Earlier dated queue states above are historical.

## Ownership and authority

Run natively as `lmolfett` on LEONARDO; verify `hostname` and `id -un`. Read the
installed `cineca-operations` skill and `references/on-cluster.md` before cluster
operations. Read its scheduler/storage/software references when relevant. Preserve
the 600-second login CPU limit and execute production preprocessing, fits, plots
and inference through Slurm. Staging and outputs belong under the user's HOME or
`/leonardo_work/IscrC_MIOSR/lmolfett/`; use `tools/tmp/` for temporary files.

The previous keeper acknowledged `DRAIN=yes` with zero executors. Do not remove
`swarm/DRAIN`, wake that keeper, change `code/latest.json`, or act on its stale
dashboard. Reconcile exact job IDs and receipts before any new submission.
`swarm/controller/STATE.md` is historical evidence, not the current authorization.

The user authorized the legacy workloads at their stated ceilings and explicitly
approved **+11.25 GPU-hours**, comprising **0.75 additional H1–H4 recovery** and
**10.50 for the separate routing-action study**. The receipt is
`steering-v1/runs/resume-v1/RESOURCE_AUTHORIZATION.json`. Its study ceilings are
qualification .75, discovery 2.00, mechanism 2.25, utility 4.75, reserve .75.
The later one-attempt approval added .15 to qualification only, raising it to .90
and the separate study total to 10.65. That attempt failed; no further retry is
authorized. `RESOURCE_CHECKPOINT_v2.md` contains the current expenditure.
Do not transfer compute between these stages or treat unused legacy allocations
as permission for extra study experiments. All cold loads, preparation, prefill,
decode, NLL, labeling, failures and teardown count.

## Immutable inputs and completed engineering

Run root `R` is
`/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24`;
`S` is `R/steering-v1`. Repository scripts are in `scripts/experimental_resume/`.
The authoritative preregistration is `S/PREREG_steering_v1.v0.3.md`, SHA256
`4483782220d3e556e6a33c4596679f8292fee3a9beb4e6d6a6309f0e899a8932`.
Its historical appendix remains preserved verbatim. Do not rerun the broad
`protocol_package.py` or `freeze_addenda.py` to update later checkpoints.

The capped runtime is `S/code/s1-9a61e32f48c04c24`, tree
`9a61e32f48c04c242acccc89c529bd750776c553cdfc776347151d359bc53430`.
Exactly two hunks in `moe_steer/manifests.py` changed; 209 other files and the
model source copy remain identical. Old manifests still validate, explicit caps
reach sampling, invalid caps fail, and natural and length stops remain distinct.
See `CAP_FIX.md`, capped dry manifest, tree inventory and regression receipts.

The frozen duplicate-family graph is `family-freeze.json`, seal
`592a71b1fa2223a29168ee29afae73de406b3a4d4408d38b69080bee152143f6`:
1,547 questions, 1,486 families, 43 families connected across confirm. Keep the
original split. Exact-ID-clean results and exclusion of confirm-connected
development families are different analyses. The 48/128/96 discovery/mechanism/
utility parent pools are disjoint and deterministic; counts are capacity, not
behavioral eligibility.

H1–H4 recovery **59092583** completed, exit **0:0**, all four checks passed on
56 requests including five preemptions. Its artifact is
`S/qualification/h14/resume-v1-recovery/H14.json`. It cost .46556 GPU-hour.
First attempt **59090613** failed at its deadline, cost .48056; preserve its raw
outputs separately. Combined cost .94611 is within the original .5 plus added .75.
Raw Q3 routing failure and the supervisor's adjudication remain distinct.

## Saved analyses

`JOB_AUDIT.json` records every resume submission's parent and step state, exit code,
allocated elapsed cost, and artifact verification. A TIMEOUT parent with exit 0:0
does not count as completion. Refresh with `job_audit.py` before pricing or reporting.

| Stage | Verified job(s) | Artifact/status |
|---|---|---|
| Clean dynamics rk1c | 59090635 | `R/reasoning-kinematics/rk1c/results.json`, seven models |
| R3-B composition/measurement | 59091848 | `R/forum/tests/r3_dynamics_validity/estimates.json`, complete |
| R3-D readout | 59093360 plus three preparation attempts | TIMEOUT; completed cell checkpoints retained |
| R3-D anticipation | 59094698, 59093809 | TIMEOUT after initial setup failure; complete fold checkpoints retained |
| DP-S3 | 59094755 | `R/depth-paths/results/resume-v1/results.json`, verified saved caches |
| M9 preparation | 59096336 | 400 assigned, 239 closure branches, 161 retained native |
| M10/M8 preparation | 59099106 | 239 matched M10 prefixes and 100 fixed M8 repeat prefixes |
| First paper snapshot | 59099383 | `PAPER_SNAPSHOT.json`; immutable figures/tables/input inventory |

R3-D spent **15.90111 CPU core-hours** across all attempts against its **16**-hour
line. Readout alone used 8.09556, exceeding the internal eight-hour reservation;
the governing combined 16-hour ceiling was preserved by shortening anticipation.
The full registered families, B1 clean refit and confirm-connected-family
sensitivity are incomplete. Do not reduce noise draws, folds, repeats, models or
windows to turn partial cells into a complete result. Do not relaunch a full fit
from the remaining .09889 hour. R3-E remains dependent on complete clean inputs,
B4 and prospective conditional-Bayes-gain power calibration; it has no result.

DP-S3 reuses verified NMI-gap caches for all seven models and available readout
caches. Historical correctness precision remains `NOT_EXCLUDED`. Existing
shuffle/proxy power does not replace a training-fold calibrated accuracy-relevant
conditional Bayes gain. Measurement disagreement from six non-test human
documents does not validate all seven models' semantic labels.

## X2 completion and NLL recovery

Generation **59093506** completed, exit **0:0**, with **6,630/6,630** requests,
no missing UIDs and enforced 1,024-token continuations. Allocation cost was
**3.68278 GPU-hours**. Manifest and digest-bound output paths are recorded in
`S/runs/x2-resume-v1/FROZEN.json`; verify `shard-0.status.json`, both telemetry
ranks, and routed/result offsets. The old `shard-0.complete.json` filename is not
the generation guard's completion receipt.

Native measurement attempt **59095712** passed the strict identical-Q3-fixture
logprob checks at batch sizes one and eight. It then raised `NameError: RS` before
measuring any X2 UID and cost **.48111 GPU-hour**. Raw native arrays and
`attempts/1/PARITY_VALIDATION.json` are preserved under the old measurement tree.
This pass qualifies that measurement load only; it is not a completed NLL result.
The cancelled incompatible UID attempt **59095026** allocated zero GPU time.

The tested correction passes the results module explicitly and uses CPU-prepared
prefixes. CPU job **59101418** completed, exit **0:0**, producing 39 unchanged
prefixes and exactly **18,556,822** teacher-forced positions for 3,354 N/E requests.
Artifact `S/runs/x2-resume-v1/measurement-prefixes/PREFIXES.json` has file SHA256
`09e8f1a900698311958f706c6512606f25635fbecbc846aa2f74ed5144343701`.

`NLL_RECOVERY_AMENDMENT.json` preserves the prior code and binds the new native
measurement tree `ec9661af160f42d145a09756925d2bc26a3f36360a25be772a41730a0e03c515`.
It releases unused generation reservation within X2's existing eight-hour total:
4.16389 actual generation/measurement + 1.5 pessimistic complete recovery + .5
contingency = **6.16389 GPU-hours committed**. It changes no sampler or parity
criterion. The 41-minute TP2 limit plus observed 196-second shutdown allowance is
1.47556 GPU-hours. Recovery job **59101870** completed, exit **0:0**, in 1,734
seconds on two GPUs (**.96333 GPU-hour**). It measured **3,354/3,354** N/E native
surprisal sequences and passed both exact Q3 parity fixtures at batch sizes one
and eight. The `native-nll.json` output is bound to the source tree and manifest.
Combined X2 generation, failed attempt and recovery spent **5.12722 GPU-hours**.

Saved-results CPU analysis **59102231** completed, exit **0:0**, using .05667 CPU
core-hour. It retains all assigned branches, measures actual TV on each policy's
own layers, prints D×layers/30, matches each random ladder by log dose, and reports
all 42 cells. Primary marker counts use **every new matching sentence**, as
registered by R1-A4; the earlier helper's onset-only count is a sensitivity.
The triggering sentence and decisions outside [0,512) are excluded. Four KL
windows compare E/M to the paired N, with short-window coverage explicit.
Simultaneous family t approximations cover all 126 E−N diagnostic contrasts plus
all 438 possible random-ladder marker contrasts; nominal question bootstrap
intervals remain separately reported. Constant/boundary intervals do not establish
retention. The original saved analysis used a wider random-dose eligibility band
for non-force arms than preregistration v0.3 allowed. The versioned
`G3_DOSE_SUPPORT_AMENDMENT.json` applies the required ±10% band to each random
set: **2/42** cells pass, versus **6/42** in the original output. The selected
suppression and promotion cells remain `reweight_m1_L1_landmark` and
`reweight_p2_BAND_landmark`. Off-band E−M estimates are unmatched diagnostics.
G3 selects a dose; neither selected marker interval excludes zero under the
reported simultaneous approximation, and no semantic-control result exists.

Do not launch X3 while native NLL or complete G3 is absent, its own full cost
projection is incomplete, or the new study's priority gate is unresolved. The
prepared X3 builder preserves the first-64 hash
`68ffb11fb9a7e60ca7b7ec2cde2a5632829372a436efd7d3ee6ac6f37089bd82` and 32k endpoint;
the new utility scout's 16k endpoint is separate.

## New routing study: gates and recovery frontier

The prefix-only parser, future-field allowlist, bounded complete-expression
adapter, sparse action constraints, dense contiguous-window summaries, receipt
recovery and ordered worker adapter are implemented and CPU tested. The worker
has device dose and inactive-routing checks, but has **not executed its GPU
qualification batches**. A parser edge-case test is not a semantic detector.
No original-prompt adaptive controller, action dictionary or trajectory has been
qualified, fitted or validated. No discovery or utility result exists.

Ordered qualification **59096925** failed on a missing module before loading a
model (.00722 GPU-hour). **59098926** failed at its conservative deadline gate
after prefix preparation, also before model loading (.13500). Both are preserved.
The final one-attempt recovery is described below; all three spent .27222 of the
revised .90 GPU-hour qualification ceiling.

CPU fixture preparation **59102494** uses the identical fixed four-family
selection, branches at 2,048 tokens, no replacements and no future fields in
execution inputs. The failed recovery preserved its bound inputs and driver.
A further attempt needs a new complete price and explicit approval. Preserve
prior immutable overlays and failure receipts; keep the worker acceptance checks.

`STUDY_GATES.json` records the deterministic parent pools, missing behavioral
eligibility, unqualified semantic detector and complete-price holds. The maximum
utility scout's historical pessimistic decode/load scenario is **6.07304**
GPU-hours before missing prefill, grading, detector and retries, above its **4.75**
stage ceiling. It is a scenario, not a qualified controller price. Price complete
validation first; neither a promising pilot nor a borderline p-value authorizes
additional samples, candidates, or resource transfers.

The user subsequently authorized **+0.15 GPU-hour for this one recovery**, raising
the qualification stage to .90 and the separate study total to 10.65. Receipt:
`S/runs/resume-v1/QUALIFICATION_RECOVERY_AUTHORIZATION.json`. GPU job **59103206**
ended FAILED, exit **3:0**, after 234 seconds, cost **.13000 GPU-hour**. Its
CPU-prepared fixture seal was validated, but setup still consumed 219.35 driver
seconds; 880 seconds remained against the unchanged 950-second load/test guard.
No model load or qualification batch occurred. Trace preparation was not the
only setup cost; this recovery proposal underestimated the remaining setup work.
All three attempts cost **.27222 GPU-hour**, leaving .62778 on the stage, but
**no further retry is authorized**. Preserve all receipts; do not weaken the guard.
The failed proposal is retained as evidence rather than repurposed for another run.

The conditional Bayes-gain generator in `routing_control/calibration.py` is CPU
tested at the five registered gains. It uses an independent balanced signal and
preserves each training context's baseline marginal probability, so its expected
oracle gain is conditional mutual information in nats/attempt. This is a specified
generator, not a completed 200-simulation pipeline power result. R3-E still needs
the clean nested B1 inputs and full refits for every simulation; do not infer
equivalence or empirical precision from these helper tests.

## Secondary closure and probes

M9 manifest `S/runs/m9-resume-v1/MANIFEST.json` seal
`63772b263e855692ecc5a37b10f634f96b30fa51e7e54bf22f5026f16e43024b`
retains all 400 assignments, cut 14,336 and total budget 16,384, with the literal
injected `</think>\n\n` and its actual token cost. Maximum decoding is 488,994
tokens and prefill 3,462,375. Native 16k, fixed integer-emission-at-cut and full
native reference are distinct comparisons. The legacy integer parser limitation
is disclosed. Matched-prefix presence, injected closure, batch isolation and
full generation/replay/grading price must pass before GPU inference.

M10/M8 manifest `S/runs/m10-resume-v1/PROBE_MANIFEST.json` seal
`0be5750567bc7df12f8024964ef09b3abdb7ca40bbe4b3a1279a7d14df24876e`
has no PAC labels yet. M9/M10 share frozen family folds. Replay uses layers
19/29/39 (zero-indexed), train-fold PCA8 primary/PCA64 secondary and the fixed
strengthened past-text baseline. M8 uses exactly the first 100 dev seed-zero
prefixes, at most 8,192 tokens, across two launches; its ICC(A,1) implementation
does not substitute membership agreement for scalar/probe-score repeat reliability.
Hidden capture, matched replay qualification and probe fits remain unexecuted.

M9's two GPU-hours and M10/M8's three include their own replay/loads/retries.
Shared J1 reserve expenditure already includes .86222 GPU-hour, leaving 1.13778
for required blind grading and other shared charges. Unknown full grading price
holds launch. Do not treat the separate authorized 10.5 study allocation as that reserve.

## Verification and paper updates

Use the existing Python environments. No login-node torch/model imports, new
environments, runtime edits, daemons, or relaxed TLS/SSH/sandbox settings are
needed. The sanctioned command escalation handles this host's expected bwrap
namespace failure. `launch_legacy.py` is one-shot: an existing submission receipt
prevents silent resubmission. A changed code/cap/manifest needs a new digest-bound
directory and engineering receipt; do not reuse completed UIDs across bindings.

Run meaningful CPU tests for changed code with private TMPDIR, `python -B`,
`--import-mode=importlib`, and disabled pytest cache. Source imports must select
the intended frozen `moe_steer` and any separately frozen worker overlay.
Verify each real submission by exact job ID, owner/account, final `sacct` parent
and step state, exit code, and input-bound output artifacts. Record failed
allocations and scheduler shutdown time in the ledger.

The final saved-result paper job **59106067** completed, exit **0:0**, in 17
seconds on one CPU. Its snapshot path is in `PAPER_SNAPSHOT.json`. All 21 hashed
inputs were copied under that snapshot's `input-archives/` before the live audit
refreshed. The supplemental routing-profile job **59106824** completed, exit
**0:0**, in six seconds on one CPU; its path is in
`ROUTING_PROFILE_SUPPLEMENT.json`. It plots native event-locked velocity,
acceleration and rarefied expert use for the no-confirm GPT and Qwen3.6
populations. Neither plot establishes a causal acceleration pattern. The latest
`FINAL_STATUS.json` reconciles both jobs, all 26 submissions and actual cost;
the primary compute chart explicitly predates its own final accounting.

The tested `QUALIFICATION_RECOVERY_PROPOSAL_v2.json` is a **proposal only**. It
binds the same worker, fixed four-family prefixes and all engineering checks to
an immutable driver that starts the unchanged 950-second guard before runtime
fingerprinting. Its 20-minute two-GPU allocation plus observed 196-second
shutdown allowance reserves .77556 GPU-hour. Together with .27222 spent, the
stage maximum would be 1.04778 GPU-hours against a proposed 1.10 ceiling,
requiring **new explicit +.20 GPU-hour authorization**. Slurm test-only passed;
no v2 GPU submission has been made under the exhausted one-attempt approval.

Any later report revision must preserve prior snapshots and exact input hashes.
Keep missing cells explicit. Report historical confirm exposure, K1's narrow
matched population, K2's inconclusiveness, G2's accepted non-rejection, and G3's
diagnostic role. A transition-associated routing profile or faster routing alone
does not establish useful semantic control. Do not manufacture absent trajectory,
accuracy–token, legacy X3 or secondary-probe figures.

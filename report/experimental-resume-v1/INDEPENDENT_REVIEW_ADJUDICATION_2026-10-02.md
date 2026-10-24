# Independent DeepSeek review adjudication, 2026-10-02

Two separate DeepSeek Flash 4.1 max-reasoning sessions completed source-only
audits: [A](DEEPSEEK_INDEPENDENT_REVIEW_2026-10-02_A.md) and
[D](DEEPSEEK_INDEPENDENT_REVIEW_2026-10-02_D.md). Both were asked to audit the
objective and raw code/protocol packet without seeing another review. The other
two calls timed out without findings. Follow-up debates
[A](DEEPSEEK_REVIEW_DEBATE_2026-10-02_A.md),
[D](DEEPSEEK_REVIEW_DEBATE_2026-10-02_D.md), and
[D2](DEEPSEEK_REVIEW_DEBATE_2026-10-02_D2.md) challenged the first conclusions
with direct source and audit facts. Reviewers used no credentials or cluster
tools; their reports are advisory, not experimental evidence.

The central finding is a scope boundary. The 13-family 256-token screen is an
exploratory, discovery-internal test of an immediate semantic transition plus
actual routing dose. It cannot establish an ordered two-pulse 1,024-token
trajectory, answer accuracy, or 16k utility. A positive screen would be a local
pilot result, not independent validation. The registered mechanism population
is 128 separate families; the utility population is 96 further families from
original prompts. No new causal outcome has been observed yet.

The review identified a real design risk in the held positive-v2 pilot: the six
arms always occupied the same serial run positions. Its random controls also
balanced sets only across pooled seeds, while the registered control rule calls
for per-seed balance. That job (59204242) was cancelled while held, at zero GPU
time. Positive-v3 freezes counterbalanced arm positions and per-transition,
per-seed random sets, validates exact replay, and uses the same 13 starts and
156 requests. It passed CPU preflight, focused tests, source-hash checks, and
Slurm test-only; job 59216285 is the canonical queued run. A separate negative
bias/force-off diagnostic, deactivation-v4, shares the starts and control
schedule, has a distinct necessity estimand, passed the same gates, and is job
59216601. The independently submitted positive-v4 job 59216368 was an exact
scientific duplicate of positive-v3 and was cancelled pending at zero GPU time.
See [active-job coordination](ACTIVE_JOB_COORDINATION_2026-10-02.md).

The serial qualification audit shows native duplicate variability: 1/8 equal
texts and 0/8 equal unordered first-token top-k expert sets, despite 8/8 equal
first-token target masks. This rules out exact hidden-state pairing or engine
equivalence. It does not by itself rule out an intention-to-treat comparison
of assigned policies with family-clustered uncertainty. The batched neighbor
audit remains insufficient for a later batched causal claim; the current jobs
force serial/eager execution.

Several initially ranked "bugs" were conditional or mistaken. The preliminary
deactivation rating price is explicitly non-executable before generation;
arm-blind rating requires a later frame-bound, 1,024-token complete-stage price.
The v2 driver reads its random schedule directly from the manifest; it does not
call `design.balanced_random_assignment`. DeepSeek D accepted this correction
and a second correction that utility must use disjoint families and original
prompts. The current v3/v4 pilot manifests have the required 8 candidate and
5 approach families and four distinct matched random sets. The old generic
validator concerns do not erase these observed sealed assignments.

Actual later-stage issues remain: the online live gate used the older detector
and a malformed display-math close; a new v2.4 detector and v2 gate have focused
CPU tests but have not been GPU-qualified. The 1,024-token trajectory requires
dense contiguous seven-class sentence labels, token/routing alignment, two
arm-blind substantive votes, closure-aware endpoints and frozen comparison
pairs. The versioned CPU analyzer now enforces that input contract and passes
eight tests; no corresponding generated/graded continuation exists yet.
`dense_window` currently rejects a missing first post-trigger sentence and
excludes a sentence crossing the horizon, so that source bug is repaired for
future analysis. Accuracy extraction, emitted-plus-injected token accounting,
and complete 16k utility pricing remain future execution gates.

Interpretation is fixed before unblinding: native duplicates quantify serial
noise, negative bias and force-off are necessity diagnostics, and a routing-dose
change alone is not semantic control. Unscored and failed assigned cells remain
in intention-to-treat receipts. The next evidence path is to complete and rate
the queued 256-token pilots, freeze one local template, then run a separate
1,024-token ordered/reversed/random comparison and the disjoint 16k utility
scout only with complete pricing and measurement qualification.

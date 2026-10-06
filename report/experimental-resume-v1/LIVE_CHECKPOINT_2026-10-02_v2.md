# Live checkpoint — 2026-10-02 13:29 CEST

Verified `login05.leonardo.local`, user `lmolfett`. `JOB_AUDIT.json` was
refreshed at 11:22:01 UTC and reconciles 121 submitted jobs, including failed
and canceled attempts. The user authorized necessary CPU/GPU work without a
fixed hour cap; scientific eligibility, measurement and complete stage-price
gates still apply. This checkpoint supersedes the morning operational status,
without altering sealed protocols or results.

## Full-prefix measurement and intervention starts

Full prefix means the original question and every generated token through the
candidate intervention boundary. The online input excludes future generated
text, correctness, reference answers and future labels. Seeing all prior
reasoning matters: an apparent candidate may have been checked earlier, or a
number may be an intermediate calculation rather than a proposed answer.
Prefix token IDs are preserved; late-prefix work must use a qualified larger
context rather than silently truncate history. Cost audit `59199073` found
1,549 detector fires through the 16,384-token utility horizon. Calling a
full-prefix side judge once at every such fire would require 6,572,311 input
tokens and up to 99,136 response tokens across the 48 discovery traces.
This is a cost projection, not an executed controller. A cheaper discovery-only
expression-state/deduplication screen is being prepared and must report what
eligible starts it misses before its rule can be frozen.

The Qwen full-prefix start audit `59186342` completed `0:0` in 37m27s:
372 windows, 744 assignments, 686 strict parsed natural stops, and 327
windows covered by both readers. Its native-model comparison `59197280`
also completed `0:0`, with 744/744 strict parsed natural-stop ratings.
`FULL_PREFIX_NATIVE_VETO_AGREEMENT_v1.json` seals the comparison:

| Fixed hypothesis | Detector-fire windows | Both Qwen readers accept start | Both native readers accept start | Accepted by both models |
|---|---:|---:|---:|---:|
| Candidate to substantive verification | 142 | 25 | 25 | 8 |
| Identified approach to commitment | 35 | 12 | 9 | 6 |
| Failed check to revision | 3 | 2 | 0 | 0 |

These are stratified discovery LLM measurements, not human ground truth,
population precision estimates, or intervention outcomes. Parse failures and
reader disagreements remain explicit. Exact-prefix pool preparation `59199245`
completed `0:0`: it selected 12 candidate, 10 approach and one failed-check
start, with two families overlapping the candidate and approach subsets.
It is not yet the globally disjoint, independently audited causal enrollment.
An independent GPT-OSS audit is being prepared. Failed-check support is
insufficient for a causal action at present. New eligible starts must remain
within discovery families; no validation family is used to repair eligibility.

## Qualification and GPU work

- Serial/eager score calibration `59198795` completed all 80 frozen requests
  and failed the unchanged acceptance rule. Cross-API B mean/p99 differences
  were 0.000567/0.007644 nats, above limits 0.000445/0.005950. The 16 sampled
  tokens agreed across all API/repeat cases. Serial execution did not resolve
  the numerical measurement problem. Expert-loss ranking remains unqualified;
  no threshold is relaxed and no additional score-profile retry is planned.
  A separate pulse/replay qualification can still support behavior-only causal
  discovery from native contrast proposals, without this added ranking metric.
- Earlier batched calibration `59196354` failed its cross-API mean and p99
  thresholds despite completing 80/80 requests. Its results remain saved.
- Batch-invariant backend qualification `59197533` failed before inference:
  installed vLLM rejects batch invariance for the model's GDN attention.
  Its dependent audit `59198296` was canceled at zero elapsed time. The
  separately sealed failure receipt is `CAUSAL_BATCH_INVARIANT_QUAL_FAILURE_v1.json`.
- X3 timing recovery `59194916` completed `0:0`, 40/40 requests, zero errors,
  971,219 decoded tokens and 1.2044 allocated GPU-hours. Measured decoding
  throughput was 838.61 tokens/s; engine initialization and other overhead
  are charged separately. Generated content remains embargoed in this
  timing-only pilot. Full X3 still requires a complete generation, grading,
  labeling and native-NLL price before launch.

The earlier four-family routing pilot established a first-stage change in
executed experts, but had zero jointly eligible semantic starts. It does not
estimate verification control. Native replay variability also remains a
qualification issue; it is not explained solely by neighboring edited requests.

## CPU analyses and paper artifacts

Canonical anticipation `59111360` runs on four CPUs through approximately
17:38 CEST. The independent GPT destination shard `59180550` runs on 16 CPUs;
its current projected completion is around 18:15 CEST, conditional on its
observed fit rate. Qwen shard fits are sealed and verified; their guarded merge
`59188564` waits for the canonical job. Continuations remain guarded against
duplicate or incompatible work.

Clean B4 feature generation `59197542` and final nested-CV analysis `59198802`
completed `0:0`. The latter assigned 509 questions across 496 families and
scored 508 questions across 495 families. The registered add-on gain is
-0.000729 nats per scored question, with
family-clustered 95% interval [-0.005313, +0.004384] and simultaneous
five-comparison interval [-0.006747, +0.005289]. This within-B4 interval does
not replace the separate registered P-A1/P-B1/P-B4 Holm adjustment. It does not
support predictive improvement. Confirm-connected-family exclusion is being
recomputed with its own training folds (`59199203` smoke and full feature
job `59199796` completed `0:0`; separate sensitivity CV is being prepared).
Any subsequent model remedies are
separate exploratory analyses and cannot overwrite this result.

Clean B1's positive association, 0.008385 nats per token-pair
[0.008044, 0.008737], pertains to the GPT source dataset, not Qwen causal
steering. Predictive associations and routing motion are kept separate from
semantic control and utility endpoints.

Saved-result figure recovery `59197523` completed `0:0`; PDFs, PNGs, source CSV
and a claim ledger are bound by `DISCOVERY_EVIDENCE_SNAPSHOT_v2.json`. The
failed first figure job `59197144` is retained. `METRIC_PROVENANCE_v0.1.md`
records literature-inspired metrics and study-specific definitions, including
the distinct native-window and per-token motion estimands.

The later measurement figure job `59200275` completed `0:0` in 14 seconds,
with zero GPUs. `MEASUREMENT_EVIDENCE_SNAPSHOT_v3.json` binds ten verified
files: full-prefix rating counts, clean B4 family intervals, both preserved
score-qualification failures, source CSVs and a separate claim ledger.

At 13:39 CEST, serial/eager pulse/replay qualification `59200002` is running
on two A100s; independent GPT-OSS parity `59199929` completed `0:0`. GPT-OSS
produced final payloads inside a detokenized analysis/final envelope that the
original JSON-only parser rejected. The immutable raw result stays intact;
a strictly anchored parser adaptation and independent expanded audit are
being prepared. Preliminary final-payload disagreement is a measurement
finding, not a semantic truth label. Fixed post-primary CPU remedy screen
`59200183` completed `0:0`; any result uses the same cohort and remains
exploratory.

All-results ETA remains a conditional planning estimate, not a completed
scientific package. Causal-stage duration depends on verified eligible-family
counts and the scoring/execution qualification results.

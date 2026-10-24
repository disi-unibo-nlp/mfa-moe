# Evidence ledger

**Current campaign audit, 2026-10-09 (ADR18):** v2 has closed with
`REPORT_COMPLETE=true`, `SCIENCE_COMPLETE=false`; no scientific jobs remain
queued or running. Seven summaries failed on a shared non-finite JSON error.
Seven full-A mappings and exact-attempt native-accuracy outputs exist, and six
single-sequence GPU pilots passed instrumentation checks. REAP reconstruction
passes only Qwen3.5/Qwen3.6 among those pilots; four models retain an explicit
fidelity limit and Qwen3-30B was skipped upstream. Full saved-sequence measurement,
compact qualification and new causal/accuracy–memory effects remain outstanding.
These are engineering/status findings, not new scientific effects.
[Audit and remaining decision requirements](NEXT_EXPERIMENTS.md#campaign-v2-audit-and-final-decision-requirements--2026-10-09),
[bound evidence](../../report/model-analysis-review-20261009/DECISION_AUDIT-r1.json).
The October8 stop statements below describe the older campaigns; ADR15/16
authorized only the separately versioned revised study.

**Paper result selection, 2026-10-09 (ADR17):** the main text now leads with
the seven-model CLEAN readout and the completed family-held-out expert
selection comparison. Three reproducible vector figures retain saved
uncertainty and all seven selection outcomes. State balancing improves
minimum held-out class coverage in five models; labelled frequency wins
in Qwen3.5 and Qwen3.6. Fold spread is descriptive. Two-model anticipation
keeps its measured scope; exploratory correctness associations and
in-sample concentration are preserved in the appendix. No pruning-accuracy
or resident-memory benefit is established. [Figure data and method](../../paper/6a510c6db0d9b08aa438c142/generated/best_results_20261009/README.md).

**All experiments stopped by the user (ADR12).** [Stop receipt](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json). Earlier continuation/progress statements retain their observation cutoff and confer no execution authority. Use the [experiment inventory](EXPERIMENT_INVENTORY.md) for direct data/result/code locators.

**Latest catalogue reconciliation:** six discovery rosters have48 canonical receipts and Qwen3.5 has42; availability is distinguished from independently validated closeout. Qwen3.6 has a completed saved recognition **FAIL** (eligible28.57% across28 apparent families; already-checked8.33% across12), joining the four failures recorded below. Its256-entry bound ledger supersedes the earlier reader-wait summary. GLM recognition, full Qwen3.5 audit and v3 causal/utility outcomes remain unmeasured. [D08](EXPERIMENT_INVENTORY.md#d08-v3-discovery-and-candidate-specific-judgments), [X10](EXPERIMENT_INVENTORY.md#x10-v2v3-qualification-failed-gates-and-interrupted-work).

Completed all-seven family-held-out selection/stability and common transitions are indexed in [X11](EXPERIMENT_INVENTORY.md#x11-family-held-out-expert-selection-and-stability) and [X06](EXPERIMENT_INVENTORY.md#x06-state-transitions-and-native-boundaries). They supersede earlier unmeasured saved-data-fit notes; resident-memory/accuracy remains unmeasured ([X12](EXPERIMENT_INVENTORY.md#x12-checkpoint-byte-accounting-and-resident-memory-boundary)).

This ledger separates descriptive association, held-out prediction, realized routing engagement, substantive semantic effects, utility, and engineering. Intervals retain their original comparison family and weighting. “Uncertain” does not mean equivalent or ineffective. [Sources](SOURCES.md) identifies archived table-backed evidence versus inspected result artifacts.

**Current v3 grading limit, 2026-10-08:** the completed shared equivalence audit matches97/100 frozen agent-consensus labels, with no false equivalences on clear-wrong controls, but fails the original maximum of two errors per stratum: all three mismatches are Minerva cases. All nine associated draws stopped and parsed normally. This limits answer adjudication across all seven models; it does not measure their causal effects. The existing strict-grading and unknown-bound fallback remains active, while separate semantic/control qualification proceeds. [Full audit closeout](../../report/verification-study-v3/J1_QUALIFICATION_CLOSEOUT-r1.json), [review](debate/sol-j1-qualification-review.md), [execution](IMPLEMENTATION.md#completed-common-audit-and-automatic-reader-advancement--2026-10-08).

**Current v3 recognition limit, 2026-10-08:** GPT's complete256-draw audit fails the frozen eligibility/already-checked gate. Apparent family-weighted confirmation precision is22.41% across29 families and18.75% across8 families, respectively, against70% and10 families for each. Only84/128 prefixes have two usable judgments; truncated, invalid and disagreeing outputs remain unconfirmed. This is a qualification limit for the current prefix controller, before nomination or treatment, rather than a measured absence of causal control. GPT remains visible; subsequent full-audit outcomes are recorded below. [Full closeout](../../report/verification-study-v3/GPT_ELIGIBILITY_CLOSEOUT-r1.json), [source-grounded review](debate/sol-gpt-recognition-review.md)

**Second v3 recognition limit, 2026-10-08:** Nemotron's complete256-draw audit fails precision for both decisions:25% eligibility confirmation across34 apparent families and10% already-checked confirmation across10 families. Both support minima pass. The audit has191 usable stopped draws,37 truncated and28 content-invalid draws, with81 usable pairs. Sixteen apparent eligibility proposals and five already-checked proposals receive agreeing rejections, so the limit is not explained solely by capped or malformed outputs. Parent and Sol reproduced the original ledger and qualification exactly and found no definite operational defect. Nemotron's causal/utility cells remain unmeasured alongside GPT's existing limit; the later Qwen3-30B outcome is recorded below. [Full closeout](../../report/verification-study-v3/NEMOTRON_ELIGIBILITY_CLOSEOUT-r1.json), [review](debate/sol-nemotron-recognition-review.md)

**Third v3 recognition limit, 2026-10-08:** Qwen3-30B's full256-draw audit fails precision for both decisions:10.10% eligibility confirmation across33 apparent families and 33.33% already-checked across12. Both support minima pass. There are193 usable stopped draws,39 truncated and24 content-invalid draws;80 prefixes have usable pairs. Agreeing stopped readers reject22 apparent eligibility proposals and five already-checked proposals, so capped or malformed outputs alone do not explain the limit. Parent and Sol reproduce the original ledger and qualification exactly and find no definite operational defect. Qwen3-30B stops before nomination/treatment with causal/utility outcomes unmeasured. Gemma's subsequent outcome is recorded below. [Full closeout](../../report/verification-study-v3/QWEN330B_ELIGIBILITY_CLOSEOUT-r1.json), [review](debate/sol-qwen330b-recognition-review.md)

**Fourth v3 recognition limit, 2026-10-08:** Gemma's complete256-draw audit fails both precision gates:25.52% eligibility confirmation across32 apparent families and41.67% already-checked across12. Both support minima pass. It provides209 usable stopped draws,30 truncated and17 content-invalid draws;91 prefixes have usable pairs. Agreeing stopped readers reject14 apparent eligibility proposals and two already-checked proposals. Parent and Sol reproduce the original ledger and qualification exactly and find no definite operational defect. Gemma stops before nomination/treatment with causal/utility outcomes unmeasured. Qwen3.5, Qwen3.6 and GLM continue under unchanged gates. [Full closeout](../../report/verification-study-v3/GEMMA_ELIGIBILITY_CLOSEOUT-r1.json), [review](debate/sol-gemma-recognition-review.md)

## E01 Annotation and hidden-state observability

The repaired response-grouped probes use 38 external DeepSeek-R1 SAT responses and 3,087 units, five outer response folds, and inner-validation layer selection. They predict episode labels from prospective hidden information on teacher-forced external text. These are not the models' own online reasoning-state labels.

| Model | Grouped macro AUROC, 95% interval | Gain over prefix-position baseline |
|---|---|---|
| Qwen | 0.852 [0.838, 0.868] | +0.286 [0.259, 0.302] |
| GPT-OSS | 0.780 [0.761, 0.797] | +0.209 [0.179, 0.227] |
| Gemma | 0.788 [0.772, 0.805] | +0.221 [0.195, 0.237] |

**Insight:** information survives removal of the original sentence-split leakage. **Limit:** only 38 independent responses, a two-feature position baseline, and bootstrap intervals conditional on saved folds/fits. The archived tables and source hashes are available; their cited raw grouped-probe result is absent in this checkout. Do not claim this audit reproduced the fits. Legacy router-layer choices were not replaced by these grouped results. [Probe sources](SOURCES.md#s03-grouped-probes)

The annotation instrument also has existing validation, not a blank slate. Static-reward grouped CV macro-F1 was 0.6505 (light) versus 0.6437 (heavy), over 2,789 units. A Qwen3.8 imported seed achieved balanced accuracy 0.7400 versus 0.7288 after optimization; the Plan-recall gate retained the seed. Low/medium effort agreed in aggregate accuracy (0.7101 on 407 reused units) but disagreed on 41 labels. An external probe inspection had 50.3% primary and 60.8% accepted-label agreement over 581 units, against a rule-assisted Codex review, not independent human ground truth. These qualify the scope of labels; another annotation optimization campaign does not address the current prefix-support failure. [Annotation sources](SOURCES.md#s04-annotation)

## E02 Correlations have model and population dependence

The imported three-model corpus has 4,647 attempts over 1,547 problems per model. AIME24/AIME25/AMC23 contribute 30/30/40 questions with 32 attempts each. The tagged Qwen cohort has 3,568 correct, 412 capped and 388 unparseable attempts; the older untagged report instead has 3,588/394/371. These are different snapshots.

Whole-trace confidence/correctness correlations change across models and benchmarks: Qwen ranges from +0.112 to +0.533 in the six reported benchmarks, Gemma from −0.256 to +0.392, GPT from −0.160 to +0.210. This supports model-specific analysis and rejects a universal confidence-increase prescription. Confidence and entropy are algebraically related; they are not independent corroborating findings. [Correlation sources](SOURCES.md#s05-correlation-and-class-descriptions)

The September audit's B cohort contains 895 attempts but only 51 distinct questions across seven models; 32 recur across models. One uncapped, length-adjusted late saved boundary-margin association survives correction: d=−0.2983 [−0.4637, −0.1330], q=0.004879. In the common reasoning range, no feature survives BH correction (minimum q=0.2190). A whole-span gap/length correlation around −0.45 approaches zero at matched absolute positions. These changes constrain interpretations of pooled signals; they do not prove all routing effects vanish. The compared metric/cohort scopes differ, so do not compute a causal “attenuation percentage.” [September audit](SOURCES.md#s06-september-routing-audit)

Retrospective Qwen expert-group associations remain small after stated controls: Analyze usage r=+0.0918; Verify r=−0.0972; Monitor r=−0.0770. Groups were selected and evaluated on the same data and overlap. A negative Verify association does not justify suppressing verification experts. [September audit](SOURCES.md#s06-september-routing-audit)

## E03 Routing information and anticipation already exist

Clean R3-D present-class readouts cover **all seven models**, 522 questions/509 families each. Both saved baseline modes and the separately reported lexical-null excess analyses show improvements within their contracts. The complete model table is in [Model coverage](MODEL_COVERAGE.md).

For next-class prediction, candidate-minus-baseline log loss is:

| Model and scope | Questions / adjacent pairs | Difference in nats/pair, simultaneous CI6 |
|---|---:|---|
| GPT primary next-class | 509 / 10,209 | −0.02073 [−0.02548, −0.01601] |
| GPT contained sensitivity | 308 / 797 | −0.01570 [−0.02575, −0.00647] |
| Qwen3.6 same-direction sensitivity | 494 / 2,492 | −0.05570 [−0.06944, −0.04154] |
| Qwen3.6 contained sensitivity | 160 / 213 | −0.03438 [−0.04927, −0.02011] |

Negative differences favor routing augmentation. The historical baseline includes past text/history and a training-only estimate of current class. However, next-class prediction is not a qualified 64-token intervention trigger or a measurement of substantive verification. GPT switch improvement is smaller (−0.00264) and fails the registered practical gate; contained switch/destination and Qwen destination sensitivities are uncertain. Do not summarize all timing questions as solved. [R3-D sources](SOURCES.md#s07-clean-readout-and-anticipation)

R3-E clean B1 has a class-associated routing gain of +0.008385 nats/token-pair [0.008044, 0.008737], with 509 questions/496 families, after excluding 69.24% of labelled tokens through its frequent-ID filter. Its different sign convention and filtered population must be retained. R3-E B4 correctness prediction is uncertain: primary log-loss change −0.000729 nats/question, nominal interval [−0.005313, +0.004384], on 508 observed questions/495 families. Its precision simulation found limited advancement rates and interval undercoverage. This does not establish equivalence or absence of useful causal control. [R3-E sources](SOURCES.md#s08-r3e-results)

## E04 Targets are grounded candidates, not semantic modules

Qwen3.6 L28/E189 and E9 have matched selection-membership differences +0.08118 and +0.06296 before a next-sentence Verify label, positive in 40/48 and 44/48 discovery families. There are 483 matched positive windows across 48 families; fold-selected proposals show positive held-out directions. The feature uses the completed source sentence and the target is an LLM class. This is conditional nomination, not independent confirmation of the final fixed pair or proof of an online opportunity. [Target sources](SOURCES.md#s09-conditional-targets)

Selection itself is sensitive to the estimator. Replacing accuracy-lift ranking with activation-rate-difference ranking retains only 14/48 Qwen, 6/16 GPT positive and 7/16 GPT negative target identities under the legacy budgets. This audit motivates an explicit selector and stability reporting; it does not measure alternative policies' utility. [Selection audit](SOURCES.md#s10-legacy-steering-and-rescoring)

## E05 Legacy global steering provides substantive negative evidence

The archived complete stochastic comparisons contain 1,395 paired attempts/465 problems. The following are **uniformly rescored** results with problem-stratified paired bootstrap intervals, under the primary attempt weighting. Generated-token means are descriptive and have no corresponding cost interval in these tables.

| Policy | Accuracy difference in percentage points, 95% interval | Mean output tokens native → guided |
|---|---|---:|
| Qwen fixed positive +1 | −0.29 [−1.15, +0.65] | 13,863 → 13,706 |
| Qwen fixed positive −1 | +1.08 [−0.14, +2.37] | 13,830 → 13,919 |
| Qwen fixed positive +2 | −2.80 [−5.16, −0.72] | 13,830 → 14,092 |
| GPT fixed positive +1 | 0.00 [−2.15, +2.01] | 6,497 → 6,580 |
| GPT fixed positive +2 | −3.30 [−6.16, −0.93] | 6,497 → 8,629 |
| GPT paper-style positive +1 | −61.65 [−69.32, −53.41] | 6,497 → 22,489 |
| GPT paper-style negative +1 | −59.57 [−67.24, −51.40] | 6,497 → 23,639 |
| Qwen margin, concurrency 25 | +0.93 [0.00, +1.86] | 13,830 → 13,810 |

No complete comparison has a positive accuracy interval strictly above zero under either reported weighting. Equal-problem weighting removes the significant-harm conclusion for the fixed +2 policies; severe paper-style promotion remains harmful. Qwen margin at concurrency 25 becomes −0.12 [−2.07, +1.80] under equal-problem weighting. These limits belong beside the headline table.

Offline rescoring covered 40,097 stored records, changing 4,054 incorrect→correct and one correct→incorrect. Earlier favorable claims must not be revived. Old GPT policies with eight targets despite native k=4 are quarantined; reported replacement comparisons use four targets. “Paper-style” is not a full external-method replication. A Qwen paper-style run stopped at 494/1,395 attempts (70.45% native versus 0.81% guided, all guided capped); it is a partial matched subset. Legacy margin hooks also changed top-8 membership in 32.7–41.8% of hook evaluations, so they do not isolate a membership-preserving margin effect. [Archived results and rescoring](SOURCES.md#s10-legacy-steering-and-rescoring)

## E06 Sparse actions engage routing; semantic effects remain uncertain

Historical A reports pair-target inclusion +0.1376 [0.0763, 0.1882] versus native across eight families; B one-pulse inclusion is +0.2940 [0.1277, 0.4761] across thirteen. These are observed first-stage effects. They do not establish that the affected experts cause a reasoning operation. [Execution and dose](SOURCES.md#s11-historical-sparse-interventions)

| Study | Horizon | Assignments | Valid semantic pairs / unknown | Scope |
|---|---:|---:|---:|---|
| A | 1,024 | 128 | 110 / 18 | 8 families; pair versus singletons and controls |
| B | 1,024 | 156 | 141 / 15 | 13 families; one versus repeated pulse |
| C | 256 | 208 | 200 / 8 | 13 families; operators |
| FRESH | 1,024 | 3,480 | 2,920 / 560 | 123 families, 134 starts; verification and commitment |

The saved A/B/C/FRESH estimates and 50,000-draw intervals have an exact historical-reproduction receipt. Do not pool horizons or treat family, start, assignment and sentence counts as interchangeable.

| Contrast | Semantic success change, percentage points | Frozen simultaneous 95% interval |
|---|---:|---|
| A pair − native | +6.25 | [−25.00, +50.00] |
| A pair − random pair | −6.25 | [−37.50, +25.00] |
| B repeat − once | −3.85 | [−26.92, +15.38] |
| C reweight − native, all starts | +3.85 | [0.00, +19.23] |
| FRESH verification force − native | +5.50 | [−4.50, +15.50] |
| FRESH verification force − random | +11.00 | [−0.50, +23.50] |

No FRESH contrast in its 28-contrast primary family strictly excludes zero. Missing-reader identification bounds widen force−native to [−16.5, +24.5] points and force−random to [−7.5, +30.0]. These are bounds, not confidence intervals. Degenerate [0,0] bootstrap intervals in small pilots cannot establish equivalence.

The frozen utility incumbent combines verification force L28/{9,189} with commitment +1 bias L24/E159. Commitment's selected point estimate is **−1.47 points** versus native [−10.29, +5.88]. Selection by the declared point-estimate rule does not make either action a proven improvement. [FRESH and incumbent](SOURCES.md#s12-fresh-semantic-results-and-incumbent)

## E07 Dense trajectories are available; utility remains a different outcome

The final historical ledger retains all 187,524 assigned sentence identities: 187,136 valid and 388 unusable, zero unresolved. The 3,480 FRESH continuations now have seven-class trajectory descriptors, observed dwell/re-entry/loops with gaps respected, expert identity frequencies/turnover and saved sparse dose telemetry. These are descriptive secondary measurements, not additional independent semantic tests. Full gate-weight trajectories, gate-TV motion, logit margins, per-sentence substantive votes and original-prompt accuracy are unavailable in this artifact. [Dense closeout](SOURCES.md#s13-dense-recovery-and-native-boundaries)

Native boundary profiles already contain 48 source traces, 6,905 records and 42 transition profiles. Offset 0 is source sentence; +1 is destination. Missing labels break adjacency. Selected window ends cannot identify closure probabilities. They are useful existing plots, not qualified token-scale anticipation.

The 3,972 historical continuation-length records have no extraction errors. In FRESH, native mean output is 984.39 tokens; bias/force/reweight means are slightly longer, not a consistent shortening pattern. All C continuations reach 256 tokens. Caps censor unrestricted lengths, and a capped post-prefix continuation is not original-prompt utility. [Length readout](SOURCES.md#s14-length-and-utility-status)

## E08 V2 engineering is qualified; semantic feasibility is not

Thirteen engineering cases passed, including both-rank all-layer capture and a 16,384-token fixture (2,637.33 seconds, 136,231,573 capture bytes). Two isolated forced-token native runs show mean expert-set turnover 0.1824; that is cross-run numerical variability, separate from passing inactive-path fidelity at the same hidden states.

The semantic pilot failed on `BatchEncoding` serialization before any of 128 assigned ratings. The correction is CPU-verified on all 64 prefixes but not GPU-qualified. More importantly, frozen apparent proposals cover **0 unchecked complete candidates, 0 checked complete candidates and 4 identified approaches**. The corresponding held-out counts are 1/0/2. An unchanged rerun cannot meet the existing ten-apparent-family requirement for all decisions. Reader failure says nothing about actual precision; proposal support is a separate feasibility problem.

Scientific pricing cells were not run. Native raw-row alignment and offline semantic labels are still unqualified for new context/anticipation fits. The 32 validation/96 final families are frozen, but v2 scientific outcomes have not been dispatched. Actual bounded implementation use was 2.6717 GPU-hours and 0.1672 dedicated CPU-core-hours; it does not price the full study. [V2 final evidence](SOURCES.md#s15-v2-final-status)

## E09 Retired or separate branches

Legacy M9 edited batch/isolation qualification failed and remains held. Native-finalization has zero of32 outcomes because its imposed exact cross-run replay gate stopped the last engineering fixture. Both native traces computed12; their first difference was newline formatting, and the instrumented output matched the second native output exactly. This does not establish an instrumentation-induced answer error. The user rejected exact cross-run replay as a future requirement for sampled GPU execution in ADR20261008 07. Counterfactual-NLL qualification remains failed; X3 is deferred. These are neither semantic failures of all routing control nor prerequisites for separately qualified free-generation work. Retain them in the failure appendix; do not restart them automatically. [Separate diagnostics](SOURCES.md#s16-separate-diagnostic-branches), [prospective correction](DECISIONS.md#adr-20261008-07-qualification-of-sampled-gpu-execution)

## E10 Current v3 qualification and discovery coverage

**Historical progress cutoff below:** the opening reconciliation and [D08](EXPERIMENT_INVENTORY.md#d08-v3-discovery-and-candidate-specific-judgments) supersede these five-discovery/four-audit waits. All work is stopped; no later treatment/utility result is inferred.

All seven backends passed14/14 engineering checks under the current source-bound executor. GPT-OSS, Nemotron, Gemma, Qwen3-30B and Qwen3.6 have now each completed all48 original discovery traces and a128-prefix audit, with no missing source. These complete frames support model-specific reader qualification; they do not establish a behavioral effect. Qwen3.5 and GLM remain in discovery; Qwen3.6 completed its retained role/context measurement and has begun priced full-audit semantic annotation. This still supplies no recognition or behavioral qualification. [Engineering closeout](../../report/verification-study-v3/QUALIFICATION_CLOSEOUT-r4.json), [Qwen3 discovery advancement](../../report/verification-study-v3/QWEN330B_DISCOVERY_ADVANCEMENT-r1.json), [Gemma discovery advancement](../../report/verification-study-v3/GEMMA_DISCOVERY_ADVANCEMENT-r1.json), [Qwen3.6 discovery advancement](../../report/verification-study-v3/QWEN36_DISCOVERY_ADVANCEMENT-r1.json)

GPT's, Nemotron's, Gemma's and Qwen3-30B's completed recognition audits fail their frozen gates, as recorded above. Qwen3.5, Qwen3.6 and GLM continue through their own semantic, target and control gates. The shared answer grader separately fails its audit and activates the existing strict-grading/unknown-bound fallback. No new conditional effect or original-prompt utility outcome is measured yet. Read [the generated status](EXPERIMENT_STATUS.md) for subsequent observed gates and [implementation](IMPLEMENTATION.md) for exact source and submission receipts.

## E11 Publication readiness review — 2026-10-08

**Preserved review at its original cutoff:** the four-model fractions, optimistic bounds, scheduler observation and PDF assessment below were not recomputed. Later Qwen3.6 qualification and the stop are indexed above; subsequent CL conversion/saved-data results are separate snapshots.

The user requested a review of the October 7 minimal-v3 implementation and set **Computational Linguistics** as the venue, with **results due by October 15**. The agreed all-seven causal-verification and joint-utility contribution is not complete. The evidence already supports seven-model routing information, historical intervention engagement and uncertain/harmful effects, and four current controller/reader qualification limits. It does not supply a new v3 causal null or useful policy.

The [source-bound review](../../report/verification-study-v3/PUBLICATION_READINESS_REVIEW-r1.json) independently reproduces all eight official confirmation fractions and support counts from the original frames and amended full decision ledgers. All 35 scientific files and 29 readiness files match their seals. No new model draw, submission or gate change was made. At 08:31:52 UTC the bounded health reader observed Qwen3.5 discovery59645811, GLM discovery59645901 and Qwen3.6 reader59686185 running, with protected successor59693197 and guards59693203/59693210; no telemetry errors or unresolved intents. Qwen3.6's first bulk reader59686166 had completed0:0 in5950seconds. This observation does not close the remaining full audit.

| Completed audit | Eligible confirmation | Already-checked confirmation | Both draws usable / 128 records |
|---|---:|---:|---:|
| GPT-OSS | 22.41% / 29 apparent families | 18.75% / 8 | 84 |
| Nemotron | 25.00% / 34 | 10.00% / 10 | 81 |
| Gemma | 25.52% / 32 | 41.67% / 12 | 91 |
| Qwen3-30B | 10.10% / 33 | 33.33% / 12 | 80 |

Both decisions require at least 70% confirmation and ten apparent families. This is a **two-draw confirmation fraction**, with invalid, truncated, null and disagreeing judgments unconfirmed. It is not independently measured semantic detector accuracy. The [reader contract](../../report/verification-study-v3/READER.json) specifies two seeded draws from one Qwen3.8-27B checkpoint. No independent calibration of this candidate-specific rubric was found in the inspected v3 evidence. Historical seven-class annotation assessments and the separate J1 equivalence audit do not validate this different task.

Missingness alone cannot explain every failed gate under the existing agreeing labels. Set every unresolved or disagreeing pair to confirmation, keep agreeing negative judgments fixed, and average within family before across families:

| Model | Eligible: observed → optimistic completion | Already checked: observed → optimistic completion |
|---|---:|---:|
| GPT-OSS | 22.41% → 72.41% | 18.75% → 56.25% |
| Nemotron | 25.00% → 66.18% | 10.00% → 65.00% |
| Gemma | 25.52% → 68.75% | 41.67% → 83.33% |
| Qwen3-30B | 10.10% → 56.57% | 33.33% → 58.33% |

Each model remains below at least one precision threshold; GPT also lacks checked-family support. These are finite-audit completion ranges **conditional on reader judgments**, not confidence intervals, bounds on semantic truth or revised gates. They do not justify re-rating the original records.

The apparent eligible-proposal records contain respectively13,16,14,22 agreeing rejections. Within those rejections, both readers mark an invalid candidate in8,9,10,15 cases; missing original-constraint binding in6,6,8,11; and already-checked status in1,4,3,6. Components overlap. This identifies concrete reader-attributed failure categories for inspecting saved text; it does not prove the shared reader is correct. Original draw dispositions and exact source hashes are in the review.

The independent discovery unit remains the **48 shared problem families**, not256 reader draws or seven independent benchmark replications. Completed frames contain46–48 families; already-checked held-out support is only2–4 families. GPT and Nemotron each have127 distinct source/prefix positions represented by128 preserved audit records across strata. The [existing count table](../../report/verification-study-v3/DISCOVERY_AUDIT_COUNTS-r1.csv) already discloses these denominators. The frozen96-family final reservation has no v3-specific expected-width or power justification in the inspected sources; feasible eligible support, reader unknowns, J1 bounds and42 comparisons must enter such an assessment before newly defined final workloads.

For publication, C1/Figure1 can reuse the seven-model CLEAN result. C2/Figure3 can report historical dose and semantic intervals under their original scope, with the current qualification table beside them; all v3 treatment cells are still unmeasured. C3/Figure4 still needs original-prompt accuracy and output-token effects for the actual qualified policy. Existing export, analysis, closeout and count helpers cover most reporting needs; a new executor is not justified by these results. The next scientific question is whether an independently assessed, prospectively qualified opportunity measurement can unlock a valid intervention test, not whether another native generation will turn the existing FAIL into PASS.

The retained121-page PDF in `PAPER_FULL_BUILD-r13.json` matches all225 bound manuscript sources at this review. It remains a working report titled *Current MoE Reasoning Experiments*, using the JAIR class and without a focused abstract/introduction/related-work/conclusion structure in the master. A Computational Linguistics submission needs a concentrated contribution, literature positioning, source-linked exhibits and the [current CL format](https://submissions.cljournal.org/index.php/cljournal/OriginalSubmissionChecklist). Reformatting or rebuilding does not fill the scientific gaps. The [next-work review](NEXT_EXPERIMENTS.md#priorities-under-the-october-15-result-deadline) and [decision record](DECISIONS.md#publication-readiness-review-and-deadline--2026-10-08) preserve the pending publication choice.

## E12 Analysis-first seven-model review — 2026-10-08

**Preserved analysis-review cutoff:** later [X11 selection](EXPERIMENT_INVENTORY.md#x11-family-held-out-expert-selection-and-stability) and [X06 transitions](EXPERIMENT_INVENTORY.md#x06-state-transitions-and-native-boundaries) close the saved-data work proposed below. Original concentration/association estimates and uncertainty remain unchanged.

The user's current objective is an all-seven paper-results contribution leading to **reduced resident expert memory with jointly measured accuracy** (ADR09), with results by October 15. No external validators or failure-centered publication fallback. This review adds source-bound analysis and manuscript integration without new inference, Slurm submissions or edits to sealed campaign sources.

**Math results recovered and checked.** [The inventory](../../report/model-analysis-review-20261008/INVENTORY.json) separates 32,529 historical attempts, 32,529 native-card-profile attempts and 3,071 continuations. Each complete corpus contains 4,647 assignments per model on 1,547 shared questions: AIME24(30), AIME25(30), AMC23(40), MATH500(500), Minerva(272), Olympiad(675). The first three have 32 attempts/question; the others have one. All 84 model/benchmark/corpus cells reproduce original strict/adjudicated means, counts, caps and token medians; saved question-bootstrap intervals are preserved rather than refitted. The main manuscript now includes the 42 native-profile cells and 33 unresolved adjudicated outcomes. Temperature, caps and checkpoint formats differ, so pooled scores are descriptive, not equal-budget model rankings. [Full cells](../../report/model-analysis-review-20261008/benchmark_cells.csv), [audit notes](../../report/model-analysis-review-20261008/AUDIT_NOTES.md).

The historical accepted J1 deployment passed its own 98/100 audit. It is distinct from the current verification campaign's 97/100 FAIL; neither supersedes the other's frozen scoring provenance. Strict/adjudicated unknown denominators and all-assignment bounds remain explicit.

**Existing positive model analysis is stronger than the old manuscript summary.** All seven clean present-class routing readouts improve the saved judge-context baseline on 522 questions/509 families each; lexical-null controls and original intervals are retained. Clean next-class evidence already exists for GPT and Qwen3.6. Qwen3.6's narrower matched Verify-entry study already has four train-selected fold proposals with positive held-out directions; the final full-discovery L28/E189 and E9 identities are nominations, not independently confirmed causal modules. The completed GPT B4 correctness-prediction study also exists. These are not missing experiments to rerun. [Readout export](../../report/model-analysis-review-20261008/routing_readout.csv), [anticipation export](../../report/model-analysis-review-20261008/anticipation.csv).

Cohort A has one outcome-blind sampled attempt on each of 1,547 shared questions/model. Selected cohort B has 895 attempts, 51 questions in the union, but only **one question in the all-seven intersection**. Keep these populations separate. The saved 147 adjusted expert-use/correctness rows are descriptive associations, not evidence of dispensability or new held-out pruning scores. Missing sentence labels must break transition adjacency.

**New descriptive result: expert concentration differs substantially by model and state.** [The new source-bound figure](../../paper/6a510c6db0d9b08aa438c142/generated/model_analysis_review/labelled_expert_coverage.pdf) uses existing compact cohort-A enrichment summaries. Within each layer it ranks experts by total selected slots across the seven labelled states. Coverage is the fraction of the original labelled token-layer-expert selections in the retained set; it is token/slot-weighted, not equal-family or equal-benchmark weighted. Ranking and measurement use the same cohort.

| Model | Coverage at half the expert identities per layer | Least-covered state | That state's coverage |
|---|---:|---|---:|
| GPT-OSS |88.34%|Monitor|84.58%|
| Nemotron |61.90%|Monitor|55.13%|
| Qwen3.5 |80.32%|Monitor|77.39%|
| Qwen3.6 |83.20%|Monitor|79.30%|
| Gemma |96.11%|Read|94.04%|
| GLM |71.92%|Monitor|59.52%|
| Qwen3-30B |92.94%|Monitor|88.35%|

Monitor is least covered in six models, illustrating how a pooled usage ranking can underrepresent one labelled episode class. This motivates a prospective state-aware retention comparison; it does not establish that the omitted experts are necessary or dispensable. Half-retention is a plotting landmark, not an adopted deployment budget. The curves concern the labelled subset only and do not measure complete top-k containment, selected-weight mass, rerouted behavior, group-feasible masks, accuracy or resident-memory savings. No held-out or inferential claim is made.

The export reconstructs exact selected counts from `rate_in * class_tokens * native_k`; the original rates are normalized slot shares. Complete state/layer/expert frames, native-k mass, monotonicity and full-set coverage pass. A separate integer reconstruction reproduces GPT and GLM's half-retention values exactly. Runtime was 12.341 seconds with existing summaries; no raw tensor processing. [Receipt](../../report/model-analysis-review-20261008/LABELLED_EXPERT_COVERAGE-r1.json), [CSV](../../report/model-analysis-review-20261008/labelled_expert_coverage.csv), [exporter](../../scripts/paper/export_labelled_expert_coverage_r1.py).

**Memory accounting is now grounded, but deployment benefit is missing.** Header-only inspection binds all seven checkpoints and separates routed experts, draft paths and other tensors. Native expert serialized bytes range from 10.166 GB (GPT) to 57.982 GB (Qwen3-30B), including expert-local quantization metadata. GLM's index declares 31.221 GB but actual tensor payload ranges sum to 62.443 GB; shard sizes independently confirm that larger payload. Runtime repacking, shared/draft paths and KV/workspace make serialized bytes different from GPU residency. [Header receipt](../../report/model-analysis-review-20261008/CHECKPOINT_HEADER_INVENTORY.json), [model byte table](../../report/model-analysis-review-20261008/checkpoint_tensor_bytes.csv).

Historical Qwen deactivation already ran on 13 families/156 short branches, kept all expert weights loaded and measured neither original-prompt accuracy nor freed memory. The current v3 operator qualifies positive pulses only. Actual resident removal, and independently held-out accuracy/memory evaluation require new work. The public package interface is deferred as an eventual paper output. Per-attempt counts and header metadata already permit the next training-only selection analysis without new generation. The [two immediate cards](NEXT_EXPERIMENTS.md#current-analysis-and-memory-priorities) and [source review](debate/package-analysis-review-20261008.md) define that boundary.

**Results-first correlation exhibit added.** The complete [49-cell matrix](../../paper/6a510c6db0d9b08aa438c142/generated/model_analysis_review/expert_usage_correctness.pdf) exports all seven models × seven class-associated expert profiles from saved fits. Cells show partial Pearson correlation between adjudicated correctness and the profile's mean selected-slot share across **all reasoning tokens**, averaging equally over layers with nominated experts. This is distinct from usage only inside the class's labelled spans.

Original controls are dataset effects, log completion length, retrospective difficulty, the class token fraction among labelled tokens, and a capped indicator when available. Qwen3.5/Qwen3.6 Analyze-associated usage has r=+0.09179/+0.08183, while Verify-associated usage has r=−0.09724/−0.10079. These modest, model-dependent associations motivate controlled selection comparisons; they cannot justify suppressing verification. All 49 coefficients are shown, including uncertain cells. The plot exports existing estimates without refitting or collecting data.

Stars retain the original within-model BH q<0.05 across 21 mean-usage tests (seven classes × all/in/out); they are not a correction across the 49 displayed cells. Original intervals use 1,000 question-ID bootstrap samples, refitting nuisance controls with expert sets and difficulty fixed; duplicate families are not merged. Full-cohort expert nomination and retrospective difficulty make this an exploratory association analysis, not a held-out deployment predictor. The five estimator source files match all seven saved provenance records (35/35). [Figure receipt](../../report/model-analysis-review-20261008/EXPERT_ASSOCIATION_FIGURE-r1.json), [full saved rows](../../report/model-analysis-review-20261008/expert_usage_correctness_saved.csv), [method audit](../../report/model-analysis-review-20261008/EXPERT_ASSOCIATION_METHOD-r1.json).

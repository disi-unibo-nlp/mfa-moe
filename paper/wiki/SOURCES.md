# Canonical sources and precedence

Start with the [experiment inventory](EXPERIMENT_INVENTORY.md) to locate data and completed analyses by scientific question. This page preserves original provenance and observation cutoffs; earlier running/pending/authorization notes are historical after [ADR12](DECISIONS.md#adr-20261008-12-user-directed-stop-and-experimentclaim-review).

| Source family | Catalogue entries |
|---|---|
| S03/S04 external probes and annotation | [D06](EXPERIMENT_INVENTORY.md#d06-external-sat-annotation-and-probe-data), [X02/X03](EXPERIMENT_INVENTORY.md#x02-reasoning-class-annotation-qualification) |
| S05/S06/S17 correlations and expert profiles | [D04/D05](EXPERIMENT_INVENTORY.md#d04-ab-cohorts-and-seven-class-sentence-annotations), [X04](EXPERIMENT_INVENTORY.md#x04-routing-correlations-class-profiles-and-adjusted-correctness-associations) |
| S07/S08 readout, anticipation and correctness | [X05](EXPERIMENT_INVENTORY.md#x05-present-class-readout-next-class-and-correctness-prediction) |
| S09/S10 conditional targets and legacy steering | [X07](EXPERIMENT_INVENTORY.md#x07-conditional-target-nomination-and-descriptive-concentration), [X08](EXPERIMENT_INVENTORY.md#x08-legacy-global-steering-and-rescoring) |
| S11–S16 interventions, dense recovery and qualification | [D07](EXPERIMENT_INVENTORY.md#d07-historical-intervention-continuations-and-dense-labels), [X09/X10](EXPERIMENT_INVENTORY.md#x09-abcfresh-actions-dense-recovery-and-utility-boundary) |
| S18 benchmark corpora and bytes | [D01–D03](EXPERIMENT_INVENTORY.md#d01-historical-benchmark-generations), [X12](EXPERIMENT_INVENTORY.md#x12-checkpoint-byte-accounting-and-resident-memory-boundary) |
| S19/S20 memory-study saved-data results | [D09](EXPERIMENT_INVENTORY.md#d09-reserved-independent-populations-and-expert-memory-artifacts), [X11](EXPERIMENT_INVENTORY.md#x11-family-held-out-expert-selection-and-stability) |

The result itself outranks an old progress note **for the same scope**. User decisions outrank reviewer recommendations. A newer study does not retroactively change an older study's cohort, horizon, scorer, estimand or authorization. Source file hashes and availability are recorded in [source manifest](source_manifest.json); selected machine-readable checks are in [evidence snapshot](evidence_snapshot.json).

“Archived report-backed” means the saved paper tables or report were inspected, without rerunning the underlying fits. “Result artifact inspected” means the saved machine-readable result or completion receipt was read; it does not mean every raw tensor was independently reproduced in this session.

## S01 User notes and current scope

[Original progress notes](../progress_report.md). Current decisions and exact scope corrections are recorded in [DECISIONS.md](DECISIONS.md). The notes are preserved verbatim. This wiki is the consolidation destination; no new cloud Page or public website was created.

## S02 Private reference

[Rational Manifolds PDF](../../report/iclr2027_conference_anonym_private.pdf). Read locally with existing Ghostscript. No full PDF was sent to either model reviewer. DeepSeek received a short methodological paraphrase within the explicitly approved research packet.

## S03 Grouped probes

Archived report-backed: [probe report](../6a510c6db0d9b08aa438c142/probeTest.tex), [summary table](../6a510c6db0d9b08aa438c142/probe_grouped_summary_table.tex), [target/layer table](../6a510c6db0d9b08aa438c142/probe_grouped_targets_table.tex), [source bindings](../6a510c6db0d9b08aa438c142/grouped_probe_sources.json), [setup](../6a510c6db0d9b08aa438c142/experimental_setup.tex). The cited raw Qwen grouped result under `results/probeTest/.../probes_grouped_s42/results.json` is absent here. The later grouped experiment repairs the legacy response-overlap issue; legacy intervention-layer selection remains separately limited.

## S04 Annotation

Archived report-backed: [judge experiments](../6a510c6db0d9b08aa438c142/gepaLLMAsJudge.tex), [effort table](../6a510c6db0d9b08aa438c142/judge_effort_table.tex), [stratified annotation](../6a510c6db0d9b08aa438c142/stratified_label_results.tex). Development, reused assessment and generated-math deployment populations must not be conflated. The nominal early locked partition had prior inspection; do not call it pristine.

## S05 Correlation and class descriptions

Archived report-backed: [imported correlation results](../6a510c6db0d9b08aa438c142/correlation_results.tex), [multimodel results](../6a510c6db0d9b08aa438c142/multimodel_correlation_results.tex), [run audit](../6a510c6db0d9b08aa438c142/correlation_run_audit.json), [class transitions](../6a510c6db0d9b08aa438c142/class_transition_results.tex), [transition statistics](../6a510c6db0d9b08aa438c142/class_transition_statistics.json). Older [report synthesis](../../report/active_synthesis.tex) uses an earlier untagged Qwen corpus. Neither population replaces the other.

## S06 September routing audit

Result artifacts and audit inspected: [review](../../report/routing-steering-audit-20260928/REVIEW.md), [reproduced evidence](../../report/routing-steering-audit-20260928/evidence-reproduced.json), [final evidence](../../report/routing-steering-audit-20260928/evidence-final.json), [proposed decisions](../../report/routing-steering-audit-20260928/DECISIONS.md), [frozen baselines](../../report/routing-steering-audit-20260928/frozen-baselines.md). Later seven-model A coverage is S17. The B union and historical confounder caveats retain their own scope.

## S07 Clean readout and anticipation

Result artifacts inspected: [seven-model readout](</leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/readout-results.v3.json>), [anticipation](</leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/anticipation-results.json>). Repository summaries: [paper snapshot v2](../../report/experimental-resume-v1/PAPER_SNAPSHOT_v2.json), [241-entry claim ledger v2](../../report/experimental-resume-v1/CLAIM_LEDGER_v2.json). Their positive R3-D results supersede earlier incomplete-cell notes. The snapshot's statement that no causal outcomes exist is subsequently superseded by S11/S12, not by rewriting the snapshot.

## S08 R3E results

[B1 summary](../../report/experimental-resume-v1/PAPER_SNAPSHOT_v2.json), [B4 observed result and precision study](../../report/experimental-resume-v1/R3E_B4_PRECISION_READOUT_v1.md). These distinguish class-associated information from correctness prediction and document power/coverage limitations. Do not infer a causal null from the B4 non-rejection.

## S09 Conditional targets

Result artifact inspected: [dense Verify source](</leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-verify-source-v2/RESULT.json>). Supporting current [candidate table](../../report/routing-study-v1/CANDIDATE_TABLE.json), [global/conditional comparison](../../report/routing-study-v1/GLOBAL_CONDITIONAL_COMPARISON.json). These nominate experts; they do not establish causal specialization.

## S10 Legacy steering and rescoring

Archived report-backed: [guided results](../6a510c6db0d9b08aa438c142/guiding_results.tex), [identity table](../6a510c6db0d9b08aa438c142/identity_comparison_table.tex), [margin table](../6a510c6db0d9b08aa438c142/margin_comparison_table.tex), [token table](../6a510c6db0d9b08aa438c142/guiding_token_table.tex), [rescoring report](../6a510c6db0d9b08aa438c142/guiding_rescoring_results.tex), [sensitivity](../6a510c6db0d9b08aa438c142/guiding_sensitivity.json), [current source bindings](../6a510c6db0d9b08aa438c142/current_results_sources.json), [invalidated OSS runs](../6a510c6db0d9b08aa438c142/invalidated_oss_runs.json), [selector audit](../6a510c6db0d9b08aa438c142/expert_selection_audit.tex).

## S11 Historical sparse interventions

Result artifacts inspected: [study inventory v3](../../report/routing-study-v1/STUDY_INVENTORY_v3.json), [historical integration](../../report/routing-study-v1/HISTORICAL_INTEGRATION.json), [exact saved-result reproduction](../../report/routing-study-v1/HISTORICAL_REPRODUCTION.json), [execution and first-stage results](../../report/experimental-resume-v1/OVERNIGHT_EXECUTION_2026-10-05.md). [Report v4](../../report/routing-study-v1/REPORT_v4.md) correctly reports historical effects, but its 11,200-label progress count is stale; use S13 for final completion.

## S12 FRESH semantic results and incumbent

Result artifacts inspected: [FRESH analysis](../../report/experimental-resume-v1/OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2/ANALYSIS.json), [assigned outcomes](../../report/experimental-resume-v1/OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2/ASSIGNED_RESULTS.json), [selected incumbent](../../report/experimental-resume-v1/utility-price-pilot-attachments-v5-ce86e4bcaa9c5c4a-59379614/SELECTED_POLICY.json). Preserve both simultaneous sampling intervals and missing-reader identification bounds. The selected policy is unproven.

## S13 Dense recovery and native boundaries

Result artifacts inspected: [final label accounting](../../report/routing-study-v1/closeout-59495690/LABEL_ACCOUNTING.json), [trajectory result](../../report/routing-study-v1/dense-recovery-v3/TRAJECTORY_RESULT.json), [native boundary profiles](../../report/routing-study-v1/NATIVE_BOUNDARY_PROFILES.json), [final identity binding](../../report/routing-study-v2/implementation-20261006-v1/closeout/DENSE_LEDGER_BINDING.json). Labels, semantic reader pairs and independent questions have different denominators.

## S14 Length and utility status

[Continuation-length readout](../../report/experimental-resume-v1/CONTINUATION_LENGTH_READOUT_2026-10-05_v1.json), [temporary priority amendment](../../report/experimental-resume-v1/ACCURACY_LENGTH_PRIORITY_2026-10-05_v1.json), [independent family manifest](../../report/routing-study-v1/INDEPENDENT_FAMILY_MANIFEST.json), [historical integration's explicit utility availability](../../report/routing-study-v1/HISTORICAL_INTEGRATION.json). Native-finalization or utility pilots are not completed original-prompt policy comparisons.

## S15 V2 final status

Result artifacts inspected: [final status](../../report/routing-study-v2/implementation-20261006-v1/final/STATUS.md), [final report](../../report/routing-study-v2/implementation-20261006-v1/final/REPORT.md), [closeout](../../report/routing-study-v2/implementation-20261006-v1/final/CLOSEOUT.json), [semantic coverage](../../report/routing-study-v2/implementation-20261006-v1/final/SEMANTIC_COVERAGE.json), [outcome-blind support precheck](../../report/routing-study-v2/implementation-20261006-v1/AUDIT_SUPPORT_PRECHECK.json), [CPU serialization repair](../../report/routing-study-v2/implementation-20261006-v1/AUDIT_SERIALIZATION_FIX.json), [implementation contracts](../../docs/routing-study-v2.md). Scheduler completion of the semantic pilot coexists with an explicit scientific failure receipt.

## S16 Separate diagnostic branches

[M9 replay adjudication](../../report/experimental-resume-v1/M9_REPLAY_ADJUDICATION_v1.md), [native-finalization diagnostic](../../report/experimental-resume-v1/NATIVE_FINALIZATION_DIAGNOSTIC_v1.json), [reconciled inventory including NLL and X3](../../report/routing-study-v1/STUDY_INVENTORY_v3.json). These holds do not automatically block a different qualified free-generation contract.

## S17 Latest seven model observational analysis

Result summaries inspected: [cross-4 summary](</leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/v3_analysis/results-r2/cross-4/F_summary.md>), [cross-4 completion](</leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/v3_analysis/results-r2/cross-4/complete.json>). Per-model A/B summaries and capture metadata in the sibling `gpt`, `nemotron`, `qwen35`, `qwen36`, `gemma`, `glm`, `qwen330b` directories supply the coverage table. [Pinned model profiles](../../src/moe_exp/correlation_pipeline/model_profiles.py) and [router adapters](../../src/moe_exp/models/router_adapters.py) define code-side contracts; local checkpoint configs were checked during the model audit.

## Changes in status that must not be forgotten

| Earlier statement | Current interpretation |
|---|---|
| Probes share responses across train/test | True of the legacy split; later grouped study exists, with limited population and report-backed provenance |
| Only three A models were analyzed | Superseded by seven-model cross-4 on September 29 |
| Only hidden metrics survive any adjustment | Too broad; the September audit found a saved boundary-margin effect under one adjustment, with weaker common-range evidence |
| No causal outcomes exist | Superseded for A/B/C/FRESH; remains true for new v2 scientific dispatch |
| Dense recovery has 11,200/187,524 committed | Superseded by 187,136 valid plus 388 unusable, zero unresolved |
| Semantic job COMPLETED | Scheduler fact only; zero reader ratings and skipped pricing remain scientific failure |
| Serialization is the only remaining audit problem | False: frozen proposal support is also insufficient |
| Qwen is the paper's causal scope | Rejected by the user; all seven models are required |

## Audit boundary

The consolidation inspected major paper tables, source bindings, September audit, clean R3 results, A/B/C/FRESH effects, dense closeouts, v2 receipts, and recent relevant conversation summaries. It did not rerun model generation, bootstraps, every historical raw record or every tensor. The source manifest includes hashes of the selected preserved evidence; it is an index, not a claim of universal reproducibility. Work-directory sources are linked without copying their datasets. Their long-term preservation remains governed by the existing storage policy, not a transfer initiated here.

## S18 October 8 analysis-first result and storage inventory

[INVENTORY.json](../../report/model-analysis-review-20261008/INVENTORY.json), [source hashes](../../report/model-analysis-review-20261008/SOURCES.json) and [audit notes](../../report/model-analysis-review-20261008/AUDIT_NOTES.md) bind both benchmark corpora, all-seven readout, narrower anticipation, A/B populations and expert summaries. All 84 saved benchmark cells were reproduced from the compact outcome sidecar; no inference or fit was rerun. The historical 98/100 accepted scorer deployment is distinct from the current v3 shared 97/100 failure.

[CHECKPOINT_HEADER_INVENTORY.json](../../report/model-analysis-review-20261008/CHECKPOINT_HEADER_INVENTORY.json) binds configurations, indexes and safetensors headers, excluding weight payload reads/hashes. It reports serialized tensor bytes, not runtime allocations; GLM's inconsistent index total is retained and actual offset bytes checked against shard sizes.

[LABELLED_EXPERT_COVERAGE-r1.json](../../report/model-analysis-review-20261008/LABELLED_EXPERT_COVERAGE-r1.json) binds the new 12.341-second saved-summary analysis, its exporter, seven compact enrichment inputs and figure/CSV hashes. It establishes labelled selection-slot concentration in the observed cohort only. Per-attempt count archives remain available for later training-only analyses; this export does not replace them or qualify a pruning policy.

The CL manuscript conversion and its latest build receipt are a separate editorial snapshot. Older scientific results and build receipts remain preserved; a compiled PDF does not establish the missing resident-memory or accuracy result.

## S19 Adopted expert-memory study implementation

**Historical implementation cutoff:** later S20 closes the saved-data fits/transitions; ADR12 cancels remaining allocations.

[Protocol](../../report/expert-memory-study-v1/PROTOCOL.json),
[campaign source closure](../../report/expert-memory-study-v1/CAMPAIGN.json) and
[implementation closeout](../../report/expert-memory-study-v1/READINESS-r1.json)
bind the separate private harness, all seven native contracts, 40 saved sources,
657 inspected runtime Python sources and the unchanged 35 scientific/29 readiness
files of verification v3. The 55 passing synthetic contracts establish admission,
fold/mask, metadata, grading, missingness and reconciliation behavior. They do
not qualify GPU loading or fidelity. No new-study workload was submitted.

[First CPU card](../../report/expert-memory-study-v1/cards/CPU_MEASUREMENT.json)
is finite and reviewable: 4 CPUs, 16 GiB, 20 minutes, serial/normal, at most
1.333 billing core-hours. [Live observation](../../report/expert-memory-study-v1/PREFLIGHT_OBSERVATION-r1.json)
and [scheduler validation](../../report/expert-memory-study-v1/SCHEDULER_VALIDATION-r1.json)
record entitlement/storage and successful test-only validation. The reported
October 11 start is a prediction, not an accepted job or a deadline guarantee.
Actual resource authorization remains required by the supplied implementation plan.

[Paper methods snapshot](../../report/expert-memory-study-v1/PAPER_BUILD-r1.json)
binds the 133-page offline PDF, current canonical hashes and visual checks on
pages 1/8/9/10/67. It adopts the selected defaults and references the generated
verification status without changing that campaign's sources or ownership.
New retrospective fits, transition results, physical GPU qualification and
prospective accuracy remain unmeasured. Population reservation waits for complete
later-exposure reconciliation; neither synthetic tests nor compilation release
any scientific allocation. This snapshot supersedes the r3 CL build only for
the manuscript source/PDF, preserving every older artifact.

[EXPERT_ASSOCIATION_FIGURE-r1.json](../../report/model-analysis-review-20261008/EXPERT_ASSOCIATION_FIGURE-r1.json) binds the complete 49-cell saved-fit export, with no sign/significance selection or new fit. [The estimator audit](../../report/model-analysis-review-20261008/EXPERT_ASSOCIATION_METHOD-r1.json) records the exact controls, original within-model 21-test BH family, question-ID bootstrap and full-cohort nomination caveats. Source implementation hashes match all seven preserved provenance records.

## S19 Current CL manuscript snapshot

[CL_MANUSCRIPT_BUILD-r3.json](../../report/model-analysis-review-20261008/CL_MANUSCRIPT_BUILD-r3.json) binds the current results-first CL source/PDF snapshot. It supersedes `PAPER_FULL_BUILD-r13.json` and the earlier CL r1/r2 receipts only as an editorial build. Historical source inputs, scientific outcomes and previous PDFs are preserved. [RUNBOOK.md](RUNBOOK.md#current-cl-manuscript-build) records the new versioned builder; compare hashes after any later export.

## S20 Authorized expert-memory jobs and observed retrospective fits

**Historical authorization/allocation snapshot:** ADR12 supersedes execution authority and cancels the queued/running GPU qualifications and handoff. The completed retrospective results below remain observed.

[WORKLOAD_AUTHORIZATION-r1](../../report/expert-memory-study-v1/WORKLOAD_AUTHORIZATION-r1.json) records the later instruction "Go on and submit everything we need". It supersedes S19's implementation-only/no-allocation status, retaining all scientific gates. [SUBMISSION_CLOSEOUT-r1](../../report/expert-memory-study-v1/SUBMISSION_CLOSEOUT-r1.json) binds nine successful CPU jobs (0.262allocated core-hours), seven accepted GPU qualifications and released primary59719965 with two prearmed guards59719999/59720001. Acceptance establishes neither GPU fidelity nor accuracy. No verification job/source/controller was changed.

[CPU_PRICE](../../report/expert-memory-study-v1/CPU_PRICE.json), [SAVED_DATA_CLOSEOUT-r1](../../report/expert-memory-study-v1/SAVED_DATA_CLOSEOUT-r1.json), [exhibits-r2](../../report/expert-memory-study-v1/exhibits-r2/EXPORT.json) and [frozen masks](../../report/expert-memory-study-v1/FROZEN_MASKS.json) supersede unmeasured retrospective-fit/transition status for all seven models. The complete five-fold family-held-out analysis preserves both positive and unfavorable comparisons. Coverage is not post-removal accuracy. Consecutive original sentence indices and same segments are required; missing labels break adjacency, label coverage and family support are reported, and dwell endpoints are censored.

[Exposure ledger](../../report/expert-memory-study-v1/LATER_EXPOSURE_LEDGER-r1.json) and [population freeze](../../report/expert-memory-study-v1/POPULATION.json) bind reviewed fixed/prepared rosters and the original100-item J1 frame. They exclude no additional eligible family, yielding32pilot-only/4319final-order families. [Continuation readiness](../../report/expert-memory-study-v1/CONTINUATION_READINESS-r1.json) seals new independent sources and nine passing handoff contracts; the original26-file campaign closure and35/29 verification closures stay unchanged. Final allocation requires measured prices, frozen N and complete scheduler/deadline assessment. Authorization persists across these gates.

[PAPER_BUILD-r2](../../report/expert-memory-study-v1/PAPER_BUILD-r2.json) binds the134-page offline CL snapshot with both observed retrospective figures and their limitations. Offline exit0 in9.667seconds; pages1/8/9/10 visually checked, with no off-page blocks, missing glyphs or unresolved references. This supersedes the earlier methods snapshot only for canonical manuscript/PDF status. The failed older-builder attempt and all previous sources/receipts/PDFs remain preserved. GPU capability, actual residence and prospective accuracy are still unmeasured.

## S21 Catalogue reconciliation at the user stop

[EXPERIMENT_INVENTORY.md](EXPERIMENT_INVENTORY.md#d08-v3-discovery-and-candidate-specific-judgments) supersedes the earlier five-roster/four-recognizer **progress summary**, preserving every original receipt and its observation cutoff. Bounded canonical-file inspection finds six rosters with48 discovery receipts and Qwen3.5 with42. GLM has a128-record frame but no independently reproduced full archive/accounting closeout in this catalogue. Qwen3.6's [saved RECOGNIZER.json](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/candidate-verification-v3-20261007-r4/qwen36/RECOGNIZER.json) binds a256-entry original-draw ledger and reports FAIL: eligible28.57%/28 apparent families, already-checked8.33%/12. This replaces its reader-wait summary; it is not a new fit or causal result.

[Validation](experiment_inventory_validation.json) records bounded availability/count/link checks and preserved source hashes. [STOP_RECEIPT](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json) remains authoritative for cancellation and all48 question-identity matches/model. Full historical text/token/routing compatibility remains uninspected. [Reuse and missingness](EXPERIMENT_INVENTORY.md#what-can-be-reused) distinguish that open inspection from absent files and unmeasured endpoints.

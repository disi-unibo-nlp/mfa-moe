# Joint measurement recovery plan — Codex contribution, 2026-09-24

**Status: proposal for agreement with Claude and the user. No new submissions authorized or made. All 18 campaign jobs were cancelled; no new watchers were armed.** This plan supersedes the old full-cohort collection order for planning purposes, while preserving its files and frozen bundles. Claude owns scoring/extraction validation and measured generation/regeneration costs. Codex audited downstream consumers, calibration, local generation settings and capture costs. Claude's numerical rescoring findings are attributed to his report, not independently certified here.

## 1. Stop receipt, accounting and preserved work

Machine-readable receipt: `results/measurement-study/stop-20260924.json`. It records ownership and live state before cancellation, original commands/bundles, final accounting, artifact locations and resume gates for every job. All final states are **CANCELLED by 133943**. ExitCode 0:0 on a cancelled job is not experiment acceptance. All pending dependencies were cancelled too; no campaign jobs remained in the queue at verification.

| Job | Stage | Elapsed | GPU-hours used | Preserved resume location/contract |
|---|---|---:|---:|---|
| 58492402 | Qwen3.6 capture shard 0 | 07:42:17 | 15.409444 | Q36/0; 168 accepted attempts |
| 58492407 | Qwen3.6 capture shard 1 | 07:42:17 | 15.409444 | Q36/1; 123 accepted attempts |
| 58492432 | Qwen3.6 capture shard 2 | 00:00:00 | 0 | Q36/2; no accepted attempts |
| 58492434 | Qwen3.6 capture shard 3 | 07:42:17 | 15.409444 | Q36/3; 133 accepted attempts |
| 58492436 | Qwen3.6 analysis | 00:00:00 | 0 | A/qwen36; rebuild with corrected score binding |
| 58516703 | Qwen3.6 supplement | 00:00:00 | 0 | S/qwen36; rebuild with corrected score binding |
| 58517706 | Qwen3 label recovery | 00:13:37 | 0.453889 | L/checkpoints/batch-*.json; 16,000 saved, 9,000 remaining in part 01 |
| 58518846 | Qwen3 label analysis | 00:00:00 | 0 | results/measurement-study/completed-labels/qwen330b |
| 58519367 | Qwen3 preparation | 00:00:00 | 0 | results/measurement-study/recovery-storage/manifests/qwen330b |
| 58519372 | Final source join | 00:00:00 | 0 | results/measurement-study/source-join-v2 and completed-labels |
| 58520134 | Qwen3 sizing | 00:00:00 | 0 | results/measurement-study/recovery-proposals/qwen330b.json |
| 58531562 | Nemotron capture shard 0 | 00:00:00 | 0 | N/0; one prior validation capture preserved |
| 58531615 | Nemotron capture shard 1 | 00:00:00 | 0 | N/1; one prior validation capture preserved |
| 58531684 | Nemotron capture shard 2 | 00:00:00 | 0 | N/2; one prior validation capture preserved |
| 58531747 | Nemotron capture shard 3 | 00:00:00 | 0 | N/3; one prior validation capture preserved |
| 58531977 | Nemotron analysis | 00:00:00 | 0 | A/nemotron; rebuild with corrected score binding |
| 58532720 | Nemotron supplement | 00:00:00 | 0 | S/nemotron; rebuild with corrected score binding |
| 58537773 | Final coverage including Qwen3.5 | 00:00:00 | 0 | report/measurement-study/production-results; regenerate from accepted stages |

Paths are relative to this checkout except Q36:

- Q36 = `/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/measurement-study-v1/capture/qwen36`. These are existing preserved inputs, not authorization for new scratch writes. Matching original manifests are under the sibling `manifests/qwen36/` directory.
- N = `results/measurement-study/recovery-native-v1/capture/nemotron`; its manifests are under `recovery-native-v1/manifests/nemotron`.
- A = `results/measurement-study/production-analysis`; S = `results/measurement-study/production-supplement`.
- L = `results/measurement-study/label-resume-v1/qwen330b/part-01`.

Capture resume: verify each `.complete.json`, its manifest binding and referenced file hashes; reuse only complete exact-input attempts. Preserve unmarked partial files but do not count them as accepted. A changed scientific cohort requires a new manifest and a provenance-preserving reference to reusable packets, not bypassing the old binding. Original frozen production-code-v3 and recovery-v12 bundles remain immutable. Qwen3 label resume must retain judge/program/source hashes and the 57,344 context contract; checkpoint records are authoritative, not the stale preflight count. Completing the remaining old labels is not a dependency of the proposed correctness experiment.

These stopped allocations used **46.682222 GPU-hours**, already included in the reconciled campaign total **94.365 / 360 GPU-hours**, leaving **265.635** with zero live commitments. Updated ledgers: recovery-accounting.json, recovery-20260924.json and production-submissions.json under results/measurement-study. Previously separate label jobs 58487704/58487708 retain their existing ledger exclusion. No outputs/checkpoints/models were deleted. No cancellation grants authority to restart.

## 2. Detection findings and validity boundaries

Full consumer inventories and line-level evidence:

- [Measurement pipeline audit](CODEX_AUDIT_measurements_20260924.md)
- [Guiding/calibration audit](CODEX_AUDIT_guiding_20260924.md)
- [Generation-settings audit](CODEX_AUDIT_generation_20260924.md)
- [Measured capture evidence](CODEX_measured_capture_costs_20260924.json)

Principal findings:

1. `scripts/analyze_available_metrics.py:132–141` trusts stored correctness and sometimes converts unknown to false; `:177–190` drops unscored rows and predominantly compares between-question averages. Correcting scores alone does not remove difficulty confounding. Source joins, production compact rows, supplements and reports inherit those outcomes.
2. The historical sampler (`../repo/src/moe_exp/correlation_pipeline/sample_stratified.py:272–279,336–400`) excludes unscored attempts and selects one eligible attempt per question using correctness/length strata. This disproportionately removes truncations; it is not a literal universal cap filter. Actual accepted data include 10 cap-flagged GPT attempts, 6 GLM and 9 Qwen3.6. GPT has 1,580 attempts over 1,514 questions and just three repeated-question groups; GLM has 1,390/1,390 and Qwen3.6 424/424. Broad within-question inference remains unsupported.
3. **Confirmed stale-cache bug:** `measurement_study/production_analyze.py:20` accepts an existing complete.json without checking input_sha256. The exact function returned an old completion marker in a changed-input reproduction. `:18,34` also imports metrics from live ROOT/scripts, escaping the frozen bundle. New analyses need fresh output roots, hash-checked resume and genuinely frozen helper imports.
4. `scripts/analyze_available_metrics.py:148–153` converts all-unknown annotations into seven zero class fractions. Emit missing fractions with observed-label denominators. Do not equate unknown classes with class absence.
5. `src/moe_exp/routing_study/calibration.py:58–84` ranks Steer experts using unpaired outcome means even though a within-question statistic is computed descriptively. Identity and margin problem balancing also do not control question difficulty. Refit correctness-derived targets only with corrected, paired evidence. Legacy checkpoint/scorer reuse checks are insufficient; see guiding audit.
6. Explore/Monitor class triggers are not implemented by `src/moe_exp/routing_study/policy.py:7–8,84–113`. Existing gates are always/persistent/unstable/random/position. Retrospective sentence labels also include next_sentence (`../repo/src/moe_exp/correlation_pipeline/annotation_partition.py:76`), so sentence F1 is not proof of a causal online trigger.
7. Reporting completion validates stage artifacts, not corrected outcomes or an adequate scientific design (`measurement_study/production_report.py:28–45,91–103,155`). Paper acceptance needs separate outcome-validity, cohort and analysis gates.

**Reuse:** generation text and token IDs, termination metadata, valid native IDs/weights/gaps and stored reducers, class-label checkpoints, engineering validations, measured throughput. Judge prompts do not use is_correct; rescoring does not require re-annotation. Existing expert/class associations are descriptive on their selected population, not validated correctness interventions. Full hidden/router tensors were not retained; new metrics needing them require replay.

**Supersede without overwriting:** old correctness fields in joined/compact rows via a hash-bound sidecar; correctness associations/p/q/intervals and headline claims; selected-cohort accuracy presented as full-corpus accuracy; correctness-derived expert rankings/policies and dependent accuracy-frontier claims. Affected accepted outputs include production-analysis/{gpt,glm}, production-supplement/{gpt,glm}, results/available-metrics/v1, source-join-v2 and the intermediate-findings/recovered-routing-associations/production-results reports. Computational acceptance remains evidence of execution, not of scientific validity. Claude reports six of eight headline associations disappearing after correction; that claim remains subject to his rescoring validation.

## 3. Order of work and proposed minimal run matrix

First freeze the corrected score sidecar with model/checkpoint, dataset/question/sample, text/token hashes, old/new outcome, scorer and extractor identities, termination status and adjudication status. Keep **correctness and termination as separate fields**: a capped output may or may not contain a valid answer; unscored is never automatically incorrect. Report both denominators. Ambiguous Minerva units/rounding need explicit adjudication/exclusion policy with counts. Bind every derivative to score, cohort, code and packet hashes.

The initial GPU design is deliberately a bounded exploratory test, not a claim of adequately powered null evidence across every model. Select questions and random seeds before looking at routing effects; publish the selection counts. Use AIME24/AIME25/AMC23 repeated attempts first. Claude's current table identifies 44 GPT and 33 Qwen3 questions with mixed uncapped outcomes (1,237/996 attempts); eligibility must be rechecked after final scorer acceptance. Qwen3.6 has only one such question in that table, so do not invent a broad uncapped correctness contrast for it.

| Run | Purpose, model and benchmarks | Attempts and reuse | Requested resources / GPU-hours | New storage estimate | Dependencies |
|---|---|---|---|---|---|
| C0 CPU | Corrected corpus inventory, outcome joins, fixed-score descriptive analyses; all generation tracks including both Qwen3.5 variants | All existing generation metadata; 1,580 GPT + 1,390 GLM + 424 Qwen3.6 accepted captures and available fixtures. No replay | Proposed CPU Slurm job, 8 CPUs/32 GiB, initially 2h; 0 GPUh | Budget up to 2 GiB for compact derivatives; fail/stage if more needed | Claude scorer acceptance; cache/import/unknown fixes and new frozen bundle; no tracked edits made in this audit |
| G1 | Routing versus correctness at fixed early horizons; GPT, AIME/AMC | Up to 24 mixed-outcome questions × 4 existing attempts = 96. Aim 2 correct/2 incorrect; where only one minority attempt exists use it plus a matched majority and report fewer attempts. Replay only missing prefixes, maximum 8,192 completion tokens each | 2 A100, 3h normal; **6 GPUh** | ~0.65 GiB compressed before contingency | C0; native prefix/token coverage tests and provenance binding; exact missing-token manifest |
| G2 | Independent within-question replication; Qwen3-30B-A3B, AIME/AMC | Same maximum 24×4 and 8,192-token prefix design; reuse exact matching capture fixtures; old 100k annotation completion is not required | 2 A100, 2.5h normal; **5 GPUh** | ~1.55 GiB compressed before contingency | C0 and G1 design acceptance; Qwen3 rate is fixture-based and must be checked on the first attempts |
| G3 | Routing versus truncation/length and Qwen3.6 controller inputs; AIME/AMC first, Olympiad only if repeated-attempt eligibility is verified | Up to 24 questions × 4 existing attempts = 96, pairing capped/uncapped attempts within question. Full saved completion, maximum 32,768 tokens each; reuse any exact overlap with the 424 captures | 2 A100, 15h normal; **30 GPUh** | ~7.40 GiB compressed before contingency | C0; verified cap/uncap support; all-token native validation including capped inputs; new HOME/WORK output manifest |
| C1 CPU | Paired analyses, grouped held-out baselines, prefix class detector and expert/class calibration | G1–G3 plus saved labeled Qwen3.6 features. Partition questions before fitting experts, detector, thresholds or tuning policies | Proposed CPU Slurm job, 8 CPUs/64 GiB, initially 4h; **0 GPUh** | Included in compact-derivative allowance; avoid expanded full tensor copies | Accepted captures and corrected sidecar; adequate label coverage. If new labels are necessary, stage a separately measured small labeling proposal |

**Core capture request ceiling: 41 GPU-hours**, before subtracting overlapping accepted attempts. It would bring campaign commitment to at most 135.365 GPU-hours, leaving 224.635, assuming no other new commitments. This is a proposal, not a reservation or permission to submit. CPU walltimes are initial operational limits, not measured completion promises. Each GPU job first validates exact token IDs/all routed layers/native weights and prefix boundaries on a short fixture, then continues only after those checks pass; stop if the observed rate or output footprint invalidates its cap. New prefix support must first pass bounded CPU tests. Do not reuse a full-completion hash as if it were an independently complete prefix experiment.

The core does not resume four full shards for every model or complete labels irrelevant to the selected contrasts. Keep Gemma, GLM, Nemotron and Qwen3.5 in corrected generation-level tables. GLM's accepted routing features remain available for descriptive replication. Additional paired capture for these models is a later, separately costed extension if the core produces an effect worth confirming or a specific paper claim requires it. Qwen3.5 is included explicitly: native GPTQ/GGUF capture cost and backend acceptance remain unresolved, and an Unsloth/BF16 proxy does not close that gap.

### Statistical interpretation

- For correctness, compare attempts **within the same question** using question fixed effects or conditional models and question-clustered uncertainty/permutation. Cluster/resample by question, not tokens or sentences. Match benchmark and evaluate feature increments over a question/difficulty and prefix-length baseline; estimate difficulty from other attempts where needed. Do not use the target attempt's outcome in its baseline covariates.
- Use prespecified horizons such as 128, 512, 1,024, 4,096 and 8,192. At-risk horizon analyses must explicitly report early completions excluded at each horizon; separately report early-completion outcomes. Recheck mixed-outcome paired support at every horizon, publish the surviving pairs and do not silently substitute between-question comparisons when an outcome arm disappears. Do not quietly drop short or capped attempts. Routing observed before a horizon may predict eventual outcomes, but a paired association does not prove a routing intervention causes accuracy.
- Balanced case-control sampling is for contrasts; it cannot estimate benchmark accuracy or natural error prevalence. Those come from the corrected full generation corpus with unknown/capped counts. Fit preprocessing/feature selection only on training questions; preserve a question-disjoint evaluation and report wide uncertainty if 24 questions is insufficient.
- Prefix reuse requires a prespecified feature inventory. Slice stored IDs/weights/gaps and use only temporal windows whose endpoint is at or before the horizon, binding each derivative to its parent packet and horizon. Full-completion hidden/router summaries, final-length bins and normalized full-sequence position features are not prefix features. If the required prefix hidden/router metric cannot be reconstructed from saved windows, replay that attempt even if a full packet exists.
- For length, capped traces are right-censored at the fixed budget, not known natural completions. G3 is a capped/uncapped case-control sample: its unweighted hazard or restricted mean does not estimate corpus token efficiency. Compute natural cap/hazard/restricted-token summaries from the complete generation corpus; routing-conditioned population estimates need known inclusion probabilities and an appropriate weighted design. Selected within-question pair contrasts are descriptive. Analyze completion hazard/restricted token use and cap incidence; avoid interpreting observed 32K as final natural reasoning length. Evaluate fixed-prefix routing increments over available prefix features; final length is an offline descriptive control, not an online predictor.
- Include prompt/question-feature baselines and label-free routing baselines. Small null effects with broad intervals do not support an all-model absence claim. Expansion must follow a prespecified precision/effect gate, not significance hunting.

## 4. Qwen3.6 Explore/Monitor intervention inputs

The minimum deliverable from C1 is a checkpoint-bound expert/class map with question-level support, a prefix-only class detector with unseen-question calibration, latency/coverage estimates and a clearly specified trigger/action. Existing sentence labels use future sentence context; they are retrospective training targets. A deployable gate must segment and predict from available tokens only, and must be evaluated at the actual first-16-token or later decision point. Do not use retrospective final sentence boundaries, future tokens, corrected outcome, or final length at inference.

Use train/tune/test question splits before selecting experts. Estimate Explore/Monitor associations conditional on position/prefix length and question where possible; report uncertain/distributed expert associations instead of assuming a unique expert per class. Capped unlabeled trajectories can inform length analyses; they do not automatically supply class ground truth. Preserve unknown labels and quantify coverage on capped traces. If a targeted new annotation batch is needed, Claude/Codex must measure judge throughput and propose its token/storage cap before approval; no justification to finish all 9,000 old-cohort records first.

After agreement and implementation, a **conditional engineering validation** can request 2 A100 × 1h normal = **2 GPU-hours**, separate from the 41-hour core. Qwen3.6 measured cold start reaches 21.1 minutes, making a 30-minute debug allocation tight. This run would validate a short saved-token replay, zero-strength identity, correct native dispatch/weights, causal class gating, matched action-rate control and bounded overhead; it cannot demonstrate an accuracy/token benefit. Proposed storage cap 1 GiB. Stop if startup/validation cannot fit. Current code has no Explore/Monitor gate, so this run is not ready to submit.

Actual intervention generation is a later randomized held-out experiment, contingent on viable detector/expert inputs. Compare baseline, class-triggered action and matched random/position trigger at identical sampler/budget; include a simple stopping/forced-answer baseline if token efficiency is the headline. Use paired questions/seeds, report accuracy noninferiority and token use jointly, and count cap/unknown outcomes. Claude must supply measured decode/startup rates and the exact question×seed×arm token ceiling. **Capture throughput cannot price generation**, so no invented GPU-hour estimate or intervention allocation is included in the core.

## 5. Measured capture costs and storage discipline

Source: CODEX_measured_capture_costs_20260924.json, accepted result/complete metadata and saved sizing fixtures. Times below are capture/replay measurements, not generation throughput.

| Model | Evidence | Mean seconds/completion token | Conservative rate for proposal | Observed startup seconds | Compressed bytes/token |
|---|---|---:|---:|---:|---:|
| GPT | 1,580 accepted attempts / 6.140M tokens | .006335 | P90 .008252 | 145–148 | 594 mean; ~877 long-fixture used for storage |
| GLM | 1,390 / 7.550M | .004843 | P90 .005090 | 265–359 | 2,070 mean; ~2,785 long fixture |
| Qwen3.6 | 424 / 8.493M preserved after stop | .009406 | P90 .010790 | 922–1,269 | 2,525 |
| Qwen3 | 15,001-token sizing fixture | .006555 | .006555 plus 1.5 runtime factor; limited evidence | 311 | 2,116 |
| Nemotron corrected native | Four long validation fixtures | .005682 | Sizing only | 394 | 1,409 |
| Gemma BF16-KV sensitivity | Two validation fixtures | .006672 | Sizing only; preserve condition label | 300 | 1,897 |
| Qwen3.5 GPTQ / GGUF | No accepted native capture sizing | unknown | Not costable from proxy | unknown | unknown |

For each two-GPU job, requested walltime = round upward to the next 30 minutes of **1.5 × (measured conservative startup + missing completion tokens × measured rate)**. G1/G2 token ceilings are 786,432 each; G3 is 3,145,728. Requests above include worst-case token ceilings with one startup per model; actual selected-token totals and reusable packets should reduce them. Long-prefix overhead and Qwen3 rate uncertainty require first-attempt remeasurement. Do not launch many small shards that each pay model startup. Two jobs can run concurrently if storage permits, staying under four GPU jobs/two debug jobs; no need to occupy four slots merely because permitted.

Core compressed growth is approximately **9.6 GiB**. With 25% output contingency, 2 GiB CPU derivatives and ~7 GiB single-job transient allowance, stage approximately **21 GiB** of HOME/WORK headroom. These are estimates, not a fresh quota certification or a guarantee of transient size. Before submission, compute exact retained/current bytes, model/cache headroom, maximum per-attempt uncompressed buffers and selected-token counts; serialize if transient concurrency would exceed the existing 200 GiB campaign envelope. Keep HOME for small private state and WORK for bulk outputs. If headroom does not fit, stage the run. No new scratch writes, model deletions or transfers are authorized by this plan. Existing scratch artifacts remain read-only inputs.

## 6. Is larger-budget regeneration needed?

**No regeneration is required to correct the existing scores, reuse captures, or run the proposed historical-sampler 32K-budget within-question/censoring study.** Larger budgets answer a different question: whether a fixed-budget failure would finish correctly, and whether a model-card evaluation condition changes the result. Do not describe 32K results as uncapped model capability.

| Model/benchmarks | Decision and reason |
|---|---|
| Qwen3.6 AIME/Olympiad, with AMC as control | Highest-priority conditional sensitivity. Claude finds cap-dominated failures; local card advises at least 128K context and shows 81,920-output thinking examples. Historical T=.6/k-disabled/PP0 differs from general-thinking T1/k20/PP1.5. If the paper claims recommended-setting performance or budget-caused failure, add a matched subset at recommended sampler with 32K output, then the same sampler with larger output/context. Keep historical sampling as a separate condition. Changing sampler and budget together cannot isolate either effect. |
| GLM AIME/Olympiad | Conditional larger-budget sensitivity for cross-model budget claims: own local card recommends 131,072 max-new-tokens for most tasks; historical output32K/context49,152 cannot reproduce that setting. Sampling T1/p.95 agrees. Extended-context feasibility on A100 must be validated; do not assume it fits current allocations. |
| Nemotron AIME/Olympiad | Cap prevalence can motivate a selected extension if the scientific claim needs it. T1/p.95 matches its own card and 32K exceeds its local 16K example; no documented recommendation forces longer regeneration. |
| Gemma | Historical T=.6/k-disabled differs from restored snapshot defaults T1/k64; historical checkpoint revision remains unverified. Resolve repetition/loop behavior and scorer issues before considering longer budget. No local card establishes a required larger output length. Preserve BF16-KV sensitivity as a distinct condition. |
| Qwen3 AIME/AMC and other benchmarks | Existing T.6/p.95/k20/minp0 agrees with own thinking guidance; 32K matches a local example. No blanket regeneration is needed for the proposed paired study. Extend only selected cap failures if a specific budget claim requires it. |
| GPT | No locally cached sampling/output recommendation establishes a violation. Reuse existing generations; no blanket extension. |
| Qwen3.5 GPTQ and GGUF | Both remain in the study's generation inventory. Own cached recommendation metadata is unavailable, so do not borrow Qwen3.6 defaults. Investigate cap versus repetitive-loop failures on saved outputs first. A native checkpoint/backend and measured long-context generation cost are prerequisites to any targeted extension or native routing claim. |
| MATH500 / Minerva across models | Fix and validate scoring first, especially unit/rounding ambiguity. No blanket regeneration justified by corrupted stored correctness. |

Claude owns exact eligible regeneration IDs, output/context sizes, decode costs and proposed GPU-hour caps. The two conditional larger-budget comparisons above are **staged, unpriced generation proposals**, not omitted costs inside the 41-hour capture budget. Until their measured costs, storage feasibility and paper scope are agreed, there is no honest ETA for all paper results. The core's requested compute path is up to 15 hours for G3 plus CPU work if jobs start together; queue time, implementation/validation and any approved sensitivity/intervention runs are additional.

## 7. Acceptance before proceeding

1. Claude validates extraction/scoring on adjudicated examples, including threaded behavior, nested/multiple boxes, invalid/unknown answers and units; corrected outcome sidecar is frozen and joined by exact content identity.
2. A future authorized code change fixes stale caches, frozen imports and unknown class handling; tests verify changed hashes cannot reuse stale outcomes, workers import correctly, and scores/caps/unknowns reconcile to the full corpus. No tracked code was changed in this audit.
3. Freeze paired/cohort/prefix manifests with explicit inclusion/exclusion counts, question splits, all generation conditions and observed overlap. Full accepted native packets remain immutable. Do not infer provenance merely from model name or same tensor shape.
4. Refresh live account/QoS and HOME/WORK quota, reconcile ledger, obtain agreement on exact proposed jobs and caps. Existing campaign allowance is not authorization to submit during this stop.
5. GPU acceptance requires exact token/layer coverage, native dispatch/weights, no hidden mismatches, checkpoint identity and reliable resume. Scientific acceptance separately requires corrected outcomes, explicit censoring/unknown denominators and question-level inference.
6. Regenerate paper tables from observed accepted artifacts; explicitly list models/conditions without native routing, including Qwen3.5. Historical correctness headlines remain superseded. No monitoring is armed in this stopped phase; follow the agreed monitoring policy only when future submissions are authorized.

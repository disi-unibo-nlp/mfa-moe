# Local MoE analysis package and memory objective: source review, 2026-10-08

Read-only code and artifact review by the publication-gap subagent. The user now wants careful all-seven analysis of correlations, transitions and expert usefulness, with a package that accepts a local MoE and benchmarks. The primary deployment objective is a smaller resident expert-memory footprint, with accuracy measured jointly. This is a review for the ongoing debate, not an adopted replacement protocol or a Slurm authorization. Existing v3 jobs, gates, sources and their finite controller chain remain unchanged.

## Recommendation

Build the package around the analysis machinery that already exists. Reuse all-seven readout, benchmark, class-dynamics and expert-profile evidence before defining another producer. Separate three outputs in its report: **observed utilization/associations**, **out-of-question predictive relevance**, and **measured consequences of changing the model**. A specialized or infrequent expert is a candidate for investigation, not evidence of safe removal.

The new scientific question should be explicit: **At a fixed reduction in resident expert bytes, does a keep-set informed by the measured profiles achieve a better held-out accuracy tradeoff than an appropriate simpler or established ranking?** This connects the user's package to a falsifiable use case. It requires actual resident removal and original-prompt evaluation. The historical force-off study does not supply those endpoints.

The package integration and source-bound figures are plausible work for the October 15 result cutoff. A qualified physical-pruning operator for all seven native checkpoint formats is the main engineering risk. No inspected artifact establishes that all-seven modified-model evaluation will finish within a week. This review does not substitute a negative-paper fallback for the user's objective.

## Reusable implementation

| Need | Existing implementation | Limit to preserve |
|---|---|---|
| Local checkpoint and saved benchmark trace input | `models/loader.py:load_model_and_tokenizer`, `correlation_pipeline/extract.py:extract_all`, `schemas.py:TraceRecord`; `--model-id`, `--revision`, `--local-files-only` | Replay checkpoint/precision and generation checkpoint/precision are distinct provenance fields. |
| Native routed layers, actual selected IDs and dispatched weights | `models/router_adapters.py:stream_router_signals` | Supports the seven architecture families; full scores are not available for every native selector. Preserve groups, shared experts, sigmoid/softmax and learned scales. |
| Analysis without semantic labels or external services | `correlation_pipeline/dynamics/cli.py:analyze` modes `class-only`, `routing-only`, `combined` | Missing labels remain missing. Retrospective automated episode labels are not latent cognitive states or an online detector. |
| Occupancy, load, entropy, overlap, turnover and temporal windows | `dynamics/routing.py:RoutingReducer`, `dynamics/experts.py` | Selection frequency sums to native k; normalized selected-weight occupancy sums to one. They are different quantities. |
| Episode transitions, runs, dwell and re-entry | `dynamics/classes.py:class_dynamics`, `sentence_table` | Saved token/annotation alignment is required; missing labels break adjacency; truncation censors duration. |
| Matched expert associations and candidate cards | `dynamics/contrasts.py:estimate_contrast`, `candidate_cards` | Train-fold candidate selection, held-out effects, support and multiplicity already exist. Associations do not establish necessity or beneficial activation. |
| Question-grouped predictive comparisons | `dynamics/evaluation.py`, `dynamics/prospective.py:evaluate` | Fits preprocessing inside folds; prefix-state features require cross-fitted prefix-only labels. Final-trace length is not a valid online feature. |
| Inventory, global question folds, extraction, costing and analysis | `dynamics/steering_cli.py`, `dynamics/contracts.py`, `dynamics/campaign.py` | These stages do not dispatch jobs. The existing reserved A fold is explicitly `exploratory_previously_inspected`, not untouched confirmation. |
| Existing v3 outcomes, unknown bounds and paper exports | `verification_study/analysis.py`, `closeout.py`, `paper.py`; versioned helpers under `scripts/paper/` | Reuse sealed versions. New memory work must not edit the pending campaign's source closure. |

The installed entry points in `pyproject.toml` already include `correlation-generate`, `correlation-forward`, `correlation-analyze`, `correlation-dynamics` and `moe-guiding`. A new executor or another general analysis engine is unnecessary.

Minimum justified integration code is a capability-aware local-model/benchmark manifest adapter, a consistent report schema over existing results, a prospective keep-set builder, and explicit links to optional evaluation artifacts. `model_profiles.py` currently uses name patterns and a Qwen-like custom fallback; an arbitrary local snapshot directory must instead resolve architecture from its config and reject unsupported operations. The hard-coded checkpoint/card paths in `model_profiles.py` and `verification_study/registry.py` are reusable project bindings, not yet a portable public API.

A small public surface can expose three operations over the existing modules:

1. **Inventory/analyze:** local checkpoint config and parameter index, benchmark records or saved generations, optional route captures and saved labels → capability manifest, family-keyed provenance table, routing/state/score analysis and figures. Required new glue resolves checkpoint architecture, inventories available signals and translates benchmark inputs into the existing `TraceRecord`/inventory schemas. Existing generation/extraction is invoked only for a specifically missing authorized input.
2. **Propose a keep-set:** declared resident-byte target, training-family split and ranking rule → original expert IDs, per-layer quotas, byte accounting, support/overlap/stability and source hashes. This is a new selection layer over existing profiles; it does not apply the set or claim benefit.
3. **Evaluate a keep-set:** an explicitly supported resident-loader adapter, frozen manifest and independently held-out benchmark → actual resident memory, assigned accuracy/unknown outcomes, paired family intervals and a joint tradeoff report. This is optional until the new loader/evaluator is qualified. Unsupported formats yield a precise capability result rather than a Qwen fallback.

The canonical storage manifest should distinguish routed experts, shared experts, routers and dense weights; name actual packed tensors, quantization metadata/scales, shapes and physical bytes; deduplicate aliases/shared storage; and record tensor-parallel ownership. Its estimated removed bytes remain **projected** until verified against the runtime allocations. A parameter count multiplied by a nominal dtype is inadequate for mixed or packed quantization.

## Reuse before a new correctness or transition fit

All-seven present-class readout is already complete in the source-linked CLEAN artifacts. Its current-label and lexical-null controls should be exported, not rerun merely to obtain a fresh package receipt. Next-class anticipation is already measured for GPT and Qwen3.6. Missing equivalent prepared anticipation caches for five models does **not** mean that their observational class transitions or route profiles are absent.

Correctness prediction is also not a blank cell. The clean GPT R3-E B4 comparison is documented in `report/experimental-resume-v1/R3E_B4_PRECISION_READOUT_v1.md`: 508 scored questions in 495 families; primary log-loss change −0.000729 nats/question, nominal interval [−0.005313,+0.004384], with a separate five-comparison simultaneous interval and explicit precision-study limitations. The primary and confirm-family-exclusion sensitivity are different populations. The failed rejection is not accuracy equivalence or a causal null. An all-seven correctness extension must name the model/feature/population cell that B4 does not already answer.

For the package, join saved outcomes and profiles on model, benchmark, source question/family and attempt, retaining source scoring rules, caps and cohort selection. Analyze A and selected B separately; do not pool B's outcome-enriched sampling as an unbiased population. Evaluate prospective ranking/feature choices inside training folds. Report effect magnitude and family uncertainty, not only a list of significant expert IDs. Use common family resampling for related model comparisons and model-specific conclusions rather than treating seven checkpoints as seven independent samples.

The parallel audit now provides a reusable bound inventory under `report/model-analysis-review-20261008/`. I read `INVENTORY.json`, `expert_profile_inventory.csv`, and `expert_usage_correctness_saved.csv`. The seven A cohorts share the same 1,547 question IDs; the B union contains 51, with only one question common to all seven. Its profile inventory points to each model's saved `E_class_experts.csv` and full `class_counts.npz`. The audit reports counts indexed by 1,547 attempts × nine class scopes × routed layers × experts, with attempt IDs joined to `attempts.parquet`. This is a much cheaper starting point for train-only frequency/state keep-set selection than raw route extraction. Use the full counts to select inside the training fold; reusing a specialist list chosen on the complete cohort would leak selection across folds.

The saved expert-usage/correctness table already supplies 147 adjusted descriptive rows. For example, Qwen3.5/Qwen3.6 Analyze-specialist whole-trace usage has small positive adjusted correlations (about 0.092/0.082), while Verify-specialist usage has negative correlations (about −0.097/−0.101). These are same-cohort observational results with length/difficulty/class-fraction adjustment, not necessity estimates. They should inform the report and candidate hypotheses rather than be regenerated as a supposedly new finding.

## Actual prior deactivation evidence

The completed deactivation branch is real and should be reused:

- `report/experimental-resume-v1/CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v4.json` (seal `85b7f90ae7295d854d32aa3bee935a5feeb53ee8810cc6657b34ce40028eeb3d`).
- `CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_SEMANTIC_ANALYSIS_v2.json` binds that manifest, generation, blinded reader and execution-position artifacts.
- `scripts/experimental_resume/run_eligible_deactivation_serial_v4.py` and its preserved predecessors implement the historical operator.

It contains 13 Qwen discovery families, two seeds and six counterbalanced arms: native, duplicate native, target negative bias, target force-off, random negative bias and random force-off. There are 156 assigned branches and 144 valid reader pairs. Target force-off minus native immediate semantic success is −3.85 percentage points, approximate simultaneous interval [−23.08,+11.54]; versus random force-off it is −3.85 points [−19.23,+15.38]. The target negative-bias contrast is +7.69 points [0,+23.08] under that historical endpoint. Every arm emits 6,656 tokens over 26 branches: all reach the 256-token cap. These are not answer-accuracy measurements or useful output-length savings.

The implementation masks routing in the full model. It does not remove resident expert tensors. Its small same-prefix population and historical reader estimand cannot establish the new original-prompt memory/accuracy claim. Do not restart this completed branch. Likewise, preserve M9's historical hold and diagnostic outcomes; they are not a reason to impose prohibited cross-execution token/route identity tests on a newly specified operator.

## What actual resident pruning still needs

The active v3 backend deliberately permits only positive singleton pulses (`verification_study/backend.py`, `adapters.py:NativeCapture`). It cannot certify negative masks or physical pruning. No selective resident-expert loader/exporter was found in the current source tree or the existing `study/qwen36-routing` branch.

The existing NF4 adapters demonstrate possible building blocks, not a ready pruning path. `models/nemotron_quantized.py` creates every expert and asserts all expert weights are on CUDA. It requires a BF16 source for NF4 replay while the generation profile may use NVFP4. `models/gemma_quantized.py` also materializes every expert and explicitly reports its replay quantizer as nonserializable. A fused tensor with removed rows replaced by zeros has unchanged allocation size and does not meet the memory objective.

A separately bound operator must declare the deployment checkpoint/format, retain the original-to-resident expert-ID mapping, preserve quantization scales/packing, and implement valid surviving selection for every native group/k/shared-path rule. It must omit or release actual routed-expert allocations. All-keep same-state checks, valid k/group constraints, within-execution capture and lifecycle checks are necessary. Exact sampled sequence replay across separate executions is not a requirement. A CPU-only shape mock cannot prove the GPU allocation or selector behavior.

The parallel audit has also completed the header-only storage collection: `report/model-analysis-review-20261008/checkpoint_header_inventory.py`, `CHECKPOINT_HEADER_INVENTORY.json`, `checkpoint_tensor_bytes.csv` and `checkpoint_expert_tensor_bytes.csv`. Reuse these rather than commission another collector. Native routed experts account for approximately 68.4–95.0% of serialized tensor bytes across the seven checkpoints, with model-specific quantization and draft paths. This establishes a substantial potential storage target, not realized memory savings. The GLM index declares 31,221,488,576 bytes while its headers total 62,442,983,168; retain and resolve this discrepancy before treating the index as authoritative. Header bindings do not attest runtime residency, tensor-parallel packing or a selective loader.

Measure routed-expert resident bytes and whole-model loaded bytes, then CUDA allocated/reserved and process-level memory at fixed context/KV capacity, precision, device placement and concurrency. vLLM can consume freed weight memory with a larger KV pool, so a process-memory screenshot alone can hide weight savings. Serialized checkpoint bytes alone do not demonstrate runtime savings. Latency and throughput may be secondary; they do not gate a claim restricted to resident memory and accuracy. Fixed native k also means lower resident memory does not automatically reduce active compute.

## Baselines and what saved captures can support

Frequency and selected-weight occupancy are directly available in the saved compact routes. State/benchmark/position-balanced coverage and training-fold profile stability can also be derived without generation when their labels and joins exist. Use the same per-layer quota and removed-byte budget when comparing rankings; otherwise a layer-allocation change is confounded with an expert-ranking change. Preserving rare states is a selection hypothesis, not proof those experts are necessary.

The parent's primary-literature audit identifies output-aware pruning as an important comparator. No REAP/MoNE or per-expert output-norm implementation was found in `src`, `scripts` or `tests`. Compact native captures contain IDs, selected weights, boundary gaps and token identities, not expert output vectors. `correlation_pipeline/forward_audit.py` explicitly describes its raw signals as router affinities/log probabilities and decoder inputs, **not expert-output activations**. `extract.py` only saves raw hidden/router tensors when requested and sets `save_expert_weights=False` in that replay pipeline.

Therefore an output-norm/variance ranking cannot be reconstructed from the compact routes alone. Audit retained raw decoder inputs before requesting compute. If a relevant strong baseline still lacks necessary activations, a bounded teacher-forced forward over existing training token sequences could produce them; new natural generations are unnecessary merely for ranking. It remains model compute, needs output-hook code and a price, and must match the evaluated checkpoint/precision/routing contract. Teacher-forced likelihood may supplement diagnosis but cannot replace free-generation accuracy of the pruned model.

Frequency plus random alone can support a limited diagnostic use-case comparison. It cannot support a broad claim of state-of-the-art pruning against methods never evaluated. The exact stronger baseline should be chosen from the parent's verified literature/adapters before freezing a new claim.

## Workload components, conditional on reuse and pricing

These describe the complete missing-cell path, not four immediate jobs or dispatch-ready cards. The two concrete next cards are specified below. None authorizes a job. A complete dispatch card still requires measured role/context cost, finite resources, live entitlement and explicit workload authorization. CINECA skill and its on-cluster/scheduler references were read for this review; no account query, allocation or submission was made by this subagent.

### A. Complete source-bound analysis and keep-set selection

- **Claim/artifact:** all-seven expert-profile and transition stability table; incremental correctness-prediction forest plot only for genuinely missing cells; prospective candidate/keep-set JSON and coverage-risk figure.
- **Reuse and missing input:** import completed CLEAN readouts, B4 and existing benchmark/dynamics outputs. Audit unresolved per-attempt joins and exact optional feature fields. Do not refit a completed cell to change its provenance. Fit only a missing comparison that the paper actually needs.
- **Controls/units:** question-family folds, matched frequency/layer/position controls, past-only features for prospective claims, A/B population separation, training-only feature/rank selection. Family bootstrap; simultaneous model/contrast family declared. Token rows increase feature resolution, not independent n.
- **Price/duration:** existing `dynamics/campaign.py:measured_resource_estimate` derives extraction prices from timed short/long samples and binds output bytes/RSS. A precise price for a new combined fit was not located. Reporting joins may remain bounded CPU work; heavier fitting belongs in a separately priced CPU card. Target October 8–10 only after actual input/price inspection, not a completion guarantee.
- **If imprecise:** retain unstable rankings and intervals in the package. Do not promote them to safe-deactivation claims or launch a new population merely to obtain significance.

### B. Output-aware training baseline, only if necessary

- **Claim/artifact:** fixed-budget comparison of a profile-informed ranking with an appropriate output-aware baseline; training-only expert-importance table and bound keep-set manifests.
- **Reuse and missing input:** retained token sequences and any raw decoder-input tensors; missing per-expert output statistics. Skip this card if a compatible saved statistic or validated existing baseline implementation supplies the cell.
- **Controls/units:** identical training families/tokens, native format and routing, per-layer quotas/removed bytes; aggregate within family before reporting ranking stability. No evaluation answers or test route profiles in selection. No external validator needed.
- **Price/duration:** no exact seven-model output-hook replay price found. Existing inference or HF replay prices are only context until code, checkpoint, sequence lengths and memory are compatible. A minimal retained outcome-blind price measurement would need its own authorized bounded envelope. It must complete early enough to freeze selection before final evaluation; otherwise this baseline claim remains untested.
- **If weak or similar:** profile information has not improved the measured ranking criterion; still compare the frozen rules if the empirical accuracy/memory question has adequate support. Do not tune on test outcomes.

### C. Physical-memory operator qualification and retained role price

- **Claim/artifact:** per-model engineering table proving actual resident-expert removal, mapping/selector fidelity and realized memory at the declared deployment configuration.
- **Reuse and missing input:** reuse seven-model capture geometry and existing loaders; missing selective allocation/export and runtime proof for each format. Existing positive-pulse PASS is not a pruning PASS.
- **Controls/units:** all-keep within-execution reference and structurally valid keep-set, same precision/device/KV settings, routing masks paired with actual omitted allocations. Cover each supported architecture and quantization path; fixtures are engineering cases, not statistical evidence of accuracy retention.
- **Price/duration:** no physical-pruning role price exists in the inspected artifacts. `DISCOVERY_PRICE_CLOSEOUT-r4.json` gives seven measured native pilot paths and 34.178 total actual billing core-hours; individual pilots range 1.893–8.196 core-hours with different output lengths. The historical deactivation price v4 used 156 short requests, a 685-second cold load, 7,587 estimated seconds and a 3-hour/two-GPU upper allocation. Neither is a transferable price for the new loader. Require a finite measured qualification/retained-price card before production. Implement and qualify by roughly October 10–11 for the one-week path to remain plausible; missing support at that date cannot be hidden by a routing mask.
- **If failure:** isolate the missing architecture/operator capability and report it as unqualified. Do not claim memory benefit or drop that model from the seven-model scope. No automatic new actuator search.

### D. Held-out original-prompt accuracy versus realized memory

- **Claim/artifact:** all-seven paired accuracy/memory table and frontier point(s) for frozen keep-sets. Required arms: full model, profile-informed, frequency-only and matched random; include the chosen stronger baseline if the paper claims improvement over it. One meaningful footprint budget first, chosen for the user's deployment target, not a sweep selected for a positive result.
- **Reuse and missing input:** native generations can be reused only for the exact same independently reserved questions, prompt/decoding/checkpoint/runtime endpoint; no such completed modified-model result was found. New pruned-model generation is necessary. Existing conditional continuations are a different population.
- **Controls/units:** equal removed bytes and layer quotas, native k/group rules, original prompts, paired question/seed assignment, gold-based offline grading, caps/no-answer/unknown outcomes retained. Random keep-set seeds are declared before evaluation; one fixed random set supports comparison with that set only, not the expectation over every possible random pruning. Across-model shared-family resampling; seeds averaged within family. No retention margin is required to estimate a joint frontier; a retention claim would require a separately chosen meaningful margin and sufficient precision.
- **Held-out boundary:** `routing-study-v1/INDEPENDENT_FAMILY_MANIFEST.json` reserves 32 validation/96 final full-MATH families without outcomes, with transitive duplicate exclusions. This is a candidate source only after reconciling its existing campaign reservation and verifying actual non-use. Do not silently repurpose its sealed v3 roles. The previously inspected dynamics reserve is not an untouched test set.
- **Price/duration:** exact modified-model original-prompt role prices, assignment count and footprint target are missing. At 96 families and two seeds, four arms imply 5,376 runs across seven models before native reuse; five arms imply 6,720. This arithmetic is not a recommended sample or a resource estimate. Runtime must include cold loads, prefill, declared output caps, shutdown, necessary retries and fixed measurement overhead. Production is justified only if the measured envelope and independent precision can finish before the agreed cutoff; queue estimates alone cannot guarantee completion.
- **If adverse/imprecise:** report the actual accuracy cost or unresolved interval at the achieved memory reduction. This does not prove safe removal. Keep the frozen result; do not adjust the budget, ranking or evaluation set in response and rerun until favorable.

## The two concrete next cards

**Card 1 — all-seven saved-data retention ledger and package report.** Reuse the new `report/model-analysis-review-20261008/` joins, all-seven `class_counts.npz` sources, saved profile/readout outputs and header inventory; emit one seven-model availability/storage table, source-bound expert/transition cards, and frozen training-only keep-set candidates at the selected deployment budget. The remaining work is integration and fold-specific ranking, not another header inventory or raw capture producer. Resolve the GLM header/index discrepancy and bind actual duplicate families. Add an explicit field stating whether the exact output-aware baseline inputs are present; do not manufacture its score from counts. This is the immediate analysis/integration task and uses no new generation or external annotation. The family's effective support and expected accuracy-interval width are a required output before sizing the later evaluation. If a needed fit exceeds bounded local analysis, price that exact CPU substage with `measured_resource_estimate`; no measured full-fit budget was found here. Desired completion is October 8–10, subject to the existing data joins. Incomplete joins and unstable candidates remain explicit; they do not trigger new caches automatically.

**Card 2 — actual resident-loader qualification with a retained outcome-blind price.** After Card 1 identifies format and byte layout, select the first implementation path on engineering compatibility and qualify a real keep-set: original-ID mapping, actual omitted allocations, native selector/group/k/shared paths, all-keep same-state behavior, and fixed-KV memory telemetry. Complete all seven capability cells, with engineering order distinguished from scientific scope. Existing seven-model native fixtures and historical loaders are inputs, not sufficient qualification. This card's primary artifact is a measured loader/memory capability table and a role/context price receipt; it must not claim accuracy usefulness from its engineering prompts. Exact GPU shape and walltime are currently unpriced and therefore not ready for authorization. Use the saved native price paths to bound a minimal proposed measurement, then obtain explicit workload/resource approval. Target initial qualification by October 10–11 so later original-prompt evaluation remains possible; the seven heterogeneous formats make completion uncertain. A failed format stays unsupported and is not replaced by mask-only memory claims.

Held-out accuracy production is the dependent D stage. It should be turned into finite reviewable cards only after the selected footprint target, baseline implementation, actual held-out support, runtime qualification and measured price exist. This avoids disguising unpriced final runs inside an engineering card. It also avoids promising a complete seven-model frontier by October 15 before knowing the operator's cost and support.

## Precision: 96 families is a reservation, not a justification

`routing_control/study_v1.py:freeze_independent_families` implements deterministic stratified 32/96 sampling without outcomes. It contains no expected-width or power calculation. No inspected v3 artifact supplies one for the proposed memory endpoint.

For illustration only, with 96 independent paired binary outcomes, small true difference, and fraction q of pairs with discordant accuracy, the usual normal planning half-width is approximately `1.96 sqrt(q/96)`. This is not a fitted power analysis or a substitute for the eventual family-resampling contract:

| Assumed discordance q | Single-comparison approximate 95% half-width | Approximate half-width with Bonferroni over 21 contrasts (z about 3.04) |
|---:|---:|---:|
| 0.10 | 6.3 percentage points | 9.8 points |
| 0.25 | 10.0 points | 15.5 points |
| 0.50 | 14.1 points | 21.9 points |

Two seeds do not create 192 independent families. Missing/unknown grades further widen identification bounds. Use saved compatible pairs to estimate plausible family variance and a range of discordance/missingness; then choose a claim whose precision is supportable. Do not infer “no accuracy loss” from a nonsignificant estimate or from a degenerate interval. The earlier B4 simulation already illustrates why nominal bootstrap coverage needs scrutiny.

## Adjudication of DeepSeek Flash/max review 1

The saved review `deepseek-analysis-first-20261008.md` usefully separates observations, predictive evidence and intervention, requests nuisance controls and grouped evaluation, and insists that masks do not save parameter memory. Several recommendations need correction before adoption:

1. Its proposed correctness analysis must first reuse B4 and other completed cells. It is not a reason for a generic all-seven rerun.
2. Its missing five-model transition-cache statement applies to equivalent clean anticipation, not absence of all observational transition evidence.
3. Stability of an associated expert does not itself authorize pruning or establish usefulness. A frozen selection rule can be evaluated prospectively even if an individual profile interval is weak; its empirical utility is decided on held-out outcomes.
4. A noninferiority margin is required for a retention claim, not for the user's current joint accuracy/memory estimation. Latency percentiles and throughput are optional secondary endpoints here.
5. “Automated labels” alone does not prove circularity. The real questions are shared input, label/predictor construction, nuisance controls and whether the endpoint is the saved visible episode class or an independently claimed latent state. No external human labeler or validator is imposed by this design.
6. This review does not adopt its proposed narrower fallback thesis. The action is to integrate the existing positive evidence and identify the smallest independently evaluable missing memory-usefulness test, with all seven models retained and no assured outcome by October 15.

## Adjudication of DeepSeek Flash/max review 2

The saved `deepseek-memory-design-20261008.md` correctly recenters the resident-memory objective, rejects unnecessary latency/noninferiority gates, adds an actual parameter/storage manifest, and emphasizes the loader and measured-price risk. Its suggested source-audit → frozen keep-set → resident evaluation structure is compatible with the three-operation package above. Its eventual one-checkpoint evaluation is an engineering staging possibility, not fulfillment of all-seven causal scope.

Three additional limits need correction:

- **Static selection does not require prefix-only training labels.** A once-frozen keep-set may use retrospective labels on training trajectories because it does not observe a test trajectory's future at deployment. Prefix-only and cross-fitted state readouts are required when the deployed routing constraint changes in response to an online estimated state. These are different policies.
- **Training correctness is not prohibited information.** It can legitimately enter a declared utility-informed ranking and be studied against a state-only ablation. Held-out evaluation correctness and future test data cannot enter selection. For the narrow question of incremental state information, keep the competing algorithms' access to training data comparable and make the score explicit.
- **B4 is a specific GPT comparison; frequency and REAP are distinct methods.** Reusing B4 does not fill every all-seven correctness cell. Likewise, a frequency-only baseline must not be called “REAP-style” without the exact output-aware statistic and a faithful implementation. Decide which missing cells are necessary for the claimed contribution before computing them.

The review's requirement that a state association must already beat all nuisance tests before any prospective keep-set evaluation is stronger than logically necessary. It is useful diagnostic evidence; causal usefulness is established by the frozen held-out comparison. State-associated rarity can motivate a training rule without proving an expert's necessity in advance. That distinction should remain visible in the package.

No scientific/operational source was edited and no job was submitted during this review.

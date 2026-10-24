# Existing data and experiments

**October 14 update:** [ADR23](DECISIONS.md#adr-20261014-23-authorized-final-r5-continuation-and-imported-state-validation)
authorizes the final separately sealed r5 continuation. Its
[committed seven-model import](../../report/research-campaign-v3-20261014-r5-r2/EVIDENCE_IMPORT.json)
validates all seven repaired derivations and complete GPT 81/Nemotron 77
coordination banks; 741 original pairs remain across five models. Every original
saved-sequence measurement manifest and the separate non-REAP memory interface
now exist. These deterministic reconstructions use the D01/D04/D05/X06/X11/X12
sources below and do not supply new resident-memory or causal measurements.
Four existing REAP qualifications are reused; corrected Nemotron is qualified,
corrected GPT remains a method limitation and corrected Gemma is pending.
Original sequences, tolerances and historical verdicts are preserved. Full native contexts and complete
measured production admission remain missing. The actual validation recovery
passed and the protected preparation/qualification chain is released. See
[current status](CAMPAIGN_V3_STATUS.md) for source bindings and exact job ownership.
Earlier campaign snapshots below retain their original dates and scope.

**October 12 update:** r3 stopped on storage admission and then failed terminal
paper export. GPT's 81 coordination pairs and Nemotron's repaired summaries plus
55 of 77 pairs remain reusable; new physical-memory/accuracy and causal effects
remain unmeasured. [The recovery review and plan](NEXT_EXPERIMENTS.md#results-first-recovery-review--2026-10-12)
supersede older running descriptions below. [ADR22](DECISIONS.md#adr-20261012-22-authorized-results-first-r4-implementation-and-protected-launch)
records explicit approval to implement and launch the separately bound r4
continuation. Validation and protected ownership are recorded on the
[status page](CAMPAIGN_V3_STATUS.md). Catalogue entries retain their original
populations and evidence boundaries.

**October9 update:** ADR15/16 authorized the separate revised campaign after
the historical stop below. V2 has now exported a terminal incomplete report;
the [ADR18 audit](NEXT_EXPERIMENTS.md#campaign-v2-audit-and-final-decision-requirements--2026-10-09)
adds seven completed ownership caches, exact-attempt native-accuracy summaries,
six retained GPU pilots and their limitations. It maps every remaining stage to
the D/X catalogue below. There are no live scientific jobs; the remaining owners
wait for the October15 snapshot. The October8 inventory and source scopes remain
preserved, rather than implying that all newly authorized work is still running.

**All experiments remain stopped.** [ADR12](DECISIONS.md#adr-20261008-12-user-directed-stop-and-experimentclaim-review) and the [verified stop receipt](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json) supersede earlier execution authorization. This catalogue is documentation and bounded evidence inspection as of October 8, 2026. Cancellation leaves unfinished measurements incomplete; it does not turn them into scientific failures or null effects.

We already have two complete seven-model benchmark corpora, saved solutions and scores, native routing captures, sparse seven-class sentence annotations, clean routing readouts, historical interventions, and completed expert-selection/transition analyses. New v3 discovery has six 48-receipt rosters and 42 retained Qwen3.5 receipts. Five saved recognition qualifications fail; GLM recognition and Qwen3.5's full discovery/audit are unfinished. Actual resident-memory reduction and its joint prospective accuracy remain unmeasured.

This is the main **what exists / what was done / where to find it** reference. [Findings](FINDINGS.md) interprets effects; [sources](SOURCES.md) records precedence; [future-work contracts](NEXT_EXPERIMENTS.md) govern proposals. Job details stay in their existing receipts. A file's presence, validated execution, and a scientifically qualified result are separate statements.

| Find | Direct entry |
|---|---|
| Existing solutions, reasoning text, gold answers and benchmark scores | [Historical generations](#d01-historical-benchmark-generations), [card-profile generations](#d02-card-profile-benchmark-generations), [scores](#d03-scores-and-historical-cap-continuations) |
| Existing Verify sentence labels | [Seven-class annotations and A/B cohorts](#d04-ab-cohorts-and-seven-class-sentence-annotations) |
| Token IDs, routing captures, expert counts and common sentence population | [Routing and analysis archives](#d05-routing-captures-count-archives-and-common-population) |
| Historical intervention solutions and dense labels | [A/B/C/FRESH collections](#d07-historical-intervention-continuations-and-dense-labels) |
| New checking judgments and discovery text | [V3 discovery and candidate-specific judgments](#d08-v3-discovery-and-candidate-specific-judgments) |
| Completed selection, held-out coverage, stability, transitions and bytes | [Selection](#x11-family-held-out-expert-selection-and-stability), [transitions](#x06-state-transitions-and-native-boundaries), [bytes](#x12-checkpoint-byte-accounting-and-resident-memory-boundary) |
| Existing cross-layer coupling, dominant-expert paths and path-based prediction | [Depth/path studies](#x13-cross-layer-routing-and-depth-path-studies) |
| What can be answered without new generation | [Reuse catalogue](#what-can-be-reused) |

## Available datasets and saved outputs

### Locations and identity conventions

Repository links are relative to this wiki. Three private WORK roots recur below:

| Alias | Exact root |
|---|---|
| `W` | [Saved analysis workspace](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/) |
| `R` | [V3 result root](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/candidate-verification-v3-20261007-r4/) |
| `H` | [Historical routing-control results](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/) |

Expand aliases literally; they are path abbreviations, not environment variables. Checkpoint names, routed geometry and original contracts are in [model coverage](MODEL_COVERAGE.md#existing-native-data-and-routing-contracts), [model profiles](../../src/moe_exp/correlation_pipeline/model_profiles.py) and [the v3 registry](../../src/moe_exp/verification_study/registry.py). The seven keys are `gpt`, `nemotron`, `qwen35`, `qwen36`, `gemma`, `glm`, `qwen330b`; historical outcome tables may call Qwen3.5 `qwen35gptq`.

Use `corpus + model + dataset + problem_id/sample_id` to distinguish attempts. The canonical question join is `dataset|source_problem_id`, falling back to the unsuffixed problem ID where required by the original producer. `trace_sha256` identifies emitted text/token records, `attempt_id` identifies a capture, and a duplicate-family ID groups related questions. Never join two corpora solely on question ID and assume identical text, tokens or routes. Sentence identities add original `sentence_index` and character offsets; prefix identities add source receipt and prefix-token digest. Reader draws add the original UID and draw/seed.

### D01 Historical benchmark generations

**Question/population:** original native benchmark performance and reasoning. All seven models, six benchmarks, **1,547 questions and 4,647 attempts per model**; 32,529 attempts in this corpus. There are 30 AIME24, 30 AIME25 and 40 AMC23 questions, each with 32 samples, and 500 MATH500, 272 Minerva and 675 Olympiad questions, each with one sample. These are 42 model/benchmark cells, not 32,529 independent questions. Counts are bound by [the existing inventory](../../report/model-analysis-review-20261008/INVENTORY.json) and [84-cell export](../../report/model-analysis-review-20261008/benchmark_cells.csv), which separates both corpora.

Each root below contains `<benchmark>/traces.jsonl`; all six files per model were found in this inspection.

| Model key | Historical generation root |
|---|---|
| `gpt` | [GPT-OSS](../../results/correlation_pipeline/gpt-oss-20b/generation/openai--gpt-oss-20b/) |
| `nemotron` | [Nemotron](../../results/correlation_pipeline/nemotron-nvfp4-dspark/generation/nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/) |
| `qwen35` | [Qwen3.5](../../results/correlation_pipeline/generation/Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/) |
| `qwen36` | [Qwen3.6](../../results/correlation_pipeline/qwen36-35b-a3b-fp8/generation/Qwen--Qwen3.6-35B-A3B-FP8/) |
| `gemma` | [Gemma](../../results/correlation_pipeline/gemma-nvfp4-nf4/generation/nvidia--Gemma-4-26B-A4B-NVFP4/) |
| `glm` | [GLM](../../results/correlation_pipeline/glm-4.7-flash/generation/zai-org--GLM-4.7-Flash/) |
| `qwen330b` | [Qwen3-30B](../../results/correlation_pipeline/qwen3-30b-a3b/generation/Qwen--Qwen3-30B-A3B/) |

**Format and joins:** JSONL `TraceRecord` records contain `prompt`, `gold_answer`, `model_answer`, `cot_text`, `steps`, `is_correct`, dataset/problem/source/sample IDs, and metadata with assistant/reasoning text, generation configuration, finish reason and token replay. Sampled records in all seven roots retain prompt/completion token IDs, tokenizer digest and completion offsets. Per-record generation metadata is authoritative; older `is_correct` fields are not the uniformly rescored outcome sidecar.

**Sampling/caps and labels:** inspected original records use a 32,768-token output cap and differing model settings. Stratified historical label exports are also retained as [final-label ZIP](../../data/labels/mfa-moe-labels-final-2026-09-18.zip) and [stratified-label ZIP](../../data/labels/mfa-moe-labels-stratified-2026-09-21.zip); archive members and coverage were not re-extracted here. They are distinct from the card-profile A/B labels in D04. Historical token/text availability does not prove compatibility with a newer runtime or capture. Producing/source locator: [original scanner](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/scan_corpus.py), [annotation report](../../paper/6a510c6db0d9b08aa438c142/stratified_label_results.tex).

### D02 Card-profile benchmark generations

**Separate complete corpus:** again all seven models, the same six benchmarks and 1,547 shared questions / 4,647 attempts per model, **32,529 attempts**. Do not add these attempts to D01 as independent problems or call the two native profiles an equal-budget comparison.

**Location:** [card-v3 generations](../../results/correlation_pipeline/card-v3/) at `<model>/generation/<served-model-directory>/<benchmark>/traces.jsonl`. The model directories are exactly the seven keys above. They use the same solution/reasoning/token-replay JSONL schema as D01. [Existing source bindings](../../report/model-analysis-review-20261008/SOURCES.json) and `W/campaign-v3/provenance/` retain provenance; [model_profiles.py](../../src/moe_exp/correlation_pipeline/model_profiles.py) binds checkpoint revisions, sampler, quantization and speculation.

**Sampling/caps:** 32 draws/question on AIME/AMC, one on the other benchmarks. Five profiles allow 131,072 output tokens; Qwen3-30B allows 38,912 within context 40,960; GPT is context-bounded without the same explicit output cap. Temperatures, top-k, context and checkpoint formats differ. Full benchmark scores exist, including **33 unknown adjudicated outcomes**; sentence labels/routing cover selected A/B attempts and sampled sentences (D04/D05), not every benchmark attempt or token. No generations were rerun for this catalogue.

### D03 Scores and historical cap continuations

**Scores:** the [immutable outcome manifest](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/outcomes/outcomes_v3-99bf8d59506e2893.manifest.json) names `W/campaign-v3/outcomes/outcomes_v3-99bf8d59506e2893.parquet`, **68,129 rows**: 32,529 historical + 32,529 card-v3 + 3,071 continuation records. It contains corpus/model/dataset/problem/source/sample IDs, completion length, termination, strict/adjudicated correctness, adjudication status, Minerva-tolerance sensitivity and scorer contract. It is a compact outcome table, not the saved solution text.

**Results:** [benchmark cells](../../report/model-analysis-review-20261008/benchmark_cells.csv), [model totals](../../report/model-analysis-review-20261008/benchmark_models.csv), and [saved original outcome analyses](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/results-v3/outcomes-99bf8d59506e2893/) (`accuracy.csv`, `budget.csv`, `continuation.csv`, `errors.csv`, `length_gap.csv`). Original question-bootstrap intervals and strict/adjudicated scored denominators are preserved. The earlier scorer's [98/100 acceptance](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/acceptance/j1.json) is separate from v3 verification's later 97/100 FAIL.

**Continuation population:** [saved capped-parent continuations](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/continuation/) contain `<model>/plan.json`, `continuations.jsonl`, `scored.jsonl`, `scored_summary.json`. These extend historical capped parents toward a cumulative 131k budget; they are neither fresh benchmark attempts nor the short A/B/C/FRESH intervention population. Join by model/dataset/parent problem and sample IDs. Selection conditions on historical truncation; annotation/routing compatibility beyond the saved records is uninspected. Code: [outcome-sidecar builder](../../src/moe_exp/correlation_pipeline/outcomes.py), [continuation producer](../../src/moe_exp/correlation_pipeline/continuation.py), [original outcome analysis](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/outcomes_analysis.py), [existing inventory exporter](../../report/model-analysis-review-20261008/build_inventory.py).

### D04 A/B cohorts and seven-class sentence annotations

**Population:** `W/campaign-v3/cohorts/<model>/cohorts.json` and `parts.json` freeze capture/annotation selections from **card-v3**. A selects one outcome-blind attempt on each of 1,547 questions/model. B is a selected correct/incorrect case-control population: **895 attempts, 51 questions in the union, one question in the seven-model intersection**. B cannot estimate benchmark accuracy. [Population export](../../report/model-analysis-review-20261008/cohort_populations.csv) binds analyzed counts.

**Labels:** [deployed seven-class annotations](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/campaign-v3/labels/) at `<model>/part-00..03/annotations.json`, with per-part `summary.json`. The JSON list stores `identity`, `inputs`, `label`, `unit`; identity includes dataset, problem/source/sample ID, `trace_sha256`, original `sentence_index`, `start`, `end`. Unit stores aligned text/offsets. Inputs retain the problem and previous/current/next sentence. To find an existing **Verify** label, select `label == "Verify"`, then join the full identity to its original D02 trace; do not renumber sampled sentences.

| Model | A attempts/questions | B attempts/questions | Completed sentence-label records across cohort parts | A median labelled sentence share |
|---|---:|---:|---:|---:|
| GPT-OSS | 1,547 / 1,547 | 276 / 37 | 116,550 | 36.36% |
| Nemotron | 1,547 / 1,547 | 96 / 14 | 105,760 | 24.29% |
| Qwen3.5 | 1,547 / 1,547 | 49 / 7 | 102,940 | 8.61% |
| Qwen3.6 | 1,547 / 1,547 | 47 / 6 | 102,820 | 7.93% |
| Gemma | 1,547 / 1,547 | 107 / 15 | 106,418 | 18.12% |
| GLM | 1,547 / 1,547 | 101 / 14 | 106,056 | 16.44% |
| Qwen3-30B | 1,547 / 1,547 | 219 / 31 | 113,135 | 17.81% |

Sentence totals reproduce cohort manifests and the four completed summaries/model. They are records in the annotation population, not independent questions or full-trace annotation coverage. Median shares describe the full A analysis and are different from pooled coverage on the smaller D05 common population. Sampling uses a nominal 100,000-label budget/model with B strata and floors; actual assigned counts above are larger. Preserve the original selection manifest rather than treating the nominal cap as the observed count. Code: [cohort selection](../../src/moe_exp/correlation_pipeline/sample_cohorts.py), [annotation](../../src/moe_exp/correlation_pipeline/annotate.py), [batch/partition/merge implementation](../../src/moe_exp/correlation_pipeline/). Provenance records in D05 bind the deployed inputs and summaries. Annotation completion is not annotation accuracy (X02).

**D04 annotation binding supplement:** all 28
`W/campaign-v3/labels/<model>/part-0[0–3]/checkpoints/batch-000000.json`
binding headers agree on the deployed program and reader configuration. The
program is `results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json`,
SHA256 `467510e4fc1759bf2833e7bc8f5a9d7937c52f2d31f530168776788a590b7a07`.
All 28 retained `logs/vllm.log` headers resolve `Qwen/Qwen3.8-27B` to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Deployment uses temperature 1.0,
top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1,
max_tokens 16,384, thinking enabled/low, preserve_thinking false, batch 64 and
one completion per sentence. The model-name-only launcher does not establish
the revision by itself; retained engine logs do.

Original annotation sources are `annotation_partition.py`, `annotation_batch.py`,
`batch_predictor.py` and `batch_judge.py` in `src/moe_exp/correlation_pipeline/`,
with frozen copies at
`W/campaign-v3/code/t1-827f1018725f2ac0/src/moe_exp/correlation_pipeline/`.
The frozen `sbatch/label_job.sbatch` records launch settings. The saved program
uses DSPy ChatAdapter (DSPy 3.3.0), Python 3.11, cloudpickle 3.1, no demonstrations
and a separately bound reader. The later direct dense adapter changes prompting,
parsing, temperature and cap; it is not the identical measurement. These are
header/log/source checks, not new annotations or a human-validity assessment.

### D05 Routing captures, count archives and common population

**Full A/B analysis outputs:** [seven-model analysis root](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/v3_analysis/results-r2/) at `<model>/<A|B>/`. `attempts.parquet` contains question/attempt identities, outcomes, lengths, label coverage and capture locators. `provenance.json` binds sources, label parts, layer/expert geometry and producing code. All 14 roots are represented in [cross-4 completion](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/v3_analysis/results-r2/cross-4/complete.json). Saved compact/result files are present; this catalogue does not reproduce every fit or tensor.

**Native routing:** read-only source locators are `/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3/manifests/<model>/<A|B>/ready.json` and shard `<index>.json`; each shard names its capture directory. Capture files use `<attempt_id>.pt.gz` (tensors), `<attempt_id>.json.gz` (packet), `<attempt_id>.complete.json` (completion marker). All 14 first-shard directories/manifests were found here. Availability and saved completeness receipts do not establish new v3 prediction-row compatibility; raw payloads were not loaded. The catalogue performs no write/transfer to scratch. GPT also has a compact root under `/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/routing-study/results/measurement-study/campaign-v3/analysis/gpt/A-99bf8d59506e2893/`; six A provenance records have `compact_root = null`, despite available newer analysis/count archives. Do not invent equivalent compact roots.

**Reusable selected-expert counts:** `W/v3_analysis/results-r2/<model>/A/class_counts.npz` is present for all seven, indexed by `attempt_id`, with `counts[N,9,L,E]`, `tokens[N,9]` and `classes`: Read, Analyze, Plan, Implement, Explore, Verify, Monitor, ALL_REASONING, ALL_LABELLED. Here N=1,547; L/E/native-k vary by checkpoint. `E_expert_enrichment.csv.gz` stores normalized selected-slot shares (`rate_in`); `E_class_experts.csv` stores selected profiles; `E_expert_usage_associations.csv` stores fits. [Profile inventory](../../report/model-analysis-review-20261008/expert_profile_inventory.csv) gives exact paths/geometry. These counts support training-only aggregation without generating or recapturing solutions; they do not contain expert output norms or identify safe deletions.

**Common sentence population:** [dynamics archives](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/dynamics-routing/results/) contain `<model>/A/sentences.parquet`, `hist.npz`; [readout freeze](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/FROZEN.readout.v3.json) binds the **common 522 questions / 509 duplicate families**. Sentence tables retain `attempt_id`, `question`, `sentence_index`, `segment`, `cls`, `n_tokens`; histogram/token features join through the saved indices. Missing labels or a segment change break adjacency. Readout uses 31,902–38,931 labelled sentences/model. Completed transitions on this same population report pooled label coverage of 7.0–28.1%, not the full-A median shares in D04. [Saved-data campaign inputs](../../report/expert-memory-study-v1/CAMPAIGN.json) name/hash all seven count, sentence and histogram sources; [saved-data reader](../../src/moe_exp/expert_memory_study/saved_data.py) defines exact joins.

**Additional compact inputs inspected during the joint review:** all seven
`hist.npz` sentence archives retain full-sentence `h` and `first16`/`last16`
selection-slot marginals with row identities. For unique native top-k selection,
inclusion frequency is `k × slot_frequency`. Source-sentence features joined to
an originally adjacent destination label can support next-class association
without another capture. They cannot reconstruct same-token cross-layer joints.
See `W/forum/tests/r3_context/code/dynrt/{extract.py,data.py,d3a_pairs.py}`.

All seven `W/v3_analysis/results-r2/<model>/A/features.parquet` schemas contain
`hidden_norm`, `hidden_step_distance` and `hidden_router_geometry`, with native
layer columns and depth summaries. This was checked from Parquet footers, not
by loading data rows. [Feature reduction](../../src/moe_exp/correlation_pipeline/features.py)
reduces hidden/router signals before serialization. These aggregate activation
summaries are reusable; they are not token-level hidden vectors or per-expert
output norms. Output norms were not established in the inspected schemas/code;
exhaustive raw archive availability remains uninspected.

**Full router-vector boundary:** the standard original A/B tensor archives
retain only `ids`, `weights`, `native_boundary_gaps` and `reasoning_tokens`.
`routing-study/measurement_study/production_capture.py::reduce_capture` in the
sibling routing-study repository creates that reduction and deletes its full
transient tensor after loading. The stopped v3 archive similarly persists
native/executed selected IDs/weights and positions/active masks, not full logits
(`src/moe_exp/verification_study/{adapters.py,storage.py}`). Its full-logit
`_verification_bank` was a worker-memory object; no persisted qualified bank or
calibration/policy output was established in the seven retained model roots.
Selected top-k weights and a boundary gap cannot reconstruct arbitrary signed
offsets for unselected experts.

A retained full-score **exception** exists in
`results/correlation_pipeline/forward-validation-v2/runs/qwen36-58344963/forward/unsloth--Qwen3.6-35B-A3B/math500/traces_with_routing.jsonl`:
four present logit/hidden tensor sets, with `raw_tensors_saved=true` in the
summary. The smallest inspected logit file is 67,299,795 bytes for 1,643 tokens
at 40 layers. Its acceptance binds an unsloth bitsandbytes-4-bit checkpoint and
an outcome-selected two-correct/two-incorrect engineering panel; the current
native Qwen3.6 checkpoint is FP8 with a different revision. Reuse as a historical
engineering fixture within scope; do not silently transfer its calibration to
the native seven-model study. Payloads were not loaded in this audit.

If a compatible persisted bank cannot be established, saved source token IDs
and native model bindings can support a separately proposed forward measurement
without regenerating solutions. Collecting missing full router vectors and
expert-output norms together may avoid separate passes where native hooks permit;
this is an implementation/measurement proposal, not established compatibility
or workload authorization.

**Families:** [family-freeze.json](../../report/experimental-resume-v1/family-freeze.json) maps canonical questions through `new_parent_pools.families`, preserving original splits and near-duplicate edges. [Original near-duplicate audit](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_integrity/out/near_dup.v2.json) is linked by that binding. Full A has 1,486 families across 1,547 questions; the common 522-question population has 509 families. These are different denominators.

### D06 External SAT annotation and probe data

**External population:** released DeepSeek-R1 SAT text, 38 responses / 3,125 labelled units; probes exclude 38 synthetic closing-think units, leaving **3,087 units**. This is external text teacher-forced through Qwen3.5, GPT-OSS and Gemma, not seven native benchmark corpora. Gold joins are `Question ID`/`response_id`, original unit index and character offsets. Loader expects `responses_original/SAT_deepseekR1_results.json` and `responses_labeled/*.json` below `data/Schoenfeld_Reasoning/`.

**Availability:** that local dataset directory is empty in this inspection; original SAT source files are unavailable here. The grouped raw Qwen `results/probeTest/qwen3.5-35b-a3b-gptq-int4/probes_grouped_s42/results.json` is also absent. [Grouped source bindings](../../paper/6a510c6db0d9b08aa438c142/grouped_probe_sources.json) retain hashes and saved result summaries for all three models; [setup](../../paper/6a510c6db0d9b08aa438c142/experimental_setup.tex), [probe tables](../../paper/6a510c6db0d9b08aa438c142/probeTest.tex), [annotation report](../../paper/6a510c6db0d9b08aa438c142/gepaLLMAsJudge.tex) remain available. Other raw grouped artifacts have not been exhaustively located. Legacy hidden-state caches are not assumed available merely because a parent results directory exists. Code: [SAT loader](../../src/moe_exp/probeTest/data.py), [probe workflow](../../src/moe_exp/probeTest/), [annotation workflow](../../src/moe_exp/gepaLLMAsJudge/). X02/X03 retain report-backed status and original uncertainty; no extraction/refit/download was performed.

### D07 Historical intervention continuations and dense labels

**Different A/B naming:** intervention A/B below are Qwen3.6 same-prefix experiments, not observational cohorts A/B in D04. `H` contains generation, blind reader and routing-first-stage roots; the [reconciled study inventory](../../report/routing-study-v1/STUDY_INVENTORY_v3.json) resolves exact raw/dense/telemetry artifact paths. Do not use its old pending recovery notes as current progress.

| Collection | Frozen roster | Raw solution/routing directory below `H` | Assignments / horizon | Semantic pairs / unknown |
|---|---|---|---:|---:|
| Intervention A | [A manifest](../../report/experimental-resume-v1/OVERNIGHT_DISCOVERY_A_MANIFEST_v1.json) | `overnight-routing-v1-9b714fe1091f1d16/` | 128 / 1,024 | 110 / 18 |
| Intervention B | [B manifest](../../report/experimental-resume-v1/OVERNIGHT_DISCOVERY_B_MANIFEST_v1.json) | `overnight-routing-v1-9c3dc95fda9cf5e9/` | 156 / 1,024 | 141 / 15 |
| Intervention C | [C manifest](../../report/experimental-resume-v1/OVERNIGHT_DISCOVERY_C_MANIFEST_v2.json) | `overnight-routing-v2-65e402b7e4fea3fe/` | 208 / 256 | 200 / 8 |
| FRESH | [FRESH manifest](../../report/experimental-resume-v1/OVERNIGHT_FRESH_COMPARISON_MANIFEST_v2.json) | `overnight-routing-v2-6b5aa286e46cb818/` | 3,480 / 1,024 | 2,920 / 560 |

A has eight families, B/C thirteen, FRESH 123 families / 134 starts; seeds/arms generate repeated branches. Each generation root has `STAGE_COMPLETION.json`; corresponding `overnight-semantic-v1|v2-<manifest-prefix>/` roots have `BLIND_FRAME.json`, `ARM_MAP.json` and reader files. Frozen rosters store family, source attempt/trace, sentence, prompt/prefix token IDs/digests, transition, arm and seed. Use assignment UID and manifest binding when joining outputs/reader votes. Original source/control assignments and reader unknowns remain included. Solution/route receipt formats are defined by [the historical execution code](../../scripts/experimental_resume/) and [historical integration](../../src/moe_exp/routing_control/historical_analysis_v1.py).

**Dense FRESH class data:** [final accounting](../../report/routing-study-v1/closeout-59495690/LABEL_ACCOUNTING.json) closes **187,524 assigned sentence identities: 187,136 valid, 388 unusable, zero unresolved**. [Trajectory results](../../report/routing-study-v1/dense-recovery-v3/TRAJECTORY_RESULT.json), [expert-count NPZ](../../report/routing-study-v1/dense-recovery-v3/EXPERT_SELECTION_COUNTS.npz), and [dense ledger binding](../../report/routing-study-v2/implementation-20261006-v1/closeout/DENSE_LEDGER_BINDING.json) locate the aligned class labels/counts and preserved reader calls. Seven-class dense labeling is complete; the **560 missing substantive FRESH reader pairs remain missing**. Full gate-weight trajectories, logit margins and original-prompt grades are unavailable in this dense artifact. [3,972-record length readout](../../report/experimental-resume-v1/CONTINUATION_LENGTH_READOUT_2026-10-05_v1.json) covers A+B+C+FRESH, separately from D03's 3,071 capped-parent continuations.

### D08 V3 discovery and candidate-specific judgments

**Roster/format:** [per-model discovery rosters](../../report/verification-study-v3/) at `<model>/DISCOVERY_ASSIGNMENTS.json` freeze the same **48 shared families/questions**, one original native assignment/model, 16,384-token cap. At `R/<model>/discovery/<assignment_uid>.json`, receipts store `assignment`, `emitted_text`, `emitted_token_ids`, finish/closure, runtime/executor/model digests and `capture`. The companion UID `.npz` stores compressed native routing; `capture.array_path/array_sha256` locates/binds it. `.attempt-*.json` and price/load JSONs are not extra scientific solutions. Count only canonical UID generation receipts.

**Audit/annotation:** `R/<model>/AUDIT_FRAME.json` has up to 128 prefix records, original candidate/constraint/prefix/source identities and frozen 36/12 apparent/held-out family split. Two fixed seeded reader draws/prefix live in `<model>/eligibility-ratings/`; the [reader contract](../../report/verification-study-v3/READER.json) and [format amendment](../../report/verification-study-v3/SEMANTIC_FORMAT_BINDING-r1.json) preserve caps, invalid/truncated/disagreeing outcomes and original schemas. `R/<model>/RECOGNIZER.json` binds `semantic_decision_ledger_path`; that wrapper points into `R/semantic-format-r1/ledgers/<model>/eligibility/` to the full original-draw interpretation ledger. No redraw or gate relaxation is implied by failure.

| Model | Canonical discovery receipts / 48 | Audit frame records | Saved recognition outcome | Evidence boundary |
|---|---:|---:|---|---|
| GPT-OSS | 48 | 128 | **FAIL** | [Validated full reader closeout](../../report/verification-study-v3/GPT_ELIGIBILITY_CLOSEOUT-r1.json) |
| Nemotron | 48 | 128 | **FAIL** | [Validated full reader closeout](../../report/verification-study-v3/NEMOTRON_ELIGIBILITY_CLOSEOUT-r1.json) |
| Qwen3.5 | 42 | No full frame | Incomplete at user stop | [Earlier 22-item shard closeout](../../report/verification-study-v3/QWEN35_DISCOVERY_SLICE_CLOSEOUT-r1.json) validates that shard; 42-file presence supersedes its earlier 30/48 progress count, not the absent full-roster closeout |
| Qwen3.6 | 48 | 128 | **FAIL** | [Saved qualification](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/candidate-verification-v3-20261007-r4/qwen36/RECOGNIZER.json), bound 256-entry original-draw ledger; supersedes the retained-reader/audit wait |
| Gemma | 48 | 128 | **FAIL** | [Validated full reader closeout](../../report/verification-study-v3/GEMMA_ELIGIBILITY_CLOSEOUT-r1.json) |
| GLM | 48 | 128 | Unmeasured | Frame and 48 receipts present; full independent archive/accounting closeout not reproduced here. Retained reader was cancelled before runtime in [STOP_RECEIPT](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json) |
| Qwen3-30B | 48 | 128 | **FAIL** | [Validated full reader closeout](../../report/verification-study-v3/QWEN330B_ELIGIBILITY_CLOSEOUT-r1.json) |

These counts are file/receipt inspection, not new tensor validation. Original [discovery count table](../../report/verification-study-v3/DISCOVERY_AUDIT_COUNTS-r1.csv) and [receipt](../../report/verification-study-v3/DISCOVERY_AUDIT_COUNTS-r1.json) retain their earlier five-discovery/four-audit cutoff. Completed frames need not represent all 48 families or 128 distinct prefix positions: GPT/Nemotron have 127 unique positions among 128 records across strata. Reader draws are repeated judgments, not independent problems. Counts and source availability from this catalogue are recorded in [validation](experiment_inventory_validation.json).

**Reuse evidence:** all **48 discovery questions match historical A records for every model**, including the six Qwen3.5 assignments without new receipts. This is directly recorded in the [stop receipt's identity inspection](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json). Historical A here is the saved card-profile analysis population (D04/D05), not proof of identical D01/D02 tokens. Full historical text/token/routing compatibility with v3 remains **uninspected**. A new capture format or a new reader rubric alone cannot justify regenerating solutions.

### D09 Reserved independent populations and expert-memory artifacts

**Question data, not measured outcomes:** [full MATH question table](../../report/routing-study-v1/FULL_MATH_QUESTION_TABLE.json) and [duplicate graph](../../report/routing-study-v1/FULL_MATH_DUPLICATE_GRAPH.json) bind 5,000 source questions at revision `21a5633873b6a120296cce3e2df9d5550074f4a3`, preserving prior benchmark/exposure exclusions. [Independent family manifest](../../report/routing-study-v1/INDEPENDENT_FAMILY_MANIFEST.json) freezes 32 validation / 96 final families from 4,479 eligible families; those reservations are not completed inference. Family/question IDs join the source table/graph; gold solutions remain separate from blinded online inputs.

**Memory-study selection results:** [seven model selection receipts](../../report/expert-memory-study-v1/) at `<model>/SELECTION.json`, [TRANSITIONS.json](../../report/expert-memory-study-v1/TRANSITIONS.json), [FROZEN_MASKS.json](../../report/expert-memory-study-v1/FROZEN_MASKS.json) and [exhibits-r2](../../report/expert-memory-study-v1/exhibits-r2/EXPORT.json) already exist. Selection JSON stores five common folds, coverage/support, stability, all-reasoning sensitivity and full-training masks, with `outcomes_used=false`. CSV/PDF exhibits expose held-out coverage, class/layer coverage and overlap; they do not measure post-removal accuracy.

**Prospective reserved population:** [POPULATION.json](../../report/expert-memory-study-v1/POPULATION.json), [later-exposure ledger](../../report/expert-memory-study-v1/LATER_EXPOSURE_LEDGER-r1.json) and [PILOT_ASSIGNMENTS.json](../../report/expert-memory-study-v1/PILOT_ASSIGNMENTS.json) freeze 4,351 eligible families, 32 pilot-only and 4,319 in final order, with 896 planned pilot assignments (32 × seven models × four arms). Their presence does not establish pilot outcomes or a final sample size. User stop interrupted the first GPU qualification and cancelled the rest; no completed all-seven loader qualification or prospective accuracy/memory result is established. Existing manifests, sources and partial files remain preserved.

## Experiments and analyses already performed

Each entry states the scientific question, inputs/controls, model coverage, completion, finding and limits. Existing uncertainty is retained; catalogue inspection adds no fit, bootstrap, inference or grade. Operational history is reached through linked receipts instead of another job ledger.

### X01 Benchmark scoring and output-budget analysis

**Question:** how correct/long are native solutions under each saved profile? **Inputs/method:** D01–D03, strict math verification plus accepted historical J1 residual equivalence and separate Minerva tolerance; original question-cluster bootstrap, explicit caps/unknown grades. **Coverage/status:** all seven, 84 saved benchmark/corpus cells inspected and reproduced by the prior inventory. **Finding:** full original benchmark outcomes and descriptive historical/card/cap-continuation comparisons exist; card-v3 has 33 unknown outcomes. **Limit:** unequal model profiles and attempt weights; continuation selection is conditional on truncation. Scored means and all-assignment bounds are different estimands. Results/data/code/provenance: [audit](../../report/model-analysis-review-20261008/AUDIT_NOTES.md), [cells](../../report/model-analysis-review-20261008/benchmark_cells.csv), [source bindings](../../report/model-analysis-review-20261008/SOURCES.json), D01–D03.

### X02 Reasoning-class annotation qualification

**Question:** how reliably does the automated annotator assign seven visible episode classes? **Inputs/method:** D06 external SAT development/assessment and separate generated-math deployment; grouped CV, seed-versus-optimization and effort comparisons, rule-assisted external probe review. **Coverage/status:** historical annotation studies complete/report-backed; deployed D04 labels exist for all seven. **Finding:** static-reward grouped macro-F1 0.6505 light / 0.6437 heavy on 2,789 units; imported Qwen3.8 seed balanced accuracy 0.7400 versus optimized 0.7288, with the seed retained by the Plan-recall gate; effort variants both 0.7101 on 407 reused units yet disagree on 41. **Limits:** reused assessment/previously inspected nominal locked partition, no independent human ground truth in the external probe check, sampled deployment labels. These measures do not validate candidate-specific checking judgments. [Annotation sources](SOURCES.md#s04-annotation), [report](../../paper/6a510c6db0d9b08aa438c142/gepaLLMAsJudge.tex), [effort table](../../paper/6a510c6db0d9b08aa438c142/judge_effort_table.tex), [code](../../src/moe_exp/gepaLLMAsJudge/), D04/D06.

### X03 Hidden-state episode probes

**Question:** can prospective hidden states read external episode labels? **Inputs/method:** D06 teacher-forced SAT; legacy sentence splits had all 38 responses in train/test, followed by repaired five outer response folds and inner layer selection, prefix-position controls. **Coverage/status:** Qwen3.5/GPT/Gemma, completed archived grouped results, raw availability limited. **Finding:** grouped macro AUROC 0.852/0.780/0.788; gains over position +0.286/+0.209/+0.221 with original intervals in [FINDINGS E01](FINDINGS.md#e01-annotation-and-hidden-state-observability). **Limit:** 38 external responses, two-feature position baseline, conditional saved-fold uncertainty; no native online detector or optimal steering-layer claim. [Source/result bindings](../../paper/6a510c6db0d9b08aa438c142/grouped_probe_sources.json), [tables](../../paper/6a510c6db0d9b08aa438c142/probeTest.tex), [producing probe code](../../src/moe_exp/probeTest/), D06. Repaired results supersede the legacy leakage objection within this scope; neither needs repeat generation.

### X04 Routing correlations, class profiles and adjusted correctness associations

**Question:** which route statistics/expert profiles accompany labels and correctness? **Inputs/method:** legacy three-model benchmark analysis, newer all-seven D04/D05 A/B analysis and September audit; length/dataset/difficulty/class-fraction controls, position/common-range sensitivities, outcome-blind class-expert nomination and original bootstrap/BH families. **Coverage/status:** latest cross-4 covers all seven and both cohorts; completed saved analyses. **Finding:** effects vary by model/benchmark; B's adjusted late boundary-margin effect d=−0.2983 [−0.4637, −0.1330], q=0.004879, has no surviving common-range counterpart (minimum q=0.2190). The saved 147 mean-usage rows include Analyze-positive/Verify-negative Qwen3.5/3.6 associations. All 49 all-reasoning mean-usage coefficients appear in the saved heatmap. **Limit:** retrospective difficulty and same-cohort expert selection, question-ID rather than duplicate-family bootstrap, original within-model 21-test BH family; these are not causal specialists or deletion evidence. [Finding/method](FINDINGS.md#e12-analysis-first-seven-model-review--2026-10-08), [method receipt](../../report/model-analysis-review-20261008/EXPERT_ASSOCIATION_METHOD-r1.json), [full rows](../../report/model-analysis-review-20261008/expert_usage_correctness_saved.csv), [September review](../../report/routing-steering-audit-20260928/REVIEW.md), [cross-4](SOURCES.md#s17-latest-seven-model-observational-analysis), [producing v3an code](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/v3_analysis/v3an/), D04/D05.

### X05 Present-class readout, next-class and correctness prediction

**Question:** does routing add information beyond text/position/history? **Inputs/method:** D05 common population, frozen duplicate-family nested CV, saved judge-context/text baselines, lexical-null controls; anticipation uses past-only features and separately tests contained transitions; correctness uses registered C8/marginal baselines and matched noise/precision simulation. **Status/coverage:** clean present-class readout complete for all seven, 522 questions/509 families each; clean next-class complete for GPT and Qwen3.6 only; R3-E B1/B4 are GPT-specific. **Finding:** all seven present-class gains and lexical-null excess pass their saved intervals. Primary next-class changes are −0.02073 nats/pair (GPT) and −0.05570 (Qwen3.6); exact intervals/scopes remain in [E03](FINDINGS.md#e03-routing-information-and-anticipation-already-exist). B1 gain +0.008385 nats/token-pair survives its filter; B4 log-loss change −0.000729 [−0.005313, +0.004384] on 508 scored questions/495 families is uncertain. **Limits:** automated visible labels, retrospective judge context for readout, no substantive candidate-check calibration/online trigger; B1 excludes 69.24% of labelled tokens, B4 uncertainty is not equivalence. Five corresponding clean anticipation results are absent.

**Locate:** [readout JSON](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/readout-results.v3.json), [anticipation JSON](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/anticipation-results.json), [compact readout](../../report/model-analysis-review-20261008/routing_readout.csv), [compact anticipation](../../report/model-analysis-review-20261008/anticipation.csv), [B4 result/precision readout](../../report/experimental-resume-v1/R3E_B4_PRECISION_READOUT_v1.md). Original B4 fits are `W/steering-v1/runs/resume-v1/r3e-cv-clean-v1/full.json`; confirm-exclusion and precision-v3 siblings are identified by that receipt. [Source-bound clean driver/code](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/code/), [claim ledger](../../report/experimental-resume-v1/CLAIM_LEDGER_v2.json) and D05 bind underlying data/provenance.

### X06 State transitions and native boundaries

**Question:** how do observed labelled episodes transition, dwell and re-enter? **Inputs/method:** legacy three-model label matrices, Qwen3.6 native boundary windows, D07 dense continuations, and completed all-seven D05 common-population analysis. Original consecutive sentence indices/same segments only, equal-family normalization with support, censored dwell endpoints. **Status:** all-seven common transitions now complete; older notes calling them missing are superseded by [saved-data closeout](../../report/expert-memory-study-v1/SAVED_DATA_CLOSEOUT-r1.json). **Finding:** model-dependent transition matrices and observed dwell/coverage exist; common label coverage 7.0–28.1%. Qwen3.6 native profiles cover 48 traces, 6,905 records and 42 transition profiles. **Limit:** sparse labels break adjacency; selected window ends cannot identify closure probabilities; neither is token-scale causal timing.

**Results/data/provenance/code:** [seven-model TRANSITIONS.json](../../report/expert-memory-study-v1/TRANSITIONS.json), [transition figure](../../report/expert-memory-study-v1/exhibits-r2/transitions.pdf), [native profiles](../../report/routing-study-v1/NATIVE_BOUNDARY_PROFILES.json), [legacy statistics](../../paper/6a510c6db0d9b08aa438c142/class_transition_statistics.json), [gap/censoring implementation](../../src/moe_exp/expert_memory_study/transitions.py), [native boundary producer](../../scripts/experimental_resume/discovery_boundaries_v1.py), D05/D07. Native offset 0 is source sentence; +1 is destination.

### X07 Conditional target nomination and descriptive concentration

**Question:** do class-transition profiles nominate stable experts, and how concentrated are labelled selections? **Inputs/controls:** Qwen3.6 matched source-class/family Verify-entry windows with train-selected fold proposals; separately seven-model A enrichment counts ranked/measured on the same labelled cohort. **Status/coverage:** Qwen3.6 conditional nomination complete, all-seven descriptive concentration complete. **Finding:** L28/E189 and E9 matched inclusion differences +0.08118/+0.06296; 483 positive windows/48 families, four fold-selected proposals with positive held-out directions. Full candidate table has 143,360 selection rows and the same number of executed-weight rows; 44,442 meet its proxy support/direction rule. Keeping half the identities per layer by labelled frequency covers 61.90–96.11% of observed slots; Monitor is least covered for six models. **Limit:** final fixed expert identities are nominations, not independently confirmed modules; same-cohort curves are not held-out performance and half-keep is a plotting landmark. [Dense Verify result](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-verify-source-v2/RESULT.json), [candidate table](../../report/routing-study-v1/CANDIDATE_TABLE.json), [global/conditional comparison](../../report/routing-study-v1/GLOBAL_CONDITIONAL_COMPARISON.json), [concentration receipt](../../report/model-analysis-review-20261008/LABELLED_EXPERT_COVERAGE-r1.json), [CSV](../../report/model-analysis-review-20261008/labelled_expert_coverage.csv), [exporter](../../scripts/paper/export_labelled_expert_coverage_r1.py), [nomination implementation](../../src/moe_exp/routing_control/), D05. Completed held-out deletion-mask coverage is a separate X11 result.

### X08 Legacy global steering and rescoring

**Question:** do fixed/margin expert promotions improve complete answers? **Inputs/method:** saved GPT/Qwen generated comparisons, native paired controls, uniform offline rescoring and problem-stratified paired bootstrap with primary attempt and equal-problem sensitivity weights. **Status:** archived complete comparisons have 1,395 paired attempts/465 problems; separate partial/quarantined runs remain marked. **Finding:** no complete policy has a positive accuracy interval strictly above zero under either weighting; GPT paper-style promotion causes severe harm (positive arm −61.65 pp [−69.32, −53.41]). Rescoring 40,097 stored records changed 4,054 wrong→right and one right→wrong. **Limits:** model/operator-specific, output-token means descriptive, legacy selection differs from conditional nomination; eight-target GPT runs violate native k=4 and remain invalidated. Qwen paper-style 494/1,395 is partial; margin hooks change membership and do not isolate pure margin effects. [Full effects](FINDINGS.md#e05-legacy-global-steering-provides-substantive-negative-evidence), [rescoring tables](../../paper/6a510c6db0d9b08aa438c142/guiding_rescoring_results.tex), [source bindings](../../paper/6a510c6db0d9b08aa438c142/current_results_sources.json), [invalidated runs](../../paper/6a510c6db0d9b08aa438c142/invalidated_oss_runs.json), [selector audit](../../paper/6a510c6db0d9b08aa438c142/expert_selection_audit.tex), [steering code](../../src/moe_exp/correlation_pipeline/dynamics/steering_cli.py). Raw per-comparison paths are resolved by the bindings; exhaustive availability was not re-audited here.

### X09 A/B/C/FRESH actions, dense recovery and utility boundary

**Question:** can sparse pulses engage routing and change a substantive check/commitment? **Inputs/method:** D07 Qwen3.6 same-prefix native/duplicate/random/singleton/operator controls, fixed horizons and original arm-blind two-reader endpoints, seed averaging and paired family bootstrap (50,000 draws) with each study's own multiplicity and missing-reader bounds. **Status:** scientific engagement/semantic analyses and dense recovery complete within their original scopes; original-prompt utility absent. **Finding:** A pair inclusion +0.1376 [0.0763, 0.1882], B once +0.2940 [0.1277, 0.4761]; no FRESH primary contrast strictly excludes zero. Verification force−native +5.50 pp [−4.50, +15.50], with unknown-reader identification bounds [−16.5, +24.5]. Dense trajectories are complete descriptions; historical cap-length data do not show consistent shortening. **Limit:** eight/thirteen/123 independent families by design, selected starts, capped horizon, sparse telemetry and reader unknowns; no safe-deletion, resident-memory, causal-null or independent original-prompt accuracy claim. Historical force-off comparisons kept expert weights loaded.

**Results/data/provenance/code:** [historical reproduction](../../report/routing-study-v1/HISTORICAL_REPRODUCTION.json), [integration](../../report/routing-study-v1/HISTORICAL_INTEGRATION.json), [FRESH analysis](../../report/experimental-resume-v1/OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2/ANALYSIS.json), [assigned outcomes](../../report/experimental-resume-v1/OVERNIGHT_FRESH_COMPARISON_ANALYSIS_v2/ASSIGNED_RESULTS.json), [dense final accounting](../../report/routing-study-v1/closeout-59495690/LABEL_ACCOUNTING.json), [historical analysis implementation](../../src/moe_exp/routing_control/historical_analysis_v1.py), [dense recovery producer](../../scripts/experimental_resume/dense_recovery_v2.py), D07. [REPORT_v4](../../report/routing-study-v1/REPORT_v4.md)'s 11,200-label progress is historical and superseded by final accounting. Selected utility-policy files are nominations, not completed utility comparisons.

### X10 V2/V3 qualification, failed gates and interrupted work

**Question:** is execution/measurement qualified for a prospective verification experiment? **Inputs/controls:** original v2/v3 frozen sources, architecture-specific same-state fidelity/lifecycle/capture fixtures, blinded semantic readers, shared equivalence audit, fixed support/precision gates. **V2 status:** Qwen3.6 passed 13 engineering cases, including 16k capture; semantic pilot produced **0/128 ratings** after serialization failure. CPU repair exists, but proposal support also fails (apparent complete-candidate counts 0 unchecked / 0 checked); scientific pricing/effects did not run. [Final status/report/closeout](../../report/routing-study-v2/implementation-20261006-v1/final/), [support precheck](../../report/routing-study-v2/implementation-20261006-v1/AUDIT_SUPPORT_PRECHECK.json), [v2 contracts/code](../../docs/routing-study-v2.md) bind this engineering-versus-science distinction.

**V3 status:** all seven backends passed **14/14** [qualification checks](../../report/verification-study-v3/QUALIFICATION_CLOSEOUT-r4.json). D08 reconciles discovery/reader availability. Original recognition gate is at least 70% apparent family confirmation and ten apparent families for **each** decision. Saved qualifications are:

| Model | Eligible confirmation / apparent families | Already-checked confirmation / apparent families | Outcome |
|---|---:|---:|---|
| GPT-OSS | 22.41% / 29 | 18.75% / 8 | FAIL |
| Nemotron | 25.00% / 34 | 10.00% / 10 | FAIL |
| Gemma | 25.52% / 32 | 41.67% / 12 | FAIL |
| Qwen3-30B | 10.10% / 33 | 33.33% / 12 | FAIL |
| Qwen3.6 | 28.57% / 28 | 8.33% / 12 | FAIL |
| Qwen3.5 / GLM | Unmeasured | Unmeasured | Incomplete at stop |

The Qwen3.6 values come directly from its sealed saved qualification and 256-entry format ledger, superseding earlier unmeasured audit notes. The first four independent closeouts/reviews retain their original ledger validation. This inspection did not independently revalidate all raw Qwen3.6 draws or refit a recognizer. Apparent confirmation measures this controller/reader pipeline, not independently established semantic truth. Invalid/truncated/disagreeing draws remain unconfirmed; support/held-out uncertainty and usable pairs have different denominators.

**Shared grading:** [full J1 closeout](../../report/verification-study-v3/J1_QUALIFICATION_CLOSEOUT-r1.json) validates 100 original items / 300 draws, 97 consensus matches, no clear-wrong false equivalences, and three Minerva errors exceeding the maximum two: **FAIL**. Existing strict-grading/unknown-accuracy bounds apply; no production J1 qualification is inferred. The older D03 98/100 accepted grader is a different deployment.

**Unmeasured/interrupted:** no new v3 nominated/calibrated treatment effect or original-prompt joint utility result is established. Qwen3.5 partial discovery and GLM retained reader stay incomplete at user stop, not FAIL. Expert-memory GPU qualification is separately interrupted/unmeasured (X12). M9 edited isolation and counterfactual NLL have preserved failed engineering branches; native-finalization has 0/32 outcomes after an exact cross-run replay gate, subsequently rejected as a future sampled-generation requirement; X3 is deferred. [Separate diagnostics](SOURCES.md#s16-separate-diagnostic-branches) retain those limits without restarting them.

**Sources/code:** [v3 protocol](../../report/verification-study-v3/PROTOCOL.json), [scientific source closure](../../report/verification-study-v3/CAMPAIGN.json), [format binding](../../report/verification-study-v3/SEMANTIC_FORMAT_BINDING-r1.json), [verification implementation](../../src/moe_exp/verification_study/), D08 and its closeouts. [Historical automation receipts](IMPLEMENTATION.md), [stop receipt](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json) and [read-only availability/count validation](experiment_inventory_validation.json) provide execution/provenance; they confer no restart authorization.

### X11 Family-held-out expert selection and stability

**Question:** does state-balanced selection cover held-out labelled classes better than frequency at matched quotas, and how stable are masks? **Inputs/method:** D05 all-seven per-attempt counts, 1,547 questions/1,486 families, five common family folds; rank only training counts with equal total family weights. State balancing averages class-conditional slot shares equally; frequency pools the same labelled counts by mass. Fixed matched random, identical per-layer quotas/ties and all-reasoning frequency sensitivity; final masks frozen before prospective outcomes, correctness excluded from ranking.

**Coverage/status/finding:** all seven five-fold and full-training fits **completed**, bound by [saved-data closeout](../../report/expert-memory-study-v1/SAVED_DATA_CLOSEOUT-r1.json). Retain 75% identities/serialized routed-expert bytes per layer: GPT24, Nemotron96, Qwen3.5/3.6 192, Gemma96, GLM48, Qwen3-30B96. State balancing has higher minimum held-out class coverage in all five folds for GPT/Nemotron/Gemma/GLM/Qwen3-30B; labelled frequency is higher for Qwen3.5/3.6. Mask overlap/stability and class/layer support are already exported. **Limit:** saved slot coverage under original routing is not rerouted accuracy, weight/output importance or physical memory benefit; sparse training labels constrain scope. Mixed results are preserved.

**Results/data/provenance/code:** D09 selection JSONs, [coverage CSV](../../report/expert-memory-study-v1/exhibits-r2/selection_coverage.csv), [overlap CSV](../../report/expert-memory-study-v1/exhibits-r2/mask_overlap_stability.csv), [class/layer CSV](../../report/expert-memory-study-v1/exhibits-r2/class_layer_coverage.csv), [figure](../../report/expert-memory-study-v1/exhibits-r2/held_out_coverage.pdf), [frozen masks](../../report/expert-memory-study-v1/FROZEN_MASKS.json), [source closure/inputs](../../report/expert-memory-study-v1/CAMPAIGN.json), [selection estimator](../../src/moe_exp/expert_memory_study/selection.py), [saved-data implementation](../../src/moe_exp/expert_memory_study/saved_data.py). Older notes proposing this fit are superseded by these results; no new fits were run here.

**Joint-review clarification of X11 inputs:** each model's `SELECTION.json`
contains five saved all-reasoning-frequency sensitivity masks in addition to
the three main policies' fold/final masks. There is no full-population
all-reasoning final mask or complete rank array in those files. Masks are
stored in expert-ID order, not importance order. The complete integer-keep
coverage CSV/JSON from X07 also discards ranking identities. Different quota
masks may therefore need deterministic derivation from the saved counts;
their absence is not missing native generation. Source:
`src/moe_exp/expert_memory_study/{selection.py,saved_data.py}` and
`scripts/paper/export_labelled_expert_coverage_r1.py::describe`.

### X12 Checkpoint-byte accounting and resident-memory boundary

**Question:** how many serialized bytes belong to native routed experts, and are realized savings/accuracy measured? **Inputs/method:** all-seven checkpoint configs/indexes/safetensors JSON headers, expert-local tensor/scale attribution, native/draft/other separation; no weight-payload reads. **Completed finding:** native expert serialized payload ranges **10.166–57.982 GB**. GLM's index declares 31.221 GB while actual header tensor offsets sum to 62.443 GB for total payload; shard sizes support the larger number. Equal per-expert header bytes support X11's count quota mapping. **Limit:** repacking, shared/draft tensors, KV/hybrid-state cache, workspaces and allocator behavior separate payload from resident GPU bytes.

**Prospective status:** private physical-removal harness exists, with 55 synthetic contracts and all-seven frozen masks. The first GPU qualification was interrupted by the user, remaining qualifications cancelled before runtime. No completed all-seven within-execution loader qualification, realized resident saving or paired prospective accuracy is established. Planned four-arm pilot/final rosters are inputs, not results. Neither old route masking nor header bytes establish the resident-memory objective.

**Results/data/provenance/code:** [header inventory](../../report/model-analysis-review-20261008/CHECKPOINT_HEADER_INVENTORY.json), [model bytes](../../report/model-analysis-review-20261008/checkpoint_tensor_bytes.csv), [expert tensor bytes](../../report/model-analysis-review-20261008/checkpoint_expert_tensor_bytes.csv); the header receipt names/hashes its generator and per-model checkpoint inputs. [Memory harness documentation](../../docs/expert-memory-study-v1.md), [implementation readiness](../../report/expert-memory-study-v1/READINESS-r1.json), [saved-data closeout](../../report/expert-memory-study-v1/SAVED_DATA_CLOSEOUT-r1.json), [private loader code](../../src/moe_exp/expert_memory_study/), [stop](../../report/experiment-stop-20261008-r1/STOP_RECEIPT.json), D09.

### X13 Cross-layer routing and depth-path studies

**Located during the October 8 joint design review:** dedicated depth/path
studies already exist for all seven models. Their omission from the first
catalogue was not evidence of absence. Reuse this branch before proposing
another generic path/correlation study.

| Existing question | Inputs, estimator and completed coverage | Exact sources |
|---|---|---|
| Are layer choices dependent and predictable? | All seven; 200 A attempts/model, at most 4,096 sampled reasoning tokens/attempt. Top-1 is the selected expert with largest executed weight. Adjacent-layer normalized mutual information (NMI), held-out-attempt bigram versus unconditional-mode prediction, top-100 three-layer path concentration, depth/weight slopes and class-conditional bigram Jensen–Shannon distances. | [Part 1 summary](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/depth-paths/results/part1/summary.md), per-model JSONs in that directory; [original plan](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/depth-paths/ANALYSIS_PLAN.md) |
| How much dependence remains after token/position control? | All seven; routed-layer gaps 1, 2, 4 and 8, token-ID × relative-position shuffle null, singleton disclosure/sensitivity. Saved adjacent-layer mean NMI excess is positive in all seven (0.0936–0.1974), with original descriptive uncertainty. | `W/depth-paths/results/s3/nmi/<model>.json`; [NMI implementation](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/depth-paths/dp/s3_nmi.py) |
| Do path features add correctness information? | GPT A primary and Qwen3.6 A replication. Depth-bigram surprise, temporal top-1 coherence and weight/depth slope added to original controls. Gains are −0.00079 [−0.00308, +0.00169] and +0.00052 [−0.00240, +0.00362] nats/attempt. Neither establishes a positive increment or equivalence. | [Part 2 summary](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/depth-paths/results/part2/summary.md), `results.json` in that directory |
| Do routing/path features describe reasoning classes? | The clean all-seven X05 readout already combines depth-bigram surprise, temporal top-1 coherence and per-layer top-1 histogram PCA with text/position/context and lexical controls. Its combined gain does not isolate interactions over marginal layer usage. | `W/forum/tests/r3_context/readout-results.v3.json`; [readout implementation](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/forum/tests/r3_context/code/r3d_readout.py) |
| Is there class-specific coordination beyond individual profiles and generic pair coupling? | Clean GPT B1: 43 layer pairs at gaps 1/4, 509 questions / 496 families, five family folds × five repeats. Held-out route-pair gain +0.008385 [0.008044, 0.008737] nats/token-pair. The fitted log-linear/IPF null preserves both class/expert margins and generic expert-pair coupling. This result is already measured; a new fit is not adopted under Q16. | `W/steering-v1/runs/resume-v1/r3e-b1-clean-v2/result.json`; [clean driver](../../scripts/experimental_resume/run_r3e_b1_clean_v2.py); `W/forum/tests/r3_context/code/dynrt/d3b_b1.py` |

**Scope and exposure:** Part 1 uses original attempt-level splits/bootstrap and
sampled-token weighting. Later lexical-null uncertainty uses ten attempt-group
jackknife replicates with ten null draws each, not duplicate-family inference
or a new simultaneous seven-model test. The [resume summary](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/depth-paths/results/resume-v1/summary.md)
retains six NMI populations as `HISTORICAL_CONFIRM_EXPOSED`; Qwen3.6 is
`EXACT_ID_CLEAN`. Later clean class readouts supersede old readout cells only;
they do not remove NMI exposure. Position-only nulls differ from lexical nulls.

**Representation and reuse:** top-1 paths project native top-k routing. D05
raw captures retain `ids[layer, completion_token, k]`, selected weights, native
boundary gaps and reasoning-token indices. [Extraction](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/v3_analysis/v3an/extract.py)
and [path features](/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/depth-paths/dp/pathfeat.py)
define alignment/geometry. These reduced tensors do not recover unselected
logits or expert output norms; marginal `class_counts.npz` loses joint order.
This review inspected source/schema and compact outputs, not all tensor
payloads. Compatibility with a new full-top-k or boundary estimator remains
uninspected; no need for fresh generations has been established.

**Compact reuse before raw tensors:** all-seven labelled-token dominant-expert
caches exist at `W/depth-paths/results/s3/_tok/<model>/<original-row>.npz`.
`dp/s3_collect.py` retains token IDs, labelled-token ranks/sentence identities,
`lab_top1` by layer and adjacency; GPT also retains `top1_all`. Bind row selection
to `W/forum/tests/r3_context/FROZEN.readout.v3.json`. These retain aligned
top-1 joint structure that the class-count archives discard. A separate inspected
full-top-k compact B1 cache is GPT-specific:
`W/dynamics-routing/results/B1/tokens/{tokens.npz,sentences.parquet,attempts.parquet}`.

All 14 clean readout loss archives exist at
`W/forum/tests/r3_context/readout/<model>/<token_identity|judge_context>/losses.npz`,
alongside fold predictions. All 12 GPT/Qwen3.6 anticipation prediction archives
exist at `W/forum/tests/r3_context/anticipation/<gpt|qwen36>/<next|switch|dest>[-contained]/predictions.npz`.
File presence was checked; source-row filtering and estimators must follow their
frozen drivers. Reaggregating saved losses cannot create an unfitted marginal-only
or per-layer comparator. B1 saves pair means and overall per-class summaries,
not pair-by-family/fold/class uncertainty arrays. These are specific residual
data limits, not grounds to rerun the completed fits.

This supplement extends the inspection covered by the preserved
`experiment_inventory_validation.json` receipt. No original result, fit, raw
archive or scientific source changed, and no new analysis was run.

## What can be reused

### R01 Existing-artifact questions

These are reuse opportunities, **not workload authorization**. Reading/re-exporting compact saved results is bounded documentation work; large archive processing or new fits still need the existing publication/resource gate and explicit resumed computation authorization.

| Question or paper exhibit | Existing inputs/results to cite | Possible without new generation | Boundary to disclose |
|---|---|---|---|
| Native benchmark performance, cap/unknown/length accounting | D01–D03, X01 | Retrieve solutions and preserve/re-export original 84 cells, intervals and score denominators | Unequal native profiles; no regrading or new uncertainty in this catalogue |
| Locate Verify reasoning and inspect checking examples | D02/D04/D08 | Join existing labelled sentences or candidate prefixes to saved solution text | Seven-class Verify and candidate-specific checking are different tasks (R02) |
| Routing/class differences and adjusted expert associations | D04/D05, X04 | Inspect/re-export every saved coefficient/profile/control sensitivity | Same-cohort selection; full tensors/output norms not implied |
| Present-class and past-only prediction | D05, X05 | Reuse seven clean readouts and two clean anticipation studies, GPT B1/B4 and precision study | Five equivalent clean anticipation result cells absent; do not rerun completed cells |
| Cross-layer coupling and dominant-expert paths | D05, X13 | Reuse seven depth/path summaries and lexical-null results, two correctness comparisons and path-containing clean readouts | Preserve exposure, attempt-level uncertainty and top-1 projection; full-top-k compatibility is a separate question |
| Held-out retention coverage/stability at adopted quota | D05/D09, X11 | Reuse completed seven-model fold/mask/support/overlap exports | Coverage is not post-removal utility; new quota/sensitivity fits are new analysis |
| Transition structure, dwell and gap sensitivity | D05/D07, X06 | Reuse completed common-seven transitions, native boundary/dense descriptors | Observed sparse labels/selected windows; do not join gaps or infer closure rates |
| Historical action engagement/semantic effects and token lengths | D07, X08/X09 | Preserve original paired effects, intervals, unknown bounds and raw branches | Selected/capped populations; 560 FRESH semantic pairs missing, original-prompt utility unavailable |
| Model/role qualification and reader failure diagnosis | D08, X10 | Inspect original stopped/truncated/invalid draws and model-specific support/confirmation | No replacement judge, redraw, relabel or gate change follows from a saved FAIL |
| Question reuse for discovery and necessity assessment | D01/D02/D04/D08 | Start from all48 historical question matches/model and inspect saved provenance/text/token locators | Exhaustive trace/routing compatibility remains uninspected; fresh provenance is not necessity |
| Static expert bytes and potential budgets | D09, X12 | Reuse seven header tables and adopted quota mapping | Serialized payload is not realized GPU residence/accuracy |

### R02 Annotation types must stay separate

The seven labels **Read, Analyze, Plan, Implement, Explore, Verify, Monitor** describe visible sentence/episode roles. Their retrospective prompt may include previous and next sentences; Verify does not itself certify a valid candidate, binding to the original problem, an unchecked opportunity or a successful substantive check.

The newer v3 judgments concern a **specific numeric/symbolic candidate and original-problem constraint**, including validity, constraint binding, already-checked status and eligibility. They are two seeded judgments from the frozen reader at a source prefix, with unknown/disagreement rules. Historical annotation scores, hidden-state probes and generic Verify labels do not calibrate this distinct instrument. Conversely, missing new candidate judgments do not mean the historical solution text, Verify labels or routes are missing. Inspect and reuse those sources before deciding what extra measurement a claim requires.

### R03 Missing, uninspected and unmeasured

| Status | Exact example | Consequence |
|---|---|---|
| **Missing/unavailable file** | Empty local SAT dataset; absent cited raw grouped-Qwen result; six missing Qwen3.5 canonical discovery receipts; 560 absent FRESH semantic pairs | Saved report/results remain usable within scope; do not fabricate records or treat them as negative outcomes |
| **Present, compatibility/validation uninspected** | Historical token/routing relation to v3; full GLM archive/accounting closeout; exhaustive legacy steering raw-result availability; historical-label ZIP contents | First inspect manifests/identities/text/tensor schema where needed; do not regenerate solely to obtain new provenance |
| **Scientifically measured FAIL** | Five completed v3 recognition qualifications and full shared J1 audit | Retain original failed gate/denominators; no causal null, relaxed threshold or automatic retry |
| **Genuinely unmeasured outcome** | New v3 treatment/utility; completed resident-removal memory/accuracy; GLM recognition; full Qwen3.5 audit; final memory-study N/outcomes | Existing observational/routing/header evidence cannot supply these outcomes; a defensible revised claim/design must precede any authorized measurement |
| **Stopped by user** | Incomplete discovery/loader work and cancelled controllers | Preserve partial artifacts and stop receipt; cancellation supplies no scientific failure or effect estimate |

### R04 Requirement for any future computation proposal

Every proposal must cite the applicable **D/X/R catalogue entries and exact source files**, identify the precise paper claim and table/figure, and state what required information those artifacts cannot supply. Distinguish a missing file from uninspected compatibility and an unmeasured endpoint. Specify independent families/effective support and expected uncertainty, defensible controls/annotation/scoring/missingness, interpretation of unfavorable or imprecise results, and the minimum measured resource cost. [ADR08](DECISIONS.md#adr-20261008-08-publication-relevance-and-necessity-before-computation) and [job-card contract](NEXT_EXPERIMENTS.md#execution-and-continuity) apply. The user stop remains controlling; a catalogue entry, old accepted card or unused reservation does not authorize computation.

### Inspection boundary and validation

This catalogue reads existing compact manifests, saved summaries/closeouts, source code, directory/file availability and a few JSON schema samples. It preserves original counts/uncertainty and separates the prior independent closeouts from new file-presence checks. It performs no generation, regrading, fitting, bulk tensor processing, scheduler submission, restart or transfer. [Validation receipt](experiment_inventory_validation.json) records count/link/source checks. Scientific/operational source seals and historical reports remain unchanged; outdated progress remains historical. No claim is made that every raw tensor, archived fit or stored solution was independently reproduced.

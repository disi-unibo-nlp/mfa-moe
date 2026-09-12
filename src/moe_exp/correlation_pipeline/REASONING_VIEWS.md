# Reexecute the reasoning analyses

Run commands from the repository root. The existing `moe-mfa-experiments:latest`
image contains the required dependencies; source files are mounted from this
checkout, so source-only updates do not require rebuilding it.

## Automated tests

Run the relevant CPU tests with networking disabled:

```bash
bash src/moe_exp/correlation_pipeline/run_docker.sh test -q \
  tests/test_reasoning_views.py \
  tests/test_correlation_launcher.py \
  tests/test_correlation_pipeline.py \
  tests/test_expert_identity_analysis.py \
  tests/test_streaming_routing_extraction.py \
  tests/test_generation_performance.py \
  tests/test_gepaLLMAsJudge.py
```

These tests do not call a live judge or download production model weights.
They cover the existing correlations and expert analysis, annotation resumption,
exact source offsets, token-boundary merges, exclusion of final-answer tokens,
class pooling, transition boundaries, position windows and overflow, missing
scope coverage, checkpoint invalidation, and repeated-attempt statistics.

## Run all new analyses on existing generations

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh --skip-generate \
  --generation-model qwen3.5-35b-a3b-mtp-ud-q4-k-xl
```

`run_all.sh` performs generation, tagging, forward replay and analysis.
`--skip-generate` reuses generations of `--generation-model`. The command above
selects the existing GGUF generations. Omit both flags to generate with the new
Qwen3.5 GPTQ checkpoint through vLLM.
The command defaults to six datasets: MATH-500, AIME24, AIME25,
OlympiadBench, AMC23 and Minerva. GPQA-Diamond and MMLU-Pro require explicit
`--datasets` selection. It performs these stages:

1. Starts `unsloth/Qwen3.8-27B-NVFP4` in vLLM and loads the frozen
   `qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json`.
2. Classifies every reasoning unit as Read, Analyze, Plan, Implement, Explore,
   Verify or Monitor, with the same question and neighboring units as context,
   using eight concurrent requests.
3. Stops its judge container before loading the Qwen3.5 target for forward replay.
4. Computes whole-reasoning, class and position features during the same replay.
5. Produces correctness/event correlations and metric-to-metric correlations
   for each view, plus the existing whole-completion and expert-identity analyses.

The selected GEPA program is used exactly as saved, including its selection
gate outcome. It is not retrained or selected again on these benchmark outputs.
The existing lexical backtracking/contradiction/self-correction flags remain
separate from the seven sentence categories.

With `--skip-generate`, all generated text and saved sample IDs are reused.
The default generation root remains `results/correlation_pipeline/generation`.
New vLLM annotations and analyses are stored under:

```text
results/correlation_pipeline/reasoning-vllm-v1/
  annotations/<generation-model>/<dataset>/
    shards/<trace-hash>.json
    annotations.jsonl
    manifest.json
  forward/<target-model>/
    position_reference.json
    <dataset>/traces_with_routing.jsonl
    <dataset>/tensors/*_extraction.json
  analysis/<target-model>/
    correlations.json
    views-v1/
      full/reasoning/
      class/Read/ ... class/Monitor/
      position/bin_00/ ... position/bin_09/ position/overflow/
      summary.json
```

Each view directory contains feature CSVs, correlation CSVs, and
`correlations.json` with coverage and statistical details. Compact per-view
features and sentence token ranges are stored in `metadata.correlation_views`.
Raw router and hidden-state tensors remain disabled by default.

Inspect the planned stages without executing them:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh --skip-generate --dry-run
```

Count annotation units before making model calls:

```bash
bash src/moe_exp/correlation_pipeline/run_docker.sh annotate \
  --judge-program results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3/selected_program_20260827_173300.json \
  --generation-model qwen3.5-35b-a3b-mtp-ud-q4-k-xl \
  --judge-model unsloth/Qwen3.8-27B-NVFP4 --dry-run
```

For a functional smoke run with one complete saved trace:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --skip-generate --generation-dir results/correlation_pipeline/generation \
  --generation-model qwen3.5-35b-a3b-mtp-ud-q4-k-xl \
  --datasets math500 --limit 1 --bootstrap-samples 20 \
  --results-dir results/correlation_pipeline/smoke
```

A trace can contain many annotation units. This smoke command exercises the
real models; the automated tests above are the quick check. One trace cannot
support correlation estimates, so empty correlation tables are expected.
`--limit` never changes sample IDs or pretends that an incomplete avg@32 group
is complete.

## Run without sentence labels

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh --skip-tagging
```

`--skip-tagging` skips the judge and class views. It preserves generation,
forward extraction, whole-reasoning and fixed-position views, and the standard
correctness, metric-pair and expert analyses. It does not read annotation files.
Outputs are stored under `results/correlation_pipeline/reasoning-vllm-untagged-v1`.
Use the same flag on later resumes and analysis-only runs to select those outputs.
To add class analyses later, omit the flag: the saved generations can be reused,
then tagging and forward replay produce the class features under the tagged root.

`--skip-annotate` has a different purpose: it reuses completed matching labels
and still runs class analyses. It does not allow missing labels.

For the current partially generated vLLM run, interrupt the existing tmux command
before relaunching with `--skip-tagging`. Keep the same generation settings and
output root so completed shards are reused; omit `--skip-generate` until all
selected datasets are complete. Flags added to the file do not alter a running
command's options.

## Resume and rerun

The previous GGUF judge outputs remain under `reasoning-v1`; vLLM results are
separate. Rerun the same command after interruption. Each labeled sentence is saved;
matching annotations and forward checkpoints are reused. Changes to the
generation text/context, classifier program/settings, labels, view schema,
position reference, or feature settings invalidate the relevant checkpoints.

Once annotation is complete, skip loading the judge:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh --skip-generate --skip-annotate \
  --generation-model qwen3.5-35b-a3b-mtp-ud-q4-k-xl
```

Once forward extraction is also complete, rerun just the statistics:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --skip-generate --skip-annotate --skip-forward --bootstrap-samples 500
```

Pass the same `--datasets`, `--results-dir` and any `--generation-dir` override
when resuming a custom run. `--results-dir DIR` stores generated traces under
`DIR/generation` and annotations/forward/analysis under `DIR/reasoning-vllm-v1`.
Use `--judge-program` and `--judge-model` to choose a different classifier,
with a separate result root for comparisons. `--judge-workers` changes request
concurrency without changing the classifier fingerprint. The wrapper replays the current Qwen3.5
target; other generation/target pairs can use the individual stage CLIs.

## Individual stages and analysis contract

`run_docker.sh annotate`, `forward`, and `analyze` expose `--help`.
The forward and analyze stages enable the new work with
`--views full class position`; forward also needs `--annotation-dir` for class
views. Existing commands without `--views` retain their previous behavior.
Use separate `--output-dir` paths when manually replaying existing baselines.

- **Reasoning text:** explicit saved reasoning content or the `<think>` body is
  used when available. For models without a separate reasoning field/tag, the
  entire completion is the available fallback.
- **Segmentation:** deterministic punctuation/newline units with exact character
  offsets; protected LaTeX expressions and decimal points remain intact. This
  segmentation has a schema version and is not claimed to be a linguistic parser.
- **Token ownership:** offsets come from the same jointly tokenized prompt and
  completion as forward extraction. Maximum character overlap assigns boundary
  tokens; earlier units win ties and whitespace gaps attach to the preceding unit.
  Tokens overlapping reasoning are counted exactly once. A tokenizer without
  exact offsets causes an explicit error.
- **Classes:** token-weighted pooling within each class and attempt. Switching,
  top-k overlap and trajectory distance use only adjacent original tokens within
  the same sentence. Disconnected spans create no artificial transitions.
- **Positions:** fixed, non-overlapping intervals with boundaries
  `ceil(model_mean_reasoning_tokens * i / 10)`. The reference uses all supplied
  generations for that model, independently of extraction `--limit`. Longer
  traces have an explicit overflow window beyond the mean. It is not per-trace
  percentage normalization. Changing the dataset set changes the recorded
  reference and invalidates the view checkpoints.
- **Missing scopes:** no tokens means missing routing/hidden metrics. Coverage
  is reported; an absent class is never assigned a zero-valued confidence.
- **Statistics:** one observation per attempt and view, with original source
  problem groups retained for bootstrap and avg@n completeness checks. A
  problem-level feature requires values for all required attempts; class/window
  absence can reduce the eligible sample size. Token count is excluded from
  position correlations. Metric-pair correlations retain the existing
  prespecified across-layer feature set. Multiplicity correction is per
  view/dataset/target or metric-pair family; comparisons across views remain
  exploratory.

Routing interventions are a subsequent experiment whose settings depend on
these correlation results; this rerun performs the observational analyses.

## Six benchmarks with one solution per repeated problem

The corrected sample retains every single-attempt problem from MATH-500,
OlympiadBench, and Minerva, and selects sample ID 0 for each AIME24, AIME25,
and AMC23 problem. This gives 1,547 original solutions and 927,496 eligible
reasoning sentences from the current vLLM generations. Only the avg@32
benchmarks are reduced to one solution per problem.

The 100,000-sentence budget is allocated in proportion to each benchmark's
all-attempt **error rate** from the September 10 untagged analysis:

| Benchmark | Retained solutions | Selected sentences |
|---|---:|---:|
| MATH-500 | 500 | 16,217 |
| AIME24 | 30 | 7,580 |
| AIME25 | 30 | 9,258 |
| OlympiadBench | 675 | 27,104 |
| AMC23 | 40 | 1,624 |
| Minerva | 272 | 38,217 |
| Total | 1,547 | 100,000 |

Sentences are sampled uniformly without replacement within each benchmark,
using seed 42. Individual sentence selection is not conditioned on correctness.
All original problem records remain present even when none of a problem's
sentences were drawn; those class metrics are missing, not zero.

Run or resume with:

```bash
bash src/moe_exp/correlation_pipeline/run_sampled_tagging.sh
```

Outputs go to
`results/correlation_pipeline/sentence-tagging-six-benchmarks-error-weighted-v2`.
The `sampling_manifest.json` records exact selected indices, source
fingerprints, weights and quotas. `status.json` and `pipeline.log` record
progress. The runner prevents duplicate launches through a file lock.
It reuses completed sentence checkpoints with the same source and classifier.

The frozen judge labels only selected sentences, with their original
preceding/following sentences as context. Unselected sentences receive no
placeholder labels. Replay preserves the complete original continuation and
pools only sampled sentence tokens into class features. Transitions remain
restricted to each original sentence. No new solutions are generated.

Tagging is followed by class-feature replay and class correctness/metric-pair
analyses with 500 problem bootstraps. Sampled class results are under
`reasoning-vllm-v1/analysis/<model>/views-v1/class/`; standard whole-continuation
outputs are ancillary and expert identity analysis is skipped. Original
avg@32 metadata remain intact, so one-attempt subsets are excluded from
repeated-attempt statistics. Sentences are not independent correctness trials.

The earlier three-benchmark sample under
`sentence-tagging-one-solution-error-weighted-v1` was based on a scope
misinterpretation and is superseded. Its outputs are retained for provenance.
The older 1,228,263-sentence inventory referred to GGUF generations, whereas
this sample uses the newer vLLM generations.

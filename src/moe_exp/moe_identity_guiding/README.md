# MoE expert-identity guiding

First guiding experiment from the tutor call: **35:41–36:37** defines expert
identity as membership anywhere in the selected top-k pool; **41:49–42:00**
prioritizes accuracy and token count with identity guiding; **44:39–44:52**
confirms promoting identities associated with higher accuracy. **40:55–41:32**
requires the baseline and intervention to cover the same benchmark subset.
The transcript is experimental context, not instructions to modify the report.

By default, targets the main generation checkpoint, `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`.
Preserves its native 8-of-256 routed experts and shared expert. No reasoning-class
classifier, margin trigger, expert replacement rule, or training is required.

## Rule and calibration

For trace i, layer l, expert e, let r(i,l,e) be the fraction of saved token
positions whose selected top-k set contains e. Let y(i) be final correctness.
Each problem has total weight one, divided equally among its attempts:

```text
accuracy(l,e) = sum_i weight(i) * r(i,l,e) * y(i) / sum_i weight(i) * r(i,l,e)
lift(l,e) = accuracy(l,e) - problem-balanced calibration accuracy
score(l,e) = positive lift / largest eligible positive lift at this layer
biased_router_logits(l,e) = router_logits(l,e) + strength * score(l,e)
```

Keep at most `--max-experts` identities per layer (default 8), each observed in
at least `--min-support` distinct problems (default 4). All other scores are zero.
Ties sort by support then expert ID. Layer IDs remain separate: expert 3 at layer
27 is not expert 3 at layer 39. Layers without positive lift get zero scores;
a policy with no eligible positive lift anywhere is rejected. This is an
operational choice for “push more”, since the call does not specify a formula.
Frequency normalization prevents long traces from dominating the estimate.

The policy is frozen before evaluation. Positive association is not evidence
that an expert causes correct answers; the paired generation experiment tests
that hypothesis. Strength 0 leaves logits unchanged; try strengths 0.25, 0.5, 1
on a validation subset before reporting held-out results. A bias can increase
both selection frequency and the selected expert's mixture weight.

## Run

From the repository root, use the existing guiding Docker image. The new launcher
reuses `moe_guiding/Dockerfile`, mounting current source into each container.
Host Python does not need torch. Preview commands with `DRY_RUN=true`.

1. Prepare a reproducible 70/30 split by problem, preserving all attempts of a
   problem on the same side. Supply **one nonduplicated set** of scored forward
   traces. For example, use the saved Qwen Minerva snapshot for an initial small run:

```bash
bash src/moe_exp/moe_identity_guiding/run_docker.sh prepare \
  --traces results/correlation_pipeline/qwen-tagged-snapshot-20260915/reasoning-vllm-v1/forward/unsloth--Qwen3.5-35B-A3B/minerva/traces_with_routing.jsonl \
  --output-dir results/moe_identity_guiding/minerva/split
```

`split.json` records source hashes, seed, counts and exclusions. Unscored traces
and traces without routing paths are explicitly excluded. The evaluation
population is therefore the scored subset, not the entire original benchmark.
`prepare` preserves original messages/system prompt and answer type. Multiple
files can follow `--traces` to calibrate across datasets. For a formal benchmark,
choose calibration and evaluation datasets yourself instead of this small split.

2. Fit a policy; JSON stores every expert's score, support, frequency-weighted
   accuracy, input/tensor hashes, and extraction checkpoint metadata:

```bash
bash src/moe_exp/moe_identity_guiding/run_docker.sh fit \
  --traces results/moe_identity_guiding/minerva/split/calibration.jsonl \
  --output results/moe_identity_guiding/minerva/policy.json
```

Tensor paths are resolved relative to the repository root by default; use
`--tensor-base-dir` if needed. Input follows correlation `TraceRecord` JSONL:
`dataset`, `problem_id`, `source_problem_id`, boolean `is_correct`,
`model_logs.selected_experts` pointing to a torch tensor `[layers,tokens,top_k]`,
and `model_logs.layer_indices`. `fit` rejects unscored traces and malformed IDs.
Calibration must contain both correct and incorrect answers. The AIME25 tagged
snapshot has no scored incorrect answers and cannot identify accuracy lift alone.
It uses all positions in the saved tensor (existing extraction saves completion
positions), without filtering by reasoning class.

**Existing data provenance:** this snapshot's generations are GPTQ Qwen, but its
routing was replayed through `unsloth/Qwen3.5-35B-A3B` with `unsloth-4bit`.
Using that policy on GPTQ assumes corresponding expert identities across these
Qwen variants. Extraction configs are saved in the policy so this transfer is
visible. Re-extract routing on the exact generation checkpoint for a strict
same-checkpoint experiment. Never transfer IDs to unrelated checkpoints.

3. Generate both conditions in fresh processes on the same prompts:

```bash
for condition in baseline guided; do
  bash src/moe_exp/moe_identity_guiding/run_docker.sh generate \
    --policy results/moe_identity_guiding/minerva/policy.json \
    --prompts results/moe_identity_guiding/minerva/split/prompts.jsonl \
    --condition "$condition" --strength 1 \
    --output-dir "results/moe_identity_guiding/minerva/$condition"
done
```

Default: temperature 0.6, top-p 0.95, top-k disabled, 32,768 output tokens,
49,152 total context, one sequence. The global launcher uses concurrency 16.
The token limits match the Qwen correlation generation pipeline, leaving context
space for the prompt. Set `--revision` to pin a checkpoint;
use identical model, revision, sampling and engine options for both conditions.
The main 35B checkpoint requires suitable GPU memory; this is not the tiny
Mixtral RTX3070 smoke experiment. Quantized weights use the native vLLM backend.
No checkpoint downloads or GPU generations are performed by `prepare` or `fit`.

The new run regenerates the baseline to match exact messages and sampling.
Calibration overlap is rejected by problem ID unless explicitly enabled with
`--allow-calibration-overlap`; such runs are marked in-sample in the manifest.
Keep dataset/source_problem_id in custom prompt files for this check. Generic
prompts need unique string `id`, `prompt`, and `gold_answer` for scoring.

4. Compare final-answer accuracy, generated-token counts, truncations, and
   paired wrong→right / right→wrong changes:

```bash
bash src/moe_exp/moe_identity_guiding/run_docker.sh compare \
  --baseline results/moe_identity_guiding/minerva/baseline \
  --guided results/moe_identity_guiding/minerva/guided
```

Comparison rejects mismatched IDs, rendered token inputs, policies, engine or
sampling settings, software versions, scoring contract, and unscored answers. Uses the correlation
pipeline's answer scorer (math-verify when available, otherwise its recorded
numeric/exact fallback); choice questions use exact boxed-letter scoring.
Outputs include `manifest.json` and `generations.jsonl`; existing runs/policies
cannot be overwritten. Token count includes thinking and final-answer text.

## Integration and verification

Follows `moe_guiding`'s eager instrumentation, worker RPC diagnostics, matched
conditions and manifests. Qwen uses a hook on each selected `mlp.gate` output,
installed via worker RPC **after warmup** and before generation. The native
expert-selection kernel receives the biased logits; expert network parameters are not
modified. Baseline installs no hooks. The hook affects prefill and decode.

The implementation was checked against [vLLM 0.29.0 Qwen's native gate](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/qwen3_next.py)
and [MoE runner](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/layers/fused_moe/runner/moe_runner.py).
Eager execution is required; EP/EPLB, PP/DP, DBO, speculative decoding and fused
shared-gate paths are rejected. A run fails if any selected layer's hook never
executes. Diagnostics record token-layer evaluations, before/after top-k identity
counts and changed sets. Counts are from logits, not a separately observed native
kernel output; tied selection ordering can differ. Do not sum replicated TP
worker counts as independent tokens. These counters include engine padding.

CPU tests cover learning, support filtering, layer identity, top-8 selection,
hooks, worker configuration, paired scoring, and problem-disjoint preparation:

```bash
python3 -m pytest tests/test_moe_identity_guiding.py -q
```

Full Qwen GPU generation must still be validated on the target machine; CPU
and mocked-runner tests do not establish an accuracy improvement.

## All Qwen datasets with one global policy

`run_global.sh` uses the complete saved GPTQ generation population for AIME24,
AIME25, AMC23, MATH500, Minerva and Olympiad. It splits **problems within each
dataset** into 70% calibration and 30% evaluation with seed 42. Thus this covers
all datasets, while the reported comparison is on their held-out problems,
not on calibration questions. The fraction is configurable with
`CALIBRATION_FRACTION` during preparation.

One shared policy is fitted from all available scored calibration routing
traces across the six datasets. Each problem has equal total weight, divided
among its scored attempts; larger datasets therefore have more total weight.
The `single`/`full` choice affects only evaluation, not policy fitting. Every
scored calibration attempt is retained. Missing calibration tensors and missing
correctness labels are counted in `split.json`. Evaluation eligibility does not
depend on saved correctness or the existence of routing tensors.

Run the complete workflow in a new output directory:

```bash
bash src/moe_exp/moe_identity_guiding/run_global.sh all
```

Or run the stages individually:

```bash
bash src/moe_exp/moe_identity_guiding/run_global.sh prepare
bash src/moe_exp/moe_identity_guiding/run_global.sh fit
bash src/moe_exp/moe_identity_guiding/run_global.sh generate
bash src/moe_exp/moe_identity_guiding/run_global.sh compare
```

Defaults: all saved attempts per held-out problem, concurrency 16, temperature
0.6, top-p 0.95, 32,768 output tokens and 49,152 total context.
Both conditions use identical settings and the same policy. Existing outputs
are never overwritten; `all` reuses existing split/policy files and then runs
generation and comparison. It still refuses to overwrite existing generation runs.
To rerun generation without redoing calibration, choose a fresh `OUTPUT_ROOT`
and copy the existing `split/` and `policy.json` there first.

To repeat **every saved attempt** for the same held-out problems, reusing the
same global policy and split:

```bash
bash src/moe_exp/moe_identity_guiding/run_global.sh generate full
bash src/moe_exp/moe_identity_guiding/run_global.sh compare full
```

This preserves saved attempt counts and recovers the original request seeds and
chat-template options from the generation traces. `prepare-sampling` checks
matching prompt identities/content and writes `prompts.full.sampling.jsonl`.
The launcher requires the sampling parameters to match the original traces.
Use explicit `single` as the second positional argument for one attempt per problem.
Aggregate full-mode accuracy is attempt-weighted; datasets with 32 attempts per
problem have greater weight in that aggregate than single-attempt datasets.

Outputs: `results/moe_identity_guiding/qwen_global/{split,policy.json,sampling}`.
New generations live under `sampling/full/{baseline,guided}`; previous greedy
`single/` and `full/` outputs remain untouched.
`split/` contains both `prompts.single.jsonl` and `prompts.full.jsonl`.
Override `OUTPUT_ROOT`, `GENERATION_ROOT`, `ROUTING_ROOT`, `MAX_NUM_SEQS`, or
`STRENGTH` as needed. `DRY_RUN=true` previews Docker commands without launching.
The same Unsloth routing-replay to GPTQ generation provenance caveat applies.


## OSS and Gemma launch presets

Select `MODEL_PROFILE=qwen` (default), `oss`, or `gemma` when invoking
`run_global.sh`. These presets choose separate generation/routing paths,
policy output directories and checkpoints. `fit` infers the correct expert
counts for these checkpoint families; inference workers check those counts
against the loaded checkpoint configuration.

| Profile | Generation checkpoint | Native routed experts | Hook location |
|---|---|---|---|
| `qwen` | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | 8 of 256 | `layers.N.mlp.gate` |
| `oss` | `openai/gpt-oss-20b` | 4 of 32 | `layers.N.mlp.router` |
| `gemma` | `nvidia/Gemma-4-26B-A4B-NVFP4` | 8 of 128 | `layers.N.router` |

OSS and Gemma's vLLM routers return raw logits directly, whereas Qwen's gate
returns a tuple. The hook handles both without replacing native dispatch.
Gemma's router normalization/projection and per-expert output scales are retained,
as is its parallel dense MLP. OSS's router bias and native mixture weighting are
retained. OSS uses text-only native loading (no `language_model_only` override);
Qwen and Gemma enable language-model-only loading. The OSS ROCm path bypasses
its router module and is explicitly rejected by this adapter.

After the correlation pipeline has saved routing tensors, launch later with:

```bash
MODEL_PROFILE=oss bash src/moe_exp/moe_identity_guiding/run_global.sh all
MODEL_PROFILE=gemma bash src/moe_exp/moe_identity_guiding/run_global.sh all
```

These commands run preparation, fit one global policy **for that model**, generate
both conditions and compare. Run one model at a time on the shared GPU. Stage
selection and `full` attempt mode work as before, for example:

```bash
MODEL_PROFILE=oss bash src/moe_exp/moe_identity_guiding/run_global.sh prepare
MODEL_PROFILE=oss bash src/moe_exp/moe_identity_guiding/run_global.sh fit
MODEL_PROFILE=oss bash src/moe_exp/moe_identity_guiding/run_global.sh generate full
MODEL_PROFILE=oss bash src/moe_exp/moe_identity_guiding/run_global.sh compare full
```

Default routing roots (override `ROUTING_ROOT` if extraction used another location):

- OSS: `results/correlation_pipeline/gpt-oss-20b/reasoning-vllm-v1/forward/openai--gpt-oss-20b`
- Gemma: `results/correlation_pipeline/gemma-nvfp4-nf4/reasoning-vllm-v1/forward/google--gemma-4-26B-A4B-it`

Results go to `results/moe_identity_guiding/oss_global` or `gemma_global`.
Each preset retains concurrency 16 and the 32,768/49,152 output/context limits;
GPU capacity must be checked for each checkpoint. `MAX_NUM_SEQS=8` reduces
concurrency if needed. `MODEL`, `OUTPUT_ROOT`, `GENERATION_ROOT` and
`ROUTING_ROOT` can override the preset paths/checkpoint. Changing checkpoint
requires a matching policy; expert identities cannot be shared across models.

The configured Gemma workflow transfers routing measured on Google's NF4 replay
to NVIDIA's NVFP4 generation checkpoint. This assumption is recorded via saved
extraction provenance, just as with Qwen's two quantized variants. No labels from
reasoning-category tagging are required for policy learning.

Adapters were checked against the installed vLLM 0.29.0 model source and cached
checkpoint configs, and covered by CPU tests for native router shapes, layer
selection, expert counts, and launch presets. End-to-end GPU guiding for OSS and
Gemma still needs a first smoke run; no new GPU jobs were started during this update.


## Matched original stochastic evaluation

The default global workflow now reproduces the original correlation sampling:
`temperature=0.6`, `top_p=0.95`, `top_k=0` (converted to vLLM's disabled value
`-1`), and 32,768 maximum output tokens. Request seeds are read from the original
trace metadata, preserving `base_seed + original_problem_index * 1000 + sample_id`.
The generation CLI `--seed` defaults to zero and acts as an offset on those saved
seeds. Both conditions receive identical per-attempt settings; they are saved in
every generated record and validated by `compare`. Original Gemma
`chat_template_kwargs`, including `enable_thinking=true`, are applied.

`full` is the default: 32 attempts for AIME24/AIME25/AMC23 and one for the other
three datasets. This retains the existing held-out problem split and fitted policy.
It does not expand evaluation onto calibration problems. No existing greedy
completion is included in the new stochastic comparison.

For the already prepared Qwen policy:

```bash
bash src/moe_exp/moe_identity_guiding/run_global.sh generate
bash src/moe_exp/moe_identity_guiding/run_global.sh compare
```

For a fresh OSS/Gemma experiment, use `MODEL_PROFILE=oss` or `MODEL_PROFILE=gemma`
with `run_global.sh all`. `all` skips existing preparation/fitting artifacts;
it resumes compatible incremental runs and skips validated completed runs.
No GPU generation is launched by changing these defaults.


## Faster execution, baseline reuse, and recovery

Global launchers now use `--diagnostics minimal` and `--resume`. Minimal diagnostics
retain per-layer hook call/token counts and the mandatory hook-activity checks,
without extra top-k comparisons, expert histograms, or post-intervention margin
statistics. The intervention arithmetic is unchanged. Use `DIAGNOSTICS=full` on
a fresh run (or `generate --diagnostics full`) for detailed router validation.
Unavailable statistics are omitted, rather than reported as zero.

Before loading a baseline model, generation searches completed runs beneath
`results/moe_identity_guiding` and `results/moe_margin_guiding`. Reuse requires
matching model/revision, engine settings, software versions, prompt-file hash,
sampling configuration, and every saved attempt's input and seed. The source
must have complete coverage and diagnostics proving no guiding hooks were
installed. Different guiding policies and the two known worker-extension names
are allowed, since baseline execution applies neither policy. Unknown worker
extensions are rejected. This search applies to all model profiles, not just Qwen.
`--baseline-search-root PATH` overrides the search roots and can be repeated;
`--no-reuse-baseline` requests a fresh baseline in a fresh output directory.

A reused baseline is copied into the destination, preserving its source manifest
and content hashes in `reused_baseline`. The destination records the current
comparison policy; the nested source manifest records the actual originating
execution. Comparison verifies the reused completion hash and source compatibility.
The source run remains intact. Existing completed destinations are validated and
skipped, not replaced by a different baseline.

Generation retains vLLM continuous batching and writes each completed, scored
attempt to `generations.jsonl`, flushing and syncing it to disk. The manifest
records `completed_count` and `expected_count`. Completions are saved in completion
order; comparison joins by ID. No partial token sequence is checkpointed.

Rerun the same global command after interruption, or pass `--resume` to `generate`.
Saved attempts are validated and skipped; unfinished attempts restart with their
original seeds. Resume refuses changed run settings, duplicate IDs, incompatible
records, or another writer holding the run lock. A torn final JSONL line is
removed on resume; corruption in complete lines is rejected. Scheduling after a
restart can differ, so seeds alone do not guarantee bit-identical regenerated
answers. Hook reports from earlier sessions are retained separately.

Incomplete legacy runs without the incremental execution marker cannot be
resumed automatically: they may still be executing in an older process. Let them
finish, or use a fresh output directory after stopping them. These code changes
do not alter processes that are already running.


### Signed strength controls

`STRENGTH` (CLI `--strength`) accepts any finite signed value. Positive values
favor the policy's selected experts; `2` doubles the original bias. `-1` subtracts
that same bias, penalizing the favored experts. This is a reversed-bias control,
not a separately learned policy selecting experts associated with incorrect
answers. Zero leaves native router logits unchanged. Margin guiding retains its
separate `[0, 1]` strength restriction.

Use a separate output root for each strength and copy the original policy and
split to keep evaluation problems fixed. Compatible baseline reuse is independent
of strength because baseline execution installs no intervention hooks. Treat a
strength sweep on an already examined evaluation set as exploratory.

### CPU-only paired uncertainty

Both identity and margin `compare` commands calculate 95% percentile intervals
for guided-minus-baseline accuracy, overall and per dataset. Defaults are
5,000 resamples, seed 42, and up to eight CPU worker processes. Use
`--bootstrap-replicates`, `--bootstrap-seed`, and `--bootstrap-workers` to
configure them. Whole source problems are sampled with replacement within
each dataset, keeping all attempts and both conditions together. Pooled
estimates retain attempt weighting. Streams are indexed by replicate, making
results independent of worker count. Intervals are not multiplicity-adjusted;
a stratum with fewer than two problems cannot supply an interval. Small
problem counts and conditioning on the saved policy/attempts limit inference.

Recompute all complete saved identity and margin pairs without GPU inference:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m moe_exp.moe_identity_guiding.bootstrap_saved \
  --bootstrap-workers 8 --bootstrap-replicates 5000 --bootstrap-seed 42
```

Each pair receives `comparison_bootstrap.json`, including input hashes,
point estimates, problem counts, and intervals. Incomplete pairs are skipped;
mismatched complete pairs fail validation. Original generations are unchanged.
The BLAS thread limits avoid nested oversubscription; eight separate worker
processes perform the bootstrap. Regenerate report tables afterwards with
`python3 report/generate_current_results.py`.


### Frozen template dates and exact baseline compatibility

GPT-OSS inserts the current date using its chat template clock. Rendering now
replaces that clock expression with a fixed date without modifying the cached
tokenizer template. Set `TEMPLATE_DATE=YYYY-MM-DD` in the global launcher or
`--template-date YYYY-MM-DD` in the generation CLI. With no override, existing
saved/paired dates are inherited; fresh runs use the fixed reference 2026-09-22.
The selected date is saved in the manifest.

Before GPU loading, generation renders and tokenizes prompts on CPU. Reuse checks
every saved rendered prompt and token sequence, including legacy baselines without
a date field. A paired-condition mismatch fails before generation. The engine's
own tokenizer must reproduce the same inputs before requests are submitted.

The two mismatched September-21 baseline copies for OSS identity strength 2 and
margin strength 1 were moved to `results/guiding_archives/oss_date_mismatch_20260922`.
Their valid September-22 guided outputs remain intact. From the repository root,
run `bash src/moe_exp/moe_identity_guiding/repair_oss_date.sh` to generate one
September-22 baseline, reuse it for the other experiment, and compare both runs.
The script is resumable and does not regenerate completed guided responses.

## Negative experts and paper-style intervention

Two independent options are frozen in the calibration policy:

- `--expert-polarity positive|negative` (launcher: `EXPERT_POLARITY`, default
  `positive`). Negative selects the most negative accuracy lifts, with the
  same support and per-layer limits. Scores store normalized magnitudes:
  positive strength **promotes failure-associated experts**; negative strength
  suppresses them. Neutral experts are excluded.
- `--guiding-method fixed|paper` (launcher: `GUIDING_METHOD`, default `fixed`).
  Paper mode implements [SteerMoE section 3.2](https://arxiv.org/html/2509.09660v1#S3.SS2):
  convert original logits to log-softmax scores, then simultaneously set
  selected experts just above the original maximum (strength 1) or below the
  original minimum (strength -1). Strength 0 is an exact no-op; other strengths
  are rejected. `--paper-epsilon` / `PAPER_EPSILON` defaults to 0.01.
  All selected experts receive the same target, without weighting by lift.
  If native dtype rounding erases the gap, the target moves to the next
  representable value beyond the extreme, preserving strict ordering; this
  can make the realized gap larger than epsilon, especially in BF16.
  Selecting more identities than native top-k cannot force all of them in.

This reproduces the paper's **intervention**, retaining our problem-balanced
accuracy-lift selection rather than its contrastive risk-difference estimator.
Older policies without these fields retain positive/fixed behavior. Settings
are part of the policy hash, so resume and comparison reject policy changes.

Promote failure-associated experts using the existing fixed bias:

```bash
EXPERT_POLARITY=negative bash src/moe_exp/moe_identity_guiding/run_global.sh all
```

Promote failure-associated experts using paper steering:

```bash
EXPERT_POLARITY=negative GUIDING_METHOD=paper STRENGTH=1 \
  bash src/moe_exp/moe_identity_guiding/run_global.sh all
```

Use `EXPERT_POLARITY=positive GUIDING_METHOD=paper` to promote success-associated
experts; add `STRENGTH=-1` to deactivate the selected set. Prefix with
`DRY_RUN=true` to preview commands. `MAX_EXPERTS` and `MIN_SUPPORT` control fitting.

For every nondefault polarity/method, the global launcher automatically appends
`variants/<polarity>_<method>[_eps_<epsilon>]/strength_<strength>` to OUTPUT_ROOT,
even when OUTPUT_ROOT is explicitly supplied. Splits, policies, generations,
and comparisons therefore live separately from existing results. Direct CLI
users choose fresh `--output` / `--output-dir` paths themselves; existing
artifacts remain protected against overwrite. Reusing the same preparation
seed and population gives the same held-out problems across variants.

The existing paired, dataset-stratified problem bootstrap reports uncertainty
for the accuracy change. A negative estimate alone does not establish degradation:
inspect its 95% interval (an interval entirely below zero supports a decrease).
These intervals are conditional on the saved runs and do not isolate decoding
seed variability or adjust for trying multiple variants. A decrease is a
hypothesis to test, not a guaranteed outcome.

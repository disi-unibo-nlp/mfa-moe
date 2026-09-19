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
it does not automatically resume or overwrite existing generation output files.
No GPU generation is launched by changing these defaults.

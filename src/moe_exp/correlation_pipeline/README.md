# Correlation pipeline

This pipeline generates reasoning attempts through vLLM, tags their sentences
with the frozen GEPA-selected classifier, teacher-forces the saved attempts
through the matching Hugging Face MoE checkpoint, and measures associations
among correctness, reasoning events, hidden-state trajectories and routing.

`run_all.sh` uses eight concurrent requests for both generation and tagging.
Tagging logs overall selected-sentence totals, percentage complete, remaining
sentences, reused checkpoints, processing speed, and estimated time remaining
every 10 seconds during concurrent requests and at dataset boundaries. Routine
LiteLLM/HTTP request logs are suppressed; warnings and errors remain visible.
On resume, only matching validated labels count as already complete; ETA uses
newly completed sentences from the current run.

Each vLLM server enables prefix caching, allows eight sequences, and batches up to
8,192 tokens. The default Qwen pair uses MTP with three speculative tokens. The servers run one
after another on the same GPU and stop before forward extraction begins.

Generation uses `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`, including its saved MTP head.
The judge uses `unsloth/Qwen3.8-27B-NVFP4` and the existing frozen GEPA program.
These retain the original base models with vLLM-compatible quantizations for
the 32 GB RTX 5090. The judge uses `--enforce-eager`, FP8 KV cache and text-only
loading for memory headroom. Forward extraction still loads
`unsloth/Qwen3.5-35B-A3B` through Unsloth with runtime 4-bit quantization.

Generation retains its 32,768-token completion budget and a 49,152-token context
to leave room for prompts. Judge context is 32,768; its temperature, thinking
mode, reasoning effort and 4,096-token completion budget are unchanged.

The quantized checkpoints differ from the old GGUF artifacts. New generations
use their actual checkpoint name, and vLLM annotations/analyses live under
`reasoning-vllm-v1`. Existing `reasoning-v1` labels are retained as a separate
run; they are not treated as outputs from the new quantized judge.

## Benchmark protocol

The first eight benchmarks follow the SPIRAL evaluation suite. Generation is
zero-shot with temperature 0.6 and top-p 0.95. The current completion budget is
32,768 tokens (increased from the original 8,192-token pilot).

| Name | Source / configuration | Sampling |
|---|---|---:|
| `math500` | `HuggingFaceH4/MATH-500`, test | pass@1 |
| `aime24` | `math-ai/aime24`, test | avg@32 |
| `aime25` | `math-ai/aime25`, test | avg@32 |
| `olympiad` | `Hothan/OlympiadBench`, `OE_TO_maths_en_COMP` | pass@1 |
| `amc23` | `math-ai/amc23`, test | avg@32 |
| `minerva` | `math-ai/minervamath`, test | pass@1 |
| `gpqa_diamond` | public Simple Evals Diamond CSV | avg@10 with seed-0 permuted choices |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro`, test | pass@1 |

GPQA-Diamond follows SPIRAL's released evaluator: each question is attempted ten
times with a fresh deterministic permutation of its four answer choices. The
attempts remain grouped under the same source question during analysis.

The defaults are `math500`, `aime24`, `aime25`, `olympiad`, `amc23`, and `minerva`.
GPQA-Diamond and MMLU-Pro remain available through explicit `--datasets` selection.
The four existing
repository loaders remain available by explicit selection: `gsm8k`, `math`,
`prm800k`, and `processbench`. The last two provide reference process labels in
their original use, but newly generated answers have no aligned gold error step.
ProcessBench generation is therefore retained for representation/routing and
reasoning-event analysis and is explicitly unscored for final-answer accuracy.

`math-verify` performs symbolic math scoring. Multiple-choice datasets require
the requested `\boxed{LETTER}` answer. A normalized exact/numeric fallback is
recorded explicitly if symbolic parsing cannot score an answer.

## 1. Generate with vLLM

The host Python on the current machine is 3.10, while the project requires
Python 3.11. Use the Docker launcher below, following the same pattern as
`probeTest/run_slurm.sh`; do not run `pip install -e .` on the host.

Build or refresh the project image once from the repository root:

```bash
docker build -t moe-mfa-experiments:latest .
```

### One-command full experiment

Install the separate, pinned inference-server image:

```bash
docker pull vllm/vllm-openai:v0.29.0
bash src/moe_exp/correlation_pipeline/run_all.sh
```

The project image runs the Python stages; vLLM runs in its own image. The shared
`/llms` Hugging Face cache is mounted into both. Models download there on first
use. `VLLM_IMAGE` can override the server image.

The stages are generation, sentence sampling, tagging, forward replay, and analysis.
Before tagging, the launcher selects one attempt per problem (`sample_id=0` for
repeated benchmarks) and at most 100,000 sentences across the selected benchmarks.
Benchmark quotas are proportional to error rate, computed from all scored source
attempts; sentences are sampled uniformly within each benchmark with seed 42.
The existing proportional allocation can select fewer sentences when supply is
limited. Sampling fails if accuracy is unavailable or all error rates are zero.
Use `--max-sentences N` to change the cap. The manifest and sampled generations
are saved under `reasoning-vllm-v1/sampling`; tagging and all forward views use
those generations, while the original generation files remain intact.
`--skip-annotate` reuses that sampled input and its annotations. Use
`--skip-sampling` only when intentionally supplying inputs directly (as the
separate `run_sampled_tagging.sh` wrapper does). `--skip-tagging` bypasses
sampling and keeps the full-data workflow. A changed sampling plan requires a
new results directory.

Analysis
includes correctness/events, metric-pair correlations, expert identities/pairs,
and full-reasoning, class and position views. Both vLLM servers listen inside
their container on `0.0.0.0:41800`; the local API is
`http://127.0.0.1:41800/v1`. Only one server runs at a time.

Use `--model` to select the checkpoint for generation, forward replay, and
analysis while keeping the tagging model and frozen GEPA program unchanged:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --model ORG/MODEL \
  --results-dir results/correlation_pipeline/my-model
```

If generation needs a separate quantized checkpoint of the same model, also
pass `--generation-model ORG/MODEL-QUANTIZED`. This option (or the
`GENERATION_MODEL` environment variable) overrides generation only, regardless
of option order. The generation manifest records `--model` as the matching
forward target. Use the same options when resuming saved runs.

Omitting `--model` retains generation's `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` and
forward/analysis's `unsloth/Qwen3.5-35B-A3B` defaults. Model profiles now select
the server parser, MTP behavior, context default, and forward loading for
Qwen3.6, Qwen3.5, Qwen3, Nemotron 3.5 Lightning, Gemma4, GLM-4.7-Flash, and
GPT-OSS-20B. New runs save and replay the actual server token IDs, including
each model's native reasoning-channel format.

The probe indices remain unchanged. GPT-OSS has no layers matching those
indices and requires an explicit `--all-router-layers` override for forward
replay. See [MODEL_SUPPORT.md](MODEL_SUPPORT.md) for the model table, commands,
memory settings, layer mappings, and verification limits.

To tag the already saved GGUF generations without regenerating answers:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh --skip-generate \
  --generation-model qwen3.5-35b-a3b-mtp-ud-q4-k-xl
```

Generation is stored under `results/correlation_pipeline/generation/<model>`.
Annotations, forward checkpoints and analyses are under
`results/correlation_pipeline/reasoning-vllm-v1`. `--results-dir` changes that
common root; `--generation-dir` overrides the generation directory. Use distinct
result roots when comparing generation checkpoints. Server logs are saved as
`slurm_logs/correlation-vllm-<generation|judge>-<timestamp>.log`.

A small isolated run including all four stages:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --datasets math500 --results-dir results/correlation_pipeline/smoke \
  --max-items 1 --samples-per-problem 1 --bootstrap-samples 20
```

`--workers` and `--judge-workers` control concurrent generation and tagging
requests; both default to eight. Sample counts per problem remain unchanged.
vLLM may temporarily queue some requests when GPU cache space is full; eight
workers do not guarantee eight active decodes for long contexts. Keep the tested
`GPU_MEMORY_UTILIZATION=0.90` default on this 32 GB GPU: higher limits caused
out-of-memory failures in the judge pilot.
Each sentence keeps the same question/previous/current/next context. Completed
labels are saved atomically even when requests finish out of order; final output
is restored to source order.

To run without any sentence tagging:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh --skip-tagging
```

This keeps generation, forward replay, whole-reasoning and position views,
correctness/event correlations, metric-pair correlations and expert analyses.
It omits class views (Read, Analyze, Plan, Implement, Explore, Verify, Monitor),
and requires neither a judge server/program nor annotation files. The output
root is `results/correlation_pipeline/reasoning-vllm-untagged-v1`; generations
keep their usual directory. This separates the reduced outputs from tagged
analyses. Adding labels later reuses generations but requires another forward
replay to compute the class features.

`--skip-annotate` instead reuses existing completed annotations and retains
class views. It is useful after tagging has finished. `--skip-generate`,
`--skip-forward`, and `--skip-analyze` skip their respective stages; skipping
forward requires checkpoints containing the selected views. For analysis only,
pass `--skip-generate --skip-annotate --skip-forward` for tagged outputs, or
`--skip-generate --skip-tagging --skip-forward` for untagged outputs.

An already-running tmux command keeps its original options. To switch it to
untagged mode, interrupt that command and relaunch with `--skip-tagging` and the
same generation settings; completed generation shards are reused. Omit
`--skip-generate` while some datasets still need generation.
`--limit N` limits saved traces in tagging/extraction/analysis; `--max-items`
and `--samples-per-problem` affect generation only. `--dry-run` prints the
complete plan without Docker calls or output changes.

For manual server management, the equivalent generation command is:

```bash
docker run --rm --gpus device=0 --ipc=host \
  -v /llms:/llms -e HF_HOME=/llms -p 127.0.0.1:41800:41800 \
  --entrypoint vllm vllm/vllm-openai:v0.29.0 \
  serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --served-model-name Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --host 0.0.0.0 --port 41800 --api-key local-vllm-key \
  --max-model-len 49152 --enable-prefix-caching --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --generation-config vllm --language-model-only
```

Then, in another shell:

```bash
bash src/moe_exp/correlation_pipeline/run_docker.sh generate
```

Each trace records its checkpoint, exact chat messages, sampling settings and
seed. Its manifest retains the matching forward target
`unsloth/Qwen3.5-35B-A3B`; extraction rejects a different target. The new model
directory is `Qwen--Qwen3.5-35B-A3B-GPTQ-Int4`.

See [REASONING_VIEWS.md](REASONING_VIEWS.md) for tests and analysis contracts.
The older llama.cpp launcher remains available to reproduce the historical
performance comparison below.

### Measure generation speed

`--workers` controls concurrent HTTP requests independently of benchmark repeat
counts. vLLM's `--max-num-seqs 8` enables actual concurrent decoding. The previous
llama.cpp pilot used one MTP slot, so its one-versus-two-worker result does not
predict vLLM throughput. See the historical measurements in
[the performance report](../../../report/generation_performance.md).

Compare workers on identical saved prompts and seeds without writing to the
generation dataset. This standard-library diagnostic also runs on host Python:

```bash
PYTHONPATH=src python3 -m moe_exp.correlation_pipeline.profile_generation \
  --generation-dir results/correlation_pipeline/generation/qwen3.5-35b-a3b-mtp-ud-q4-k-xl \
  --output-dir results/generation_performance/new-comparison \
  --base-url http://127.0.0.1:8080/v1 \
  --workers 1 2 --requests 32 --repeats 2 --max-tokens 1024
```

The output directory must be new. Each setting executes all 32 requests, with
the setting order reversed on the second repeat. A short warmup precedes each
trial. `summary.json` records actual server slots, wall throughput and token
limit hits; `requests.jsonl` retains request latencies and response hashes.
These capped outputs measure speed and are not accuracy evaluations. Request
latencies include queueing and must not be summed for wall throughput.

To attribute a completed serial run's actual time to benchmarks, match its
server timing log to the saved generation shards:

```bash
PYTHONPATH=src python3 -m moe_exp.correlation_pipeline.audit_generation_performance \
  --generation-dir results/correlation_pipeline/generation/qwen3.5-35b-a3b-mtp-ud-q4-k-xl \
  --server-log slurm_logs/correlation-mtp-20260903_152848.log \
  --output results/generation_performance/historical.json
```

This audit requires one complete serial log whose completed requests match all
the shards under the supplied directory. It verifies every completion token
count and checks timestamp alignment; it rejects mismatched or overwritten
runs. Partial benchmark shards are reported separately from completed datasets.

### Sentence classes and reasoning positions

The pipeline also supports a frozen GEPA sentence-annotation stage and full,
class-specific, and position-specific reasoning views. See
[REASONING_VIEWS.md](REASONING_VIEWS.md) for the complete rerun command,
unit tests, smoke runs, checkpoint rules, and output interpretation.

## 2. Teacher-forced forward pass

Stop the vLLM server to release VRAM. Forward replay uses the corresponding
Qwen3.5 base model through the existing Unsloth backend:

```bash
src/moe_exp/correlation_pipeline/run_docker.sh forward \
  --quantization unsloth-4bit
```

Hidden states are extracted by default. `--router-only` removes hidden
trajectory and hidden/router geometry features. By default, extraction reads
`results/probeTest/qwen3.5-35b-a3b-gptq-int4/probes/results.json` and retains
only the union of its `best_by_target` layers. Pass `--probe-results` to use
another completed probe run. Probe hidden-state indices without a corresponding
router layer (currently index 40) are reported and excluded. The original layer
numbers are stored in each trace and preserved in analysis column names.

The forward stage now reduces router and hidden tensors on the fly. Normalized
confidence (derived from entropy), top-k confidence measures, switching, overlap, hidden
norm, trajectory distance, and hidden/router geometry are written into each
trace as compact scalar features. Full router and hidden tensors are released
without serialization. Top-k expert IDs remain on disk because expert-identity
and expert-combination analyses need them and they are small. The default output
is:

```text
results/correlation_pipeline/forward/<hf-model>/<dataset>/
  traces_with_routing.jsonl
  tensors/*_experts.pt
  tensors/*_extraction.json
```

Each trace contains a storage audit with tensor shapes, dtypes, element counts,
projected raw bytes, and actual persisted bytes. The model-level `summary.json`
aggregates these values and reports the raw payload avoided. The geometry metric
uses at most 128 uniformly sampled tokens by default; change this contract with
`--max-geometry-tokens`.

Use `--save-raw-tensors` only for a small audit or ablation run. It additionally
writes `*_logits.pt` and `*_hidden.pt` while retaining the same compact
features. The correlation stage never saves normalized expert-weight tensors.
Analysis excludes generations with `finish_reason=length` or completion tokens
at or above `max_tokens` from all trace, expert, and reasoning-view statistics.
Unknown token-limit status is retained. Problem-level avg@n statistics still
require all n attempts, so groups made incomplete by this filter are ineligible.
The output records excluded counts per dataset; the generation-budget audit
continues to cover all input traces. Existing analysis artifacts must be rerun
to apply this policy.

Entropy is retained in the compact audit values, but the analyzer reports its
normalized confidence transform instead of treating both affine-equivalent
quantities as separate correlation evidence.

The GGUF and Hugging Face checkpoints represent the same post-trained model but
use different 4-bit formats, so their numerical activations are not identical.
Quantizing either copy does not change the tokenization/chat-template
requirement; the exact generation messages are replayed with the Hugging Face
tokenizer. Qwen3.5 exposes `Qwen3_5MoeForConditionalGeneration`; Unsloth's
`FastModel` keeps the standard forward API used to request router logits and
hidden states. Extraction calls the decoder backbone directly, so it does not
materialize vocabulary logits or the CausalLM wrapper's auxiliary
load-balancing loss. KV caching is also disabled. Qwen3.5's router API returns
probabilities despite the `router_logits` name; extraction stores equivalent
log-probabilities so downstream softmax-based metrics preserve the native router
distribution exactly.

The Docker launcher sets `UNSLOTH_DISABLE_STATISTICS=1` to avoid an unrelated
Hub telemetry probe and places Unsloth/Torch compile caches under the writable
shared cache. The loader also defaults `UNSLOTH_DISABLE_VENDORED_FLA=1` and
`UNSLOTH_COMPILE_DISABLE=1`: with Torch 2.11, the vendored gated-delta Triton
kernel and the compiled router/hidden-state graph terminate natively, while the
official pure-Torch/eager fallbacks complete within the RTX 5090 memory budget.
These settings can be overridden for a future compatible stack; they do not
disable model downloads or alter weights.

## 3. Analyze correlations

```bash
src/moe_exp/correlation_pipeline/run_docker.sh analyze \
  --bootstrap-samples 500
```

This reads the compact features directly (and remains backward-compatible with
older raw-tensor runs) and writes:

```text
results/correlation_pipeline/analysis/<hf-model>/
  trace_features.csv
  correlations.json
```

Features include per-layer and across-layer summaries of router entropy,
margin, top-1 switching, top-k overlap, hidden-state norm, consecutive hidden
trajectory distance, and hidden/router geometry correlation. Binary outcomes
are final correctness, backtracking, contradiction, and self-correction.

Repeated attempts are not treated as independent. The output contains:

1. exploratory point-biserial trace-level correlations;
2. problem-cluster bootstrap intervals for prespecified across-layer and
  structure features;
3. problem-level Spearman associations between pass rate and feature mean or
   variability; and
4. within-problem correct-minus-incorrect feature contrasts for prompts that
   produce both outcomes.

GPQA-Diamond is excluded from the last contrast because its ten attempts also
permute answer positions; mixing correct and incorrect attempts would confound
reasoning differences with option-position bias. Its trace-level and
problem-level results remain available.

Naive independent-sample p-values and layer-specific feature rankings are
retained only as exploratory diagnostics and are labelled as such. Any later episode classifier can add numeric values under
`TraceRecord.metadata["episode_features"]`; the analyzer automatically includes
them as `episode_*` columns.

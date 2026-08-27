# Correlation pipeline

This pipeline generates reasoning attempts with the OpenAI-compatible
`llama.cpp` server in `src/common/llamacpp`, teacher-forces the saved attempts
through the matching Hugging Face MoE checkpoint, and measures associations
among answer correctness, lexical reasoning events, hidden-state trajectories,
and router behavior.

Generation and forward extraction are separate because they cannot normally
share one GPU: stop the GGUF server before loading the Hugging Face checkpoint.
Every generated attempt and every tensor extraction has an atomic per-trace
checkpoint, so interrupted multi-day runs resume without repeating completed
work.

The default pilot model is `unsloth/Qwen3.5-35B-A3B`. Generation uses
`unsloth/Qwen3.5-35B-A3B-MTP-GGUF` at `UD-Q4_K_XL`; its embedded native-MTP
weights propose tokens and the target layers verify them. Forward extraction
loads the matching Unsloth BF16 Transformers repository through `FastModel`
with runtime 4-bit quantization and `text_only=True`, which skips the unused
vision tower. In the current Unsloth release, `FastLanguageModel` delegates
Qwen3.5 to this same `FastModel` path. The Transformers repository is not
itself a prequantized BNB checkpoint.

## Benchmark protocol

The first eight benchmarks follow the SPIRAL evaluation suite. Generation is
zero-shot with temperature 0.6, top-p 0.95, and at most 8,192 new tokens.

| Name | Source / configuration | Sampling |
|---|---|---:|
| `math500` | `HuggingFaceH4/MATH-500`, test | pass@1 |
| `aime24` | `math-ai/aime24`, test | avg@32 |
| `aime25` | `math-ai/aime25`, test | avg@32 |
| `olympiad` | `Hothan/OlympiadBench`, `OE_TO_maths_en_COMP` | pass@1 |
| `amc23` | `math-ai/amc23`, test | avg@32 |
| `minerva` | `math-ai/minervamath`, test | pass@1 |
| `gpqa_diamond` | public Simple Evals Diamond CSV | avg@10 with permuted choices |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro`, test | pass@1 |

GPQA-Diamond follows SPIRAL's released evaluator: each question is attempted ten
times with a fresh deterministic permutation of its four answer choices. The
attempts remain grouped under the same source question during analysis.

The four existing repository loaders are also included: `gsm8k`, `math`,
`prm800k`, and `processbench`. The last two provide reference process labels in
their original use, but newly generated answers have no aligned gold error step.
ProcessBench generation is therefore retained for representation/routing and
reasoning-event analysis and is explicitly unscored for final-answer accuracy.

`math-verify` performs symbolic math scoring. Multiple-choice datasets require
the requested `\boxed{LETTER}` answer. A normalized exact/numeric fallback is
recorded explicitly if symbolic parsing cannot score an answer.

## 1. Generate with llama.cpp

The host Python on the current machine is 3.10, while the project requires
Python 3.11. Use the Docker launcher below, following the same pattern as
`probeTest/run_slurm.sh`; do not run `pip install -e .` on the host.

Build or refresh the project image once from the repository root:

```bash
docker build -t moe-mfa-experiments:latest .
```

### One-command recommended pilot

The orchestrator starts native MTP, waits for `/health`, generates the tutor's
recommended MATH500/AIME24/Minerva pilot, stops llama.cpp to release VRAM, and
then runs forward extraction and analysis:

```bash
src/moe_exp/correlation_pipeline/run_all.sh
```

It prints timestamped stage updates and writes the llama.cpp server log to
`slurm_logs/correlation-mtp-<timestamp>.log`. All stages are resumable. A
single `/llms` model folder is shared by the server and Python stages. Missing
GGUF weights are downloaded there with resumable `.part` files, so they are
fetched only once. The Hugging Face cache under the same directory stores the
BF16 source shards used by Unsloth's runtime quantizer. A small end-to-end smoke
run is:

```bash
src/moe_exp/correlation_pipeline/run_all.sh \
  --max-items 1 \
  --samples-per-problem 1 \
  --bootstrap-samples 20
```

Use `--skip-generate`, `--skip-forward`, or `--skip-analyze` to reuse completed
stages explicitly. The individual commands below remain available for manual
control.

Start the recommended Qwen3.5 target with its embedded MTP weights:

```bash
src/common/llamacpp/serve_qwen3_5_35b_a3b_mtp.sh
```

In another shell, run all twelve datasets:

```bash
src/moe_exp/correlation_pipeline/run_docker.sh generate \
  --workers 1
```

For a smoke test that does not run `avg@32`:

```bash
src/moe_exp/correlation_pipeline/run_docker.sh generate \
  --datasets math500 aime24 gpqa_diamond gsm8k \
  --max-items 2 \
  --samples-per-problem 1
```

The default output is:

```text
results/correlation_pipeline/generation/<served-model>/<dataset>/
  generation_shards/*.json
  traces.jsonl
  manifest.json
```

Each trace stores the exact chat messages and sampling seed sent to llama.cpp.
Its manifest also records the target `unsloth/Qwen3.5-35B-A3B` and integrated
MTP artifact `unsloth/Qwen3.5-35B-A3B-MTP-GGUF:UD-Q4_K_XL`. Forward extraction
refuses to run if its Hugging Face target differs from the generation manifest.
The default served-model name is `qwen3.5-35b-a3b-mtp-ud-q4-k-xl`. Do not rename
`--model` between resumed invocations; it is part of the output path and
provenance. The plain `serve_llamacpp.sh` remains available as a no-speculation
baseline.

## 2. Teacher-forced forward pass

Stop the llama.cpp server to release VRAM. Then supply the Hugging Face
checkpoint corresponding to the GGUF used for generation:

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

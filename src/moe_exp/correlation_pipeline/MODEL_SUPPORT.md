# Correlation model support

Select the model with `run_all.sh --model HF_CHECKPOINT`. The selected checkpoint
is used for generation, the generation manifest's forward target, forward
replay, and analysis. The tagging checkpoint and frozen GEPA program retain
their existing defaults.

Omitting `--model` still uses `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` for generation
and `unsloth/Qwen3.5-35B-A3B` with Unsloth 4-bit loading for forward replay.

## Profiles

| Checkpoint | vLLM reasoning parser | Generation MTP | Forward quantization default |
| --- | --- | --- | --- |
| [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | `qwen3` | Yes | `unsloth-4bit` |
| [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `qwen3` | Yes | `unsloth-4bit` |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `qwen3` | No | `none` |
| [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | `nemotron_v3` | Yes | `none` |
| [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | `gemma4` | No | `bnb-4bit` |
| [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | `glm45` | Yes | `none` |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | `openai_gptoss` | No | `mxfp4-bf16` (dequantized MXFP4, CPU offload) |

The server image remains `vllm/vllm-openai:v0.29.0`. Its
[reasoning parser registry](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/reasoning/__init__.py)
and [speculative configuration](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/speculative.py)
provide these parser and MTP implementations. MTP uses three speculative tokens
by default. `--generation-speculation none` disables it for generation, including
when a quantized checkpoint does not retain the MTP head. The judge's MTP settings
are independent and unchanged.

Qwen3-30B-A3B's default context is capped at its native 40,960 tokens. Other
profiles retain the 49,152-token generation context. All retain the existing
32,768-token completion budget and benchmark sampling settings. An explicit
`--ctx-size` overrides the default; any larger context must be supported by the
checkpoint/server configuration.

Gemma generation explicitly requests `chat_template_kwargs.enable_thinking=true`.
The other models use their native thinking defaults; GPT-OSS defaults to medium
reasoning effort. Prompt token IDs retain the resulting exact chat template.
Only the multimodal Qwen3.5/3.6 and Gemma profiles request text-only server loading.

## Run and memory settings

### Nemotron: NVFP4 generation and NF4 forward replay

Use the official NVFP4 checkpoint for vLLM generation and the BF16 checkpoint
with `--quantization bnb-4bit` for replay:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --generation-model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --generation-speculation none \
  --quantization bnb-4bit \
  --results-dir results/correlation_pipeline/nemotron-nvfp4-nf4
```

Do not pass `--generation-quantization bitsandbytes`: the pinned vLLM 0.29.0
image rejects it. vLLM reads the NVFP4 checkpoint's native ModelOpt configuration.
The forward loader quantizes the BF16 checkpoint to bitsandbytes NF4 with double
quantization. It converts Nemotron's non-gated experts into individual linear
layers before loading weights, avoiding Transformers' unquantized fused expert
tensors. Mamba mixers, routers, and the output head retain native floating-point
weights. Native routing, correction biases, and the existing capture hooks are
preserved. Loading fails if checkpoint tensors are missing or any expert is
not materialized in 4-bit on CUDA.

The generation and replay quantizers differ, as in the Qwen GPTQ/Unsloth setup.
Captured routes describe the **NF4 replay model**, not necessarily the routes
selected by NVFP4 generation. This path does not use Unsloth's gated-expert
implementation and does not load ModelOpt weights into Transformers.

Rebuild the project image after updating dependencies. `kernels==0.11.7` is pinned
because newer releases reject the hub-kernel registrations in Transformers 5.5.
For the locally prepared compatibility image, prefix the command above with
`IMAGE_NAME=moe-mfa-experiments:nemotron-forward`.

Validation: a tiny real hybrid Nemotron checkpoint preserves native logits and
router capture before quantization, and loads every expert in NF4 and completes
router/hidden-state extraction on an RTX 5090. Full 30B loading and long-context
memory fit are not yet validated. The BF16 source download requires approximately
61.3 GiB in the model cache, in addition to the NVFP4 generation checkpoint.
For a first full-model smoke test, add `--datasets math500 --max-items 1
--samples-per-problem 1 --limit 1 --workers 1 --skip-tagging`; this includes
forward replay and analysis while omitting the judge.

### Quantized Gemma checkpoints

Gemma 4 MoE now has a text-only NF4 forward adapter. Use the unquantized Google
checkpoint as the forward source and NVIDIA's NVFP4 checkpoint for generation:

```bash
IMAGE_NAME=moe-mfa-experiments:quantized-forward \
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --model google/gemma-4-26B-A4B-it \
  --generation-model nvidia/Gemma-4-26B-A4B-NVFP4 \
  --quantization bnb-4bit \
  --results-dir results/correlation_pipeline/gemma-nvfp4-nf4
```

The local `quantized-forward` image is an alias of the compatible
`nemotron-forward` image; both use the source mounted by `run_docker.sh`.
A rebuilt project image also includes the pinned dependency fix.

The adapter splits each fused `gate_up_proj`/`down_proj` tensor during checkpoint
loading and quantizes **every** expert independently with double-quantized NF4.
Routers, normalization parameters, and tied input/output embeddings remain in
floating point. Vision/audio towers are omitted from text replay. Checkpoint
mismatches and unquantized experts fail explicitly; exporting and reloading this
adapter as a new quantized checkpoint is not implemented.

Tests use tiny real Gemma models to check native-logit equivalence before
quantization, each expert's quantized weights, text and multimodal checkpoint
loading, tied embeddings, scaled routing weights, and CUDA router/hidden-state
capture. The full 26B NF4 model also completed all 38 gold-corpus forwards
on an RTX 5090 during the probe experiment. This verifies memory fit for those
traces; longer correlation traces can still require more memory.
Routes describe the NF4 replay model and may differ from NVFP4 generation.

`nvidia/Gemma-4-26B-A4B-NVFP4` selects the Gemma generation profile
(`gemma4` reasoning parser, thinking enabled, no MTP, text-only loading).
NVIDIA documents this checkpoint for vLLM on Blackwell. Its packed weights use
`quant_method: modelopt`, which the current project container's Transformers
5.5.0 forward loader does not support. Selecting this checkpoint with `--model`
therefore does **not** provide a working end-to-end run. Launcher tests only
validate command construction, not model loading or memory fit.

Google's `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` is an official 4-bit QAT
checkpoint for llama.cpp. The unified launcher does not currently implement a
llama.cpp generation backend. Combining this generator with NVIDIA NVFP4 replay
would still require a compatible forward backend; it does not resolve the
Transformers loading limitation. Do not substitute the Google
`qat-q4_0-unquantized` checkpoint expecting packed 4-bit memory usage.

Sources: [NVIDIA model card](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4),
[NVIDIA checkpoint configuration](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/blob/main/config.json),
[Google GGUF checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf).

For example, with enough generation memory for the selected checkpoint:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --model Qwen/Qwen3.6-35B-A3B \
  --results-dir results/correlation_pipeline/qwen36
```

The BF16 checkpoints in the table do not fit entirely on a 32 GB GPU. The flag
does not silently replace them with quantized repositories. Options are:

- `--generation-model ORG/MATCHING-QUANTIZED-CHECKPOINT`: serve a quantized
  checkpoint of the same model while keeping `--model` as the forward target.
  The generation option has precedence regardless of option order. Recognized
  checkpoint-name prefixes retain their family profile across quantized variants.
- `--generation-quantization MODE`: request a quantization mode supported by
  vLLM and that checkpoint; this does not change the forward loader.
- `--cpu-offload-gb N`: offload generation weights to host RAM per GPU. This
  requires enough RAM and can significantly reduce throughput.
- `CUDA_VISIBLE_DEVICES=0,1 ... --tensor-parallel-size 2`: expose multiple GPUs
  and use tensor parallelism for generation. The tagging server retains its
  single-device model configuration; forward loading uses Accelerate's automatic
  device placement.
- `--quantization MODE`: override forward loading with `none`, `bnb-4bit`,
  `bnb-8bit`, or `unsloth-4bit`, subject to that backend's architecture support.
  `none` preserves a checkpoint's native quantization, including GPT-OSS MXFP4.
  Bitsandbytes alone may leave fused expert tensors unquantized; do not assume
  every MoE checkpoint fits a 32 GB GPU with this option.

Use a different `--results-dir` for each checkpoint or quantization comparison.
Add `--dry-run` to inspect commands without Docker calls, downloads, or output
writes. A dry run validates argument construction, not GPU memory or model loading.

## Probe layers

The launcher and direct forward command select gold probe results by forward
model. Missing results stop a real launcher run before a server starts.

| Forward model | Probe results under `results/probeTest/` |
| --- | --- |
| GPT-OSS-20B, MXFP4 expanded to BF16 | `gpt-oss-20b/probes/results.json` |
| Gemma4-26B, NF4 replay | `gemma-4-26b-a4b-it-nf4/probes/results.json` |
| Qwen and the remaining profiles | `qwen3.5-35b-a3b-gptq-int4/probes/results.json` |

Each model's selection is the union of the seven `best_by_target.layer_idx`
values, selected by test accuracy with ties resolved toward the earlier index.
The probe uses the input of the decoder at that index (index 0 is the embedding
output). Only actual router layers are retained. The final normalized output
has no next router and is excluded; indices are never shifted or renumbered.
The forward summary records retained and excluded indices. Gemma's completed
2026-09-15 run selects `16, 19, 21, 22, 25, 27, 29`; OSS selects
`15, 19, 21, 23`. All are valid router indices. Per-target metrics and
convergence diagnostics are summarized in the [probe README](../probeTest/README.md#completed-runs-2026-09-15).

GPT-OSS defaults to `mxfp4-bf16`: the original MXFP4 checkpoint values are
expanded to BF16, with eager expert execution and CPU offload. This avoids
the native Triton compiler crashes observed on the RTX 5090. Generation still
uses vLLM's native MXFP4 checkpoint; the replay precision is recorded in the
probe and forward metadata. The native forward mode remains available through
`--quantization none` on compatible runtimes.

The Google Gemma checkpoint defaults to the same `bnb-4bit` forward adapter
used by its probes. For NVFP4 generation, keep Google Gemma as `--model` and
select the NVIDIA checkpoint with `--generation-model`.

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --model openai/gpt-oss-20b \
  --results-dir results/correlation_pipeline/gpt-oss-20b

IMAGE_NAME=moe-mfa-experiments:quantized-forward \
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --model google/gemma-4-26B-A4B-it \
  --generation-model nvidia/Gemma-4-26B-A4B-NVFP4 \
  --results-dir results/correlation_pipeline/gemma-nvfp4-nf4
```

`--probe-results PATH` explicitly overrides the model-specific file.
`--all-router-layers` bypasses probe selection. The remaining profiles still
reuse the historical Qwen selection; those are not independently measured
best layers for those families.

## Replay and routing contracts

The launcher adds `--save-token-ids` to generation. vLLM must return the actual
prompt and completion token IDs. The pipeline saves those IDs, a tokenizer
vocabulary fingerprint, and character offsets into the decoded continuation.
Forward replay uses the saved IDs directly. It does not reconstruct Harmony,
Gemma channels, or `<think>` wrappers from parsed response fields. This also
avoids duplicating a thinking prefix already supplied by a Qwen chat template.

Sentence labels and reasoning views use offsets aligned to those same tokens,
including Unicode characters spread across several tokens. Missing token IDs,
a mismatched forward tokenizer, or inconsistent saved completion text cause a
clear error. Final-answer scoring still uses the parsed final content. The
generation fingerprint includes the replay version and explicit template
options, so older reconstructed-text shards are not reused as exact-token runs.
Legacy generations remain readable through the legacy replay path.

Streaming router hooks capture selected layers' decoder inputs and native
expert selections. They avoid retaining every layer's full sequence on the GPU.
GLM and Nemotron use sigmoid affinities and selection correction biases, so raw
logit top-k would produce incorrect expert identities. Their full affinities are
normalized for entropy/geometry, while their actual selected experts are retained.
Qwen3.5/3.6 and Gemma expose probabilities, which are converted to log-probabilities
for the common downstream interface. Qwen3 and GPT-OSS expose logits.

Feature schema 3 computes selected mass and boundary gaps using the actual
selected set, including when bias-adjusted selection differs from raw affinity
ranking. Such boundary gaps can be negative. Checkpoints include the native
router-capture version and exact replay data in their invalidation contract.

## Verification

The project requires Transformers 5.5 or later within major version 5,
Tokenizers 0.22.2 or later, and the `kernels` dependency for native MXFP4 loading.
Rebuild the project image when using an older environment:

```bash
docker build -t moe-mfa-experiments:latest .
```

Tests cover all seven launcher profiles, native chat-marker/Unicode alignment,
exact-token generation and resume, fixed-layer rejection, and real tiny
Transformers architectures. The Qwen3.5 tests also cover Qwen3.6's shared
architecture. Actual tokenizers from all seven repositories were checked offline.
Full pretrained checkpoints have not been run end-to-end on the GPU; memory,
quantization kernels, and throughput must be verified with a small run in the
intended hardware configuration.

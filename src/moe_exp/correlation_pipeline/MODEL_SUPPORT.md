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
| [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | `gemma4` | No | `none` |
| [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | `glm45` | Yes | `none` |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | `openai_gptoss` | No | `none` (native MXFP4) |

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

### Nemotron on LEONARDO Booster (A100, native Slurm)

`sbatch/native_nemotron_generate.sbatch` serves
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` from the project
`vllm-cu129` environment on two A100s and runs the resumable generation stage
with the dedicated `envs/correlation-client-3.11` client. Checkpoints are read
from `cache/hf` with `HF_HUB_OFFLINE=1`, so they must be downloaded on a login
node first.

A100 (SM80) needs the settings that `sbatch/native_nemotron_server.sh` pins:

- `--attention-config '{"backend":"TRITON_ATTN"}'`: the FlashInfer paged-prefill
  kernel image is invalid on this driver stack (job 57729862).
- `--kv-cache-dtype bfloat16`: a native FP8 KV cache needs SM89+.
- `--linear-backend marlin`: the ModelOpt mixed-precision layout routes both the
  W4A16 NVFP4 expert layers and the FP8 mixer projections through Marlin.
- `--mamba-backend triton --mamba-cache-mode align`: the verified setting on
  this node. NVIDIA's Ampere recipe uses FlashInfer, but FlashInfer's Mamba SSU
  JIT hits a gcc internal compiler error while building
  `invoke_selective_state_update_mtp.cuh`, so that path cannot start a server.
  FlashInfer remains selectable with `--mamba-backend flashinfer`.

The checkpoint declares ModelOpt `MIXED_PRECISION` (5,935 W4A16_NVFP4 expert
layers, 46 FP8 mixer projections), so no `--quantization` flag is passed:
vLLM 0.29 detects the layout and selects the per-layer kernels itself.

That combination is the verified default of both launchers, not just an example:
tensor parallel 2, 8 concurrent sequences, 8,192 batched tokens, 0.90 GPU memory
utilization, DSpark with 3 speculative tokens, Triton Mamba, Marlin linears,
Triton attention and bfloat16 KV. `--plan-only` on either launcher prints the
resulting command without touching a GPU.

Measured on 2026-09-18 (debug QoS, 2x A100-SXM-64GB, checkpoint revisions above):

| Probe | Configuration | Result |
| --- | --- | --- |
| 58117845 | DSpark, 2 requests | acceptance length 3.45, draft acceptance 81.7% |
| 58121810 | DSpark, fresh AIME traces | acceptance length 2.92, draft acceptance 64.1% |
| 58122776 | DSpark, 8 concurrent, 12-request batch | 826 accepted tokens/s in the vLLM metric window; 13,321 tokens in 30 s end to end (444 tokens/s including prefill and trace writes) |

The 58122776 verdict is kept at
`results/correlation_pipeline/nemotron-nvfp4-throughput-dspark-8w/probe/job-58122776.json`.
At ~800 tokens/s sustained, the six-benchmark suite (4,647 traces) projects to
roughly 6-10 h, so it fits one 24 h `boost_usr_prod` job with resumption room.

Generation uses the checkpoint's recommended sampling, temperature 1.0 and
top-p 0.95, a 49,152-token context and a 32,768-token completion budget, and
writes token-replay traces for exact forward replay later. The default run
covers the six SPIRAL MATH benchmarks with their per-dataset attempt counts
(`math500`, `olympiad`, `minerva` pass@1; `aime24`, `aime25`, `amc23` avg@32:
500/30/30/675/40/272 problems, 4,647 traces) under
`results/correlation_pipeline/nemotron-nvfp4-<dspark|plain>/generation`.

```bash
# 30-minute debug-QoS feasibility probe (DSpark on A100 is unvalidated upstream)
sbatch sbatch/native_nemotron_probe.sbatch --speculation dspark

# full six-benchmark run, only after the probe passed
sbatch sbatch/native_nemotron_generate.sbatch --speculation dspark
```

The suite can also run as independent per-dataset shards that share one
generation root and finish in parallel. Each shard keeps its own provenance,
logs and compile cache under `<results>/parallel/<label>/`, and verifies only
its own per-dataset manifest, so the shards cannot overwrite each other:

```bash
for ds in math500 aime24 aime25 olympiad amc23 minerva; do
  sbatch sbatch/native_nemotron_generate.sbatch --job-label "$ds" --datasets "$ds"
done
```

A labelled shard writes a root `summary.json` that lists only its dataset, so
regenerate the merged root summary from the per-dataset manifests once every
shard has finished before downstream labeling or forward replay reads the root.
The launcher's job IDs and settings for the 2026-09-18 run are recorded in
`results/correlation_pipeline/nemotron-nvfp4-dspark/parallel/launch_record.json`.

DSpark is validated by NVIDIA on Hopper/Blackwell only, but the CINECA probes
(jobs 58117845, 58118129, 58121810 on `lrdn0250`/`lrdn1554`/`lrdn2026`) show it
working on the A100s with the Triton Mamba backend: the server loads the
`Qwen3DSparkModel` draft, captures its CUDA graphs, and reports a mean
acceptance length of 2.9-3.5 with 64-82% draft acceptance. Re-run the probe
before trusting another node, driver or vLLM build; if it fails, fall back to
`--speculation none`.
`generation/server_manifest.json` records the run contract (speculation mode,
model revisions, backends, argv, versions, repository SHA) and the launcher
refuses to resume a directory whose contract differs; resubmitting the same
command after a walltime stop continues from the per-trace shards. Labeling and
forward replay are unchanged and can consume the new trace root.

### GLM-4.7-Flash and Qwen3-30B-A3B on LEONARDO Booster (A100, native Slurm)

`sbatch/native_source_generate.sbatch --source glm|qwen330b` serves these two
BF16 checkpoints from the project `vllm-cu129` environment on two A100s, with the
argv built by the pure-stdout helper `sbatch/native_source_server.sh`. Neither
checkpoint is quantized, so no `--quantization`, `--linear-backend` or
`--mamba-backend` flag is passed, and neither is multimodal, so
`--language-model-only` must not be passed either.

| | GLM-4.7-Flash | Qwen3-30B-A3B |
| --- | --- | --- |
| architecture | `Glm4MoeLiteForCausalLM` | `Qwen3MoeForCausalLM` |
| attention | **`TRITON_MLA`** | `FLASH_ATTN` |
| reasoning parser | `glm45` | `qwen3` |
| `--max-model-len` | 49152 | **40960** (native cap) |
| MTP head | present, opt in with `--speculation mtp` | none, `mtp` is refused |
| sampling | temperature 1.0, top-p 0.95, top-k 0 | temperature 0.6, top-p 0.95, top-k 20 |
| size / revision | 62.5 GB, `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | 61.1 GB, `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |

Two settings are load-bearing, not stylistic:

- vLLM 0.29 lists `glm4_moe_lite` in `is_deepseek_mla()`, so GLM-4.7-Flash runs
  the MLA attention kernels. `FLASH_ATTN` is not a valid backend on that path,
  and of the MLA backends only `TRITON_MLA` supports SM80. Passing an empty
  `--attention-backend` omits the flag and lets vLLM select, which is the probe
  fallback.
- `glm45` is an alias, not a legacy parser: `vllm/reasoning/__init__.py` maps
  both `glm45` and `glm47` to `glm47_moe_reasoning_parser`, so GLM-4.7-Flash is
  served by its own parser.

Checkpoints are read from `cache/hf` with `HF_HUB_OFFLINE=1`, so they must be
downloaded on a login node first with `download_model.sh <repo-id> ...`, which
verifies every shard named in `model.safetensors.index.json`.

Probe before committing a full run: `sbatch/native_source_probe.sbatch --source
<key>` is a 30-minute `boost_qos_dbg` job whose `SMOKE_OK` verdict additionally
requires a positive, non-constant reasoning-length metric on every trace,
resolved with the sampler's own `sample_stratified.reasoning_tokens`.
`boost_qos_dbg` permits at most two running or pending jobs per user.

### One server port per concurrent job, chosen deterministically

Leonardo Booster nodes hold four A100s, so Slurm can place two 2-GPU jobs on the
same node. Every launcher that binds a fixed port therefore risks a collision:
probe 58282929 shared port 41800 with a concurrent GLM probe, passed `/health`
against the sibling's server, and had every request rejected because the served
model id did not match. Nothing in the failure named the port; it surfaced only
as `Inference request failed after 5 attempts`.

`native_source_probe.sbatch` and `native_source_generate.sbatch` therefore derive
the port from the source and, for generation, the dataset shard: probes use 41810
(glm) and 41820 (qwen330b); generation uses 41900-41905 and 41920-41925. The
mapping must stay deterministic rather than random, because `generate.py` folds
`--base-url` into `generation_sha256`, so a shard only resumes from its own cached
traces when it is handed the same port again. The older Nemotron and Qwen3.6
generation launchers still share port 41800 across their six shards and are
exposed to this if two of their shards ever land on one node.

### Server readiness for unquantized BF16 checkpoints

Qwen3-30B-A3B spends ~350 s loading its 16 BF16 shards and a further ~130 s on
profiling, KV-cache creation and CUDA-graph capture, so it answers `/health`
about 750 s after launch. The Nemotron probe's 720 s readiness cap cut that off
(job 58283627) even though the engine had come up correctly, so the source probe
defaults to 1200 s; the `boost_qos_dbg` wall of 1800 s still leaves room for the
four-problem slice. GLM-4.7-Flash spreads the same weight volume over 48 smaller
shards and loads comfortably faster.

### Reasoning-token reporting is not uniform across model families

vLLM does not populate `usage.completion_tokens_details.reasoning_tokens` for
every family. The gpt-oss harmony path reports a constant `0` while its
`openai_gptoss` parser still fills `reasoning_content` correctly. Gemma,
Qwen3.6 and Nemotron report real values. Because the stratified sampler uses
that field as its length stratum, the first gpt-oss sample was stratified into a
single length bucket. `sample_stratified.py` now derives the count from the
saved `token_replay` when the server value is absent or zero, and refuses a
dataset whose eligible traces all report the same length. Any new model should
be checked with the probe rather than assumed to report the field.

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
capture. Full 26B loading and long-context memory fit remain unverified.
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

The default probe file and selected indices remain unchanged:
`results/probeTest/qwen3.5-35b-a3b-gptq-int4/probes/results.json`, currently
`27, 34, 35, 36, 38, 39, 40`.

Only existing MoE decoder layers are extracted. Excluded indices are reported;
indices are never renumbered to match a compressed list of routers.

| Model | Current probe indices that have routers |
| --- | --- |
| Qwen3.5 / Qwen3.6 | 27, 34, 35, 36, 38, 39 |
| Qwen3-30B | 27, 34, 35, 36, 38, 39, 40 |
| Nemotron 3.5 Lightning | 27, 34, 36, 38, 40 |
| Gemma4-26B | 27 |
| GLM-4.7-Flash | 27, 34, 35, 36, 38, 39, 40 |
| GPT-OSS-20B | None: it has layers 0–23 |

A real GPT-OSS run with the fixed selection stops before starting a server.
To explicitly use all its router layers, add `--all-router-layers`:

```bash
bash src/moe_exp/correlation_pipeline/run_all.sh \
  --model openai/gpt-oss-20b \
  --all-router-layers \
  --results-dir results/correlation_pipeline/gpt-oss-20b \
  --datasets math500 --max-items 1 --samples-per-problem 1 \
  --bootstrap-samples 20
```

This override is opt-in; it does not modify the probe file or the defaults for
other runs. Omit `--datasets`, `--max-items`, `--samples-per-problem`, and
`--bootstrap-samples` to run the usual suite with its default sampling settings.
The selected layer policy and actual layer indices are recorded in the forward
summary. Reusing Qwen-selected indices on other families is an explicit experiment
choice; it does not imply those are their best probe layers.

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

# MoE guiding

An exploratory intervention on **live top-2 expert routing**. For each token
at each selected transformer layer, compute probabilities over all experts:

```text
p = softmax(router_logits.float())
margin = 2 * (p[top1] + p[top2]) - 1
intervene = margin > threshold          # default: 0.4, strictly greater
```

At the default threshold, intervention requires the original top two experts
to hold more than 70% of the probability mass. Keep top-1 and replace top-2
with the least probable expert. Otherwise, retain ordinary top-2 routing.
This is a hypothetical experimental rule, with no assumed accuracy benefit.

## Conditions

| `--condition` | Expert selection when triggered | Weights before renormalization |
|---|---|---|
| `baseline` | Native Mixtral top-2; no custom callback installed | Native top-2 probabilities |
| `selected` | Top-1 + least-1 | Those selected experts' own probabilities |
| `transfer` | Top-1 + least-1 | Original top-1 and top-2 probabilities |

`transfer` gives the displaced second expert's coefficient to the least
probable expert. `selected` may give that expert a tiny or even numerically
zero contribution. These are separate experimental conditions. Mixtral
renormalizes the two weights; the standalone callback also supports
`renormalize=False`.

For `[0.50, 0.25, 0.15, 0.10]`, the margin is `0.50`: the intervention selects
`[0, 3]`. Renormalized weights are `[5/6, 1/6]` for `selected` and `[2/3, 1/3]`
for `transfer`. For `[0.40, 0.25, 0.20, 0.15]`, the margin is `0.30`: both
retain `[0, 1]`. Tied minima exclude the retained top-1 expert so IDs remain
distinct; top-k tie ordering follows PyTorch.

## Check the arithmetic

With PyTorch and the project installed, no vLLM or model download is needed:

```bash
python -m moe_exp.moe_guiding.run sanity
python -m moe_exp.moe_guiding.run sanity --device cuda
python -m pytest tests/test_moe_guiding.py -q -p no:cacheprovider
```

`routing.margin_route` has the four-argument vLLM callback signature from the
proposal and defaults to `selected` at threshold `0.4`. `MarginRouter` accepts
a `RoutingConfig` for the configurable conditions and records compact counters.
Both return contiguous float32 weights and int32 expert IDs of shape
`[num_tokens, 2]`. Logits are assumed finite.

## vLLM setup

The initial adapter supports **unquantized FP16/BF16 Mixtral checkpoints**,
with native top-2 routing and at least three experts, on supported CUDA
hardware. Provide a checkpoint that fits the available GPUs; tensor parallelism
is configurable. This adapter does not implement the repository's Qwen
experiments or force a native top-k model down to top-2.

Install in a separate Linux/CUDA vLLM environment to let vLLM select its
compatible PyTorch build. From the repository root:

```bash
pip install -e ".[moe-guiding]"
```

Install the project in **every worker environment**. This registers
`moe_guiding` in the `vllm.general_plugins` entry-point group. The entry point
lazily registers `MoEGuidingMixtralForCausalLM`; standard Mixtral and other
model architectures retain their normal implementation. If `VLLM_PLUGINS`
is set, its allowlist must include `moe_guiding`.

The adapter injects `custom_routing_function` into Mixtral's
`FusedMoEFactory` call while constructing the model in each worker. It restores
the module binding after construction, including on failure. The older
`FusedMoE` constructor name is accepted if its signature supports the keyword,
but the rest of the runner requires the current `moe_backend`,
`additional_config`, and `LLM.collective_rpc` APIs. An older constructor name
alone does not establish compatibility with an old vLLM release.

The implementation was checked against upstream APIs; end-to-end vLLM model
generation has **not** been validated in this workspace. There is no certified
vLLM version pin yet. Record and pin the version after a successful GPU smoke
run; each manifest records the installed versions automatically.

## Generate a comparison

Input is JSONL with a unique string `id` and nonempty `prompt` per row. Additional
fields, such as gold answers, are preserved. Two smoke prompts are supplied in
`prompts.jsonl`; they are not an accuracy benchmark.

Run each condition in a fresh process, keeping checkpoint revision, prompts,
layer selection and sampling settings fixed:

```bash
MODEL=/path/to/unquantized-mixtral-checkpoint
for condition in baseline selected transfer; do
  python -m moe_exp.moe_guiding.run generate \
    --model "$MODEL" \
    --prompts src/moe_exp/moe_guiding/prompts.jsonl \
    --condition "$condition" \
    --threshold 0.4 \
    --layers 12 \
    --chat \
    --max-tokens 512 \
    --seed 42 \
    --output-dir "results/moe_guiding/layer12/$condition"
done
```

Use `--revision COMMIT_SHA` for a remote checkpoint, `--layers all` for every
MoE layer, or `--layers 0,12` for a subset. Indices are zero-based and validated
against the model. Use `--chat` only with a checkpoint that has a chat template;
otherwise the strings are passed as raw completion prompts. The rendered
prompts and token IDs are saved for comparison. Greedy decoding is the default
(`temperature=0`, `top_p=1`); matching seeds do not guarantee identical outputs
across different routing conditions.

The runner fixes the expert backend to Triton and enables eager execution.
It disables prefix caching so prompt computations participate in each run.
Quantization, expert parallelism/load balancing, pipeline/data parallelism,
dual batch overlap and speculative decoding are rejected for this initial
adapter. Graph/compilation experiments require a separate implementation
without these eager instrumentation constraints.

Outputs per condition:

```text
manifest.json       settings, prompt digest, versions, status, worker routing counters
generations.jsonl   original records, rendered prompts, outputs, token IDs, finish reasons
```

After warmup, counters are reset through an RPC in every worker. Following
generation, every selected layer must report routed token evaluations; a
missing/bypassed callback makes the run fail. An intervention count of zero is
valid when no evaluated margin exceeds the threshold. The native baseline has
no custom counters. Counters combine prefill and decode and count token-layer
evaluations supplied by the engine, including any padding or recomputation;
they are not unique generated-token counts. Tensor-parallel workers may report
the same routing decisions: **do not sum their counts as independent tokens**.

An existing manifest prevents overwriting or mixing runs. For an interrupted
or failed run, use a new output directory. Automatic resume and accuracy
scoring are outside this initial routing experiment.

## Selected model for an RTX 3070 (8 GB)

Use [`Isotonic/TinyMixtral-4x248M-MoE`](https://huggingface.co/Isotonic/TinyMixtral-4x248M-MoE)
for the initial routing smoke experiment. Its published configuration uses the
same `MixtralForCausalLM` architecture supported by this adapter: four experts,
native top-2 selection, and 12 transformer layers. No routing-rule change is
needed. The launcher pins revision `1e3516176a6279ce93923060fcad848dc043af79`.

The repository contains 701,053,952 parameters in FP32. At runtime, `--dtype
float16` converts the weights to approximately **1.40 GB / 1.31 GiB**, without
quantization. The download is approximately 2.8 GB. With a 2,048-token total
context and one concurrent sequence, this should leave sufficient room for
KV cache, CUDA and vLLM overhead on an otherwise available 8 GB GPU. This is a
size-based estimate, not a measured end-to-end vLLM run on a 3070. The configured
80% GPU budget includes the weights and KV cache; it is not a claim that the
process will use only 1.4 GB.

The RTX launcher uses Docker, following the correlation pipeline's generation
setup. It defaults to the same `vllm/vllm-openai:v0.29.0` base image and builds
`moe-guiding:vllm-0.29.0` on first use to install the project and worker plugin.
Host Python is not used. Docker GPU access is required; the first build and
checkpoint download require network access. Run from the repository root:

```bash
bash src/moe_exp/moe_guiding/run_rtx3070.sh selected
```

Compare all conditions in separate processes:

```bash
for condition in baseline selected transfer; do
  bash src/moe_exp/moe_guiding/run_rtx3070.sh "$condition"
done
```

Results go to `results/moe_guiding/rtx3070/<condition>`. The launcher selects
**layer 6** because this model has layers **0 through 11**; the general runner's
default layer 12 is invalid for it. It uses FP16, one GPU, a batch concurrency
limit of 1, at most 2,048 tokens per engine step, and 256 generated tokens.
The 2,048-token context limit includes both prompt and generated tokens.

Pass additional runner options after the condition, for example:

```bash
bash src/moe_exp/moe_guiding/run_rtx3070.sh transfer \
  --layers all \
  --output-dir results/moe_guiding/rtx3070/all_layers/transfer
```

Use a new output directory for each rerun. `HF_CACHE_DIR` defaults to `/llms`,
`CUDA_VISIBLE_DEVICES` defaults to `0`, and `PHYS_DIR` defaults to the repository
root. The workspace and model cache are mounted into the container, with the
same rootless Docker user mapping as the correlation stage launcher. Override
`IMAGE_NAME` to select a prepared image, or `VLLM_IMAGE` to change the base used
when building a missing image. Rebuild after changing package metadata:

```bash
docker build -f src/moe_exp/moe_guiding/Dockerfile -t moe-guiding:vllm-0.29.0 .
```

Preview without Docker calls or filesystem changes:

```bash
DRY_RUN=true bash src/moe_exp/moe_guiding/run_rtx3070.sh selected
```

Run CPU arithmetic checks in the same image with
`bash src/moe_exp/moe_guiding/run_docker.sh sanity`. Keep the pinned model
revision with this checkpoint when comparing conditions.

This checkpoint is a community merge of small instruction-tuned models, not an
official Mistral release. Its model card provides no reasoning benchmark
evidence. Use it to test callback execution, expert replacement, and the
generation pipeline; answer-quality conclusions need a separately validated
model and evaluation protocol.

Verified metadata:
[model configuration](https://huggingface.co/Isotonic/TinyMixtral-4x248M-MoE/blob/1e3516176a6279ce93923060fcad848dc043af79/config.json),
[parameter metadata](https://huggingface.co/api/models/Isotonic/TinyMixtral-4x248M-MoE),
and [RTX 3070 memory specification](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3070/).

## Serve the same intervention

After installing the project in each worker environment:

```bash
vllm serve "$MODEL" \
  --hf-overrides '{"architectures":["MoEGuidingMixtralForCausalLM"]}' \
  --additional-config '{"moe_guiding":{"condition":"selected","threshold":0.4,"layers":[12]}}' \
  --dtype bfloat16 \
  --moe-backend triton \
  --enforce-eager \
  --no-enable-prefix-caching
```

Set `layers` to JSON `null` for all layers. The offline `generate` command
performs the post-generation callback check and saves counters; the server
command alone does not perform that check.

The intervention affects **both prompt processing and generated tokens** at
the selected layers, including any text outside `<think>`. Reasoning-only
intervention needs a separate per-token mask and is not implemented here.

## Upstream references

- [vLLM plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/):
  process-wide entry-point discovery and re-entrant registration.
- [Mixtral implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mixtral.py)
  and [MoE factory](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/layer.py):
  callback placement during construction.
- [Engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/):
  backend selection and eager execution.
- [LLM control RPC](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py):
  collecting diagnostics from inference workers.

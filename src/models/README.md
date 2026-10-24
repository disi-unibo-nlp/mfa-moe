# src/models

Loads the checkpoints and reads the routers of each model family. It is the base of `capture/`, and the facts below are what the capture and steering code assume.

| File | What it does |
| --- | --- |
| `loader.py` | Loads a pinned Hugging Face snapshot in bfloat16 (quantised checkpoints keep their own format), with its tokenizer. |
| `router_adapters.py` | Hooks that stream the router signal, the experts that were really selected, their weights and the hidden state of each routed layer, keeping the absolute layer index. |
| `inference.py` | Teacher-forced forward pass over a saved prompt and completion. |
| `token_replay.py` | Saves and checks the exact token ids of a generation, with a character span for each token. |
| `engine.py` | The offline vLLM settings of the study models, and the marker where reasoning closes. |

## The models

| Name | Checkpoint (revision) | Weights | Routed layers L, experts E, selected k | Context |
| --- | --- | --- | --- | --- |
| `gpt` | `openai/gpt-oss-20b` (`6cee5e8`) | MXFP4 | 24, 32, 4 | 131,072 |
| `nemotron` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (`bee7596`) | NVFP4 | 23, 128, 6 | 133,120 |
| `qwen35` | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (`3af5ca2`) | GPTQ-Int4 | 40, 256, 8 | 133,120 |
| `qwen36` | `Qwen/Qwen3.6-35B-A3B-FP8` (`95a723d`) | FP8 | 40, 256, 8 | 133,120 |
| `gemma` | `nvidia/Gemma-4-26B-A4B-NVFP4` (`a19cfe0`) | NVFP4 | 30, 128, 8 | 133,120 |
| `glm` | `zai-org/GLM-4.7-Flash` (`7dd2089`) | BF16 | 46, 64, 4 | 133,120 |
| `qwen330b` | `Qwen/Qwen3-30B-A3B` (`ad44e77`) | BF16 | 48, 128, 8 | 40,960 |
| `lfm25` | `LiquidAI/LFM2.5-24B-A2B` (`1769516`) | BF16 | 38, 64, 4 | 131,072 |

The router captures replay the same checkpoint that generated the traces, so the routes are those of the model that wrote the text. GPT-OSS and Qwen3-30B are captured with Hugging Face transformers, the others with vLLM on two GPUs. LFM2.5 runs in the steering engine (`capture/lfm_replay.py`).

## What the routers return

The families differ, and the code never uses raw top-k when it can read the real selection.

- Qwen3.5, Qwen3.6 and Gemma expose probabilities. They are stored as log-probabilities, so later softmax steps give the native distribution again.
- Qwen3 and GPT-OSS expose logits.
- GLM and Nemotron use sigmoid affinities plus a selection-correction bias (and grouped selection). A top-k of the raw affinities would name the wrong experts. The hook reads the experts the model dispatched; the affinities are normalised only for entropy and similar measures.
- The selected set can differ from the raw ranking, so the gap between the last selected and the best unselected expert can be negative.

## What a replay needs

- The saved prompt and completion token ids, from vLLM's `return_token_ids`. They are replayed directly. Chat templates, Harmony channels and `<think>` wrappers are never rebuilt from parsed text.
- The tokenizer of the replay must have the same vocabulary hash as the one that generated the text (`validate_token_replay`), and the saved ids must decode to the saved text.
- Character offsets per token come from an incremental decode, so a character spread over several tokens is shared by them. Labels and views use these offsets.
- Final-answer scoring still uses the parsed answer text. Token-limit status and `reasoning_tokens` are described in [`src/generation`](../generation/README.md).

Tests: [`tests/models`](../../tests/models/README.md) and [`tests/generation`](../../tests/generation/README.md) (tiny real architectures, no downloads).

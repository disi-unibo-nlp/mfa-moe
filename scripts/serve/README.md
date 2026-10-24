# scripts/serve

One short vLLM server script per model, plus one for the judge. You do not run them by hand. A job that needs a model calls `start_server NAME` from `scripts/slurm/env.sh`, which starts `scripts/serve/NAME.sh` in the background and waits for it.

`args.sh` holds the flags every server shares: two GPUs with tensor parallelism, bfloat16 weights and KV cache (the A100 has no FP8 KV cache), prefix caching, 8,192 batched tokens, 90 percent of GPU memory, and a local-only address on the job's port.

| Script | Model and revision | Reasoning parser | Context | Notes |
| --- | --- | --- | --- | --- |
| `gpt.sh` | `openai/gpt-oss-20b` `6cee5e8` | `openai_gptoss` | 131,072 | Native MXFP4 weights. Sampling and the medium reasoning level are set by the client. |
| `nemotron.sh` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` `bee7596` | `nemotron_v3` | 133,120 | DSpark draft model, 3 speculative tokens. Triton attention and Mamba backend, Marlin linear layers. |
| `qwen35.sh` | `Qwen3.5-35B-A3B-GPTQ-Int4` `3af5ca2` (local copy) | `qwen3` | 133,120 | MTP, 3 speculative tokens. Text only. |
| `qwen36.sh` | `Qwen/Qwen3.6-35B-A3B-FP8` `95a723d` | `qwen3` | 133,120 | MTP, 3 speculative tokens. Text only. |
| `gemma.sh` | `nvidia/Gemma-4-26B-A4B-NVFP4` `a19cfe0` (local copy) | `gemma4` | 133,120 | Text only, Marlin linear layers. Thinking is switched on by the client. |
| `glm.sh` | `zai-org/GLM-4.7-Flash` `7dd2089` | `glm45` | 133,120 | `TRITON_MLA` attention: GLM uses MLA kernels, and this is the only MLA backend that supports the A100. |
| `qwen330b.sh` | `Qwen/Qwen3-30B-A3B` `ad44e77` | `qwen3` | 40,960 | The model's native window, no YaRN. No MTP head. |
| `judge.sh` | `Qwen/Qwen3.8-27B` `1d4bf0f` | `qwen3` | 49,152 | The sentence labeller and answer checker. MTP, 3 tokens, eager mode, 64 sequences. |

LFM2.5 has no script here. Its generation, labeling and replay run inside the steering code (`src/steering`), which owns its engine.

Why the A100 needs these settings (found in probe jobs in September 2026):

- Nemotron: the FlashInfer paged-prefill kernel is invalid on this driver stack, so attention is `TRITON_ATTN`. A native FP8 KV cache needs SM89 or newer, so the KV cache is bfloat16. The checkpoint declares ModelOpt mixed precision (W4A16 NVFP4 experts, FP8 mixer projections), so no `--quantization` flag is passed; vLLM picks the kernels, and `--linear-backend marlin` serves both. FlashInfer's Mamba kernel fails to compile (a gcc internal error), so the Mamba backend is Triton with `--mamba-cache-mode align`. DSpark is validated by NVIDIA only on Hopper and Blackwell, but it works here: mean acceptance length 2.9 to 3.5, 64 to 82 percent of draft tokens accepted. Rerun a probe before trusting another node, driver or vLLM build, and fall back to no speculation if it fails.
- GLM-4.7-Flash: vLLM 0.29 lists `glm4_moe_lite` as an MLA model, where `FLASH_ATTN` is not a valid backend and `TRITON_MLA` is the only MLA backend that supports SM80. `glm45` is an alias that maps to the GLM-4.7 parser, not a legacy one. The checkpoint is not multimodal, so `--language-model-only` must not be passed.
- Qwen3-30B-A3B: plain GQA attention, native 40,960-token window, no MTP head, not multimodal.
- Qwen3.5, Qwen3.6, Gemma: multimodal checkpoints, served text only (`--language-model-only`).

Things that cost time before:

- Readiness is slow. A bfloat16 Qwen3-30B-A3B needs about 750 s before `/health` answers, so the wait in `env.sh` is 45 minutes.
- The checkpoints of `qwen35` and `gemma` are local copies under `mfa-moe/cache/native-replay/`, because the compute nodes are offline.
- `src/generation/model_profiles.py` records the same settings (revision, parser, context, attention, speculation) in every trace, and `tests/generation/test_serve_scripts.py` checks that it matches these scripts. Change both together.

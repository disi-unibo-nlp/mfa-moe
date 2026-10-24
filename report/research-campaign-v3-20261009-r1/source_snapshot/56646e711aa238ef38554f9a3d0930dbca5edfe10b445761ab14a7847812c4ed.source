"""Server defaults for the supported correlation checkpoints (stdlib only).

The shell launcher executes this file on the host; model dependencies stay in
the project container. Match checkpoint names, including quantized variants,
but never substitute another checkpoint for the one the caller selected.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    family: str
    reasoning_parser: str
    mtp: bool
    context: int
    forward_quantization: str = "none"
    language_model_only: bool = False
    num_layers: int | None = None


def model_profile(model: str) -> ModelProfile:
    name = model.rsplit("/", 1)[-1].lower()
    if name.startswith(("qwen3.5-35b-a3b", "qwen3.6-35b-a3b")):
        return ModelProfile("qwen3_5_moe", "qwen3", True, 262144, "unsloth-4bit", True, 40)
    if name.startswith("qwen3-30b-a3b"):
        return ModelProfile("qwen3_moe", "qwen3", False, 40960, num_layers=48)
    if name.startswith("nvidia-nemotron-3.5-lightning-30b-a3b"):
        return ModelProfile("nemotron_h", "nemotron_v3", True, 262144, num_layers=52)
    if name.startswith("gemma-4-26b-a4b-it") or name == "gemma-4-26b-a4b-nvfp4":
        return ModelProfile("gemma4", "gemma4", False, 262144, language_model_only=True, num_layers=30)
    if name.startswith("glm-4.7-flash"):
        return ModelProfile("glm4_moe_lite", "glm45", True, 202752, num_layers=47)
    if name.startswith("gpt-oss-20b"):
        return ModelProfile("gpt_oss", "openai_gptoss", False, 131072, num_layers=24)
    # Retain the existing custom-alias behavior. The seven explicit profiles
    # above are the supported set; arbitrary architectures are not inferred.
    return ModelProfile("custom", "qwen3", True, 49152, "unsloth-4bit", True)


P_ROOT = "/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe"
CARD_OUTPUT_TOKENS = 131_072
CARD_CONTEXT = 133_120  # output cap plus prompt room; every chosen checkpoint supports it

# Model-card sampling for the v3 primary corpus (user decision 2026-09-25): each model's
# own recommended thinking-mode sampler, and a 131,072-token output cap wherever the
# native context allows. Citations point at the local card / generation_config snapshot.
CARD_PROFILES: dict[str, dict] = {
    "qwen36": dict(
        model="Qwen/Qwen3.6-35B-A3B-FP8", revision="95a723d08a9490559dae23d0cff1d9466213d989",
        weights=None, tokenizer="Qwen/Qwen3.6-35B-A3B-FP8", parser="qwen3",
        sampler=dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,
                     repetition_penalty=1.0),
        max_tokens=CARD_OUTPUT_TOKENS, max_model_len=CARD_CONTEXT, template_kwargs={},
        reasoning_effort=None, speculation={"method": "mtp", "num_speculative_tokens": 3},
        attention="FLASH_ATTN", language_model_only=True, max_num_seqs=16, extra_args=[],
        card="Qwen3.6-35B-A3B-FP8 README.md:670 (thinking, general tasks)"),
    "qwen35": dict(
        model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", revision="3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b",
        weights=f"{P_ROOT}/cache/native-replay/Qwen3.5-35B-A3B-GPTQ-Int4/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b",
        tokenizer=None, parser="qwen3",
        sampler=dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,
                     repetition_penalty=1.0),
        max_tokens=CARD_OUTPUT_TOKENS, max_model_len=CARD_CONTEXT, template_kwargs={},
        reasoning_effort=None, speculation={"method": "mtp", "num_speculative_tokens": 3},
        attention="FLASH_ATTN", language_model_only=True, max_num_seqs=16, extra_args=[],
        card="Qwen3.5-35B-A3B-GPTQ-Int4 README.md:1008 (thinking, general tasks)"),
    "glm": dict(
        model="zai-org/GLM-4.7-Flash", revision="7dd20894a642a0aa287e9827cb1a1f7f91386b67",
        weights=None, tokenizer="zai-org/GLM-4.7-Flash", parser="glm45",
        sampler=dict(temperature=1.0, top_p=0.95, top_k=0, min_p=None, presence_penalty=None,
                     repetition_penalty=None),
        max_tokens=CARD_OUTPUT_TOKENS, max_model_len=CARD_CONTEXT, template_kwargs={},
        reasoning_effort=None, speculation=None, attention="TRITON_MLA",
        language_model_only=False, max_num_seqs=8, extra_args=[],
        card="GLM-4.7-Flash README.md:48-50 (T 1.0, top-p 0.95, max new tokens 131072)"),
    "nemotron": dict(
        model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        revision="bee7596271d1495f6992ae224aefde4410e816b8", weights=None,
        tokenizer="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", parser="nemotron_v3",
        sampler=dict(temperature=1.0, top_p=0.95, top_k=0, min_p=None, presence_penalty=None,
                     repetition_penalty=None),
        max_tokens=CARD_OUTPUT_TOKENS, max_model_len=CARD_CONTEXT, template_kwargs={},
        reasoning_effort=None,
        speculation={"method": "dspark",
                     "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
                     "revision": "8a0177116d138011e63103110f136ec0ca09ebbf",
                     "num_speculative_tokens": 3},
        attention="TRITON_ATTN", language_model_only=False, max_num_seqs=16,
        extra_args=["--linear-backend", "marlin", "--mamba-backend", "triton",
                    "--mamba-cache-mode", "align"],
        card="Nemotron-3.5-Lightning NVFP4 README.md:54,606 (T 1.0, top-p 0.95)"),
    "gemma": dict(
        model="nvidia/Gemma-4-26B-A4B-NVFP4", revision="a19cfe00be84568a6867111c9a68c9c44fdcffe6",
        weights=f"{P_ROOT}/cache/native-replay/Gemma-4-26B-A4B-NVFP4/a19cfe00be84568a6867111c9a68c9c44fdcffe6",
        tokenizer=None, parser="gemma4",
        sampler=dict(temperature=1.0, top_p=0.95, top_k=64, min_p=None, presence_penalty=None,
                     repetition_penalty=None),
        max_tokens=CARD_OUTPUT_TOKENS, max_model_len=CARD_CONTEXT,
        template_kwargs={"enable_thinking": True}, reasoning_effort=None, speculation=None,
        attention=None, language_model_only=True, max_num_seqs=8,
        extra_args=["--linear-backend", "marlin"],
        card="Gemma-4-26B-A4B-NVFP4 generation_config.json:11-13 (T 1.0, top-k 64, top-p 0.95); "
             "BF16 KV cache because the FP8 KV cache is unsupported on A100"),
    "qwen330b": dict(
        model="Qwen/Qwen3-30B-A3B", revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        weights=None, tokenizer="Qwen/Qwen3-30B-A3B", parser="qwen3",
        sampler=dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=None,
                     repetition_penalty=None),
        max_tokens=38_912, max_model_len=40_960, template_kwargs={}, reasoning_effort=None,
        speculation=None, attention="FLASH_ATTN", language_model_only=False, max_num_seqs=16,
        extra_args=[],
        card="Qwen3-30B-A3B README.md:131,329 (thinking T 0.6, top-p 0.95, top-k 20, min-p 0); "
             "38,912 competition output length; native 40,960 context (no YaRN)"),
    "gpt": dict(
        model="openai/gpt-oss-20b", revision="6cee5e81ee83917806bbde320786a8fb61efebee",
        weights=None, tokenizer="openai/gpt-oss-20b", parser="openai_gptoss",
        sampler=dict(temperature=1.0, top_p=1.0, top_k=0, min_p=None, presence_penalty=None,
                     repetition_penalty=None),
        max_tokens=None, max_model_len=131_072, template_kwargs={}, reasoning_effort="medium",
        speculation=None, attention="TRITON_ATTN", language_model_only=False, max_num_seqs=16,
        extra_args=[],
        card="gpt-oss-20b card gives no sampler: checkpoint defaults (T 1, top-p 1) and the "
             "default medium reasoning level (README.md:147-155); output bounded by the "
             "131,072-token context"),
}


def card_server_argv(source: str, port: int, *, download_dir: str | None = None,
                     tensor_parallel_size: int = 2, max_num_seqs: int | None = None,
                     speculation: bool = True, max_model_len: int | None = None) -> list[str]:
    c = CARD_PROFILES[source]
    argv = ["serve", c["weights"] or c["model"]]
    if not c["weights"]:
        argv += ["--revision", c["revision"]]
        if download_dir:
            argv += ["--download-dir", download_dir]
    argv += ["--served-model-name", c["model"], "--host", "127.0.0.1", "--port", str(port),
             "--api-key", "local-vllm-key", "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16"]
    if c["attention"]:
        argv += ["--attention-config", json.dumps({"backend": c["attention"]})]
    argv += ["--reasoning-parser", c["parser"], "--generation-config", "vllm",
             "--enable-prefix-caching"]
    if c["language_model_only"]:
        argv.append("--language-model-only")
    argv += ["--max-model-len", str(max_model_len or c["max_model_len"]),
             "--max-num-seqs", str(max_num_seqs or c["max_num_seqs"]),
             "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.90",
             "--tensor-parallel-size", str(tensor_parallel_size), *c["extra_args"]]
    if speculation and c["speculation"]:
        argv += ["--speculative-config", json.dumps(c["speculation"])]
    return argv


def card_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Card-profile helpers for launchers")
    parser.add_argument("command", choices=["server-argv", "profile-json"])
    parser.add_argument("--source", required=True, choices=sorted(CARD_PROFILES))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--download-dir")
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--no-speculation", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "profile-json":
        print(json.dumps(CARD_PROFILES[args.source], sort_keys=True))
        return
    for item in card_server_argv(args.source, args.port, download_dir=args.download_dir,
                                 max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
                                 speculation=not args.no_speculation):
        print(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-model", required=True)
    parser.add_argument("--forward-model", required=True)
    parser.add_argument("--check-probe-results", type=Path)
    args = parser.parse_args()
    generation = model_profile(args.generation_model)
    forward = model_profile(args.forward_model)
    if args.check_probe_results is not None:
        try:
            best = json.loads(args.check_probe_results.read_text())["best_by_target"]
            indices = sorted({int(row["layer_idx"]) for row in best.values()})
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(f"Cannot read fixed probe indices: {error}")
        if not indices or (forward.num_layers is not None and not any(
            0 <= index < forward.num_layers for index in indices
        )):
            parser.error(
                f"Probe indices {indices} do not exist in {args.forward_model} "
                f"({forward.num_layers} layers). Use --all-router-layers to explicitly "
                "override the fixed selection. No models have been started."
            )
    for value in (
        generation.family, generation.reasoning_parser,
        str(generation.mtp).lower(), min(49152, generation.context),
        forward.forward_quantization, str(generation.language_model_only).lower(),
    ):
        print(value)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("server-argv", "profile-json"):
        card_cli(sys.argv[1:])
    else:
        main()

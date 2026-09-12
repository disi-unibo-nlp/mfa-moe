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
    main()

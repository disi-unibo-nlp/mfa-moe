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
        quantization = "bnb-4bit" if name == "gemma-4-26b-a4b-it" else "none"
        return ModelProfile("gemma4", "gemma4", False, 262144, quantization,
                            language_model_only=True, num_layers=30)
    if name.startswith("glm-4.7-flash"):
        return ModelProfile("glm4_moe_lite", "glm45", True, 202752, num_layers=47)
    if name.startswith("gpt-oss-20b"):
        return ModelProfile("gpt_oss", "openai_gptoss", False, 131072, "mxfp4-bf16", num_layers=24)
    # Retain the existing custom-alias behavior. The seven explicit profiles
    # above are the supported set; arbitrary architectures are not inferred.
    return ModelProfile("custom", "qwen3", True, 49152, "unsloth-4bit", True)


def default_probe_results(model: str) -> Path:
    """Select independently trained gold probes for each supported probe model."""
    slug = {
        "gpt_oss": "gpt-oss-20b",
        "gemma4": "gemma-4-26b-a4b-it-nf4",
    }.get(model_profile(model).family, "qwen3.5-35b-a3b-gptq-int4")
    return Path("results/probeTest") / slug / "probes/results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-model", required=True)
    parser.add_argument("--forward-model", required=True)
    parser.add_argument("--probe-results", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--check-probe-results", type=Path, nargs="?", const=Path("auto"))
    args = parser.parse_args()
    generation = model_profile(args.generation_model)
    forward = model_profile(args.forward_model)
    probe_results = args.probe_results or default_probe_results(args.forward_model)
    if args.check_probe_results is not None:
        check_path = (
            probe_results if args.check_probe_results == Path("auto") else args.check_probe_results
        )
        check_path = args.repo_root / check_path
        try:
            best = json.loads(check_path.read_text())["best_by_target"]
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
        probe_results.as_posix(),
    ):
        print(value)


if __name__ == "__main__":
    main()

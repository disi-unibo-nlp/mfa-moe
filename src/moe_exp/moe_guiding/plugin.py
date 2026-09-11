"""Worker-safe vLLM entry point; ordinary model architectures are unaffected."""

from moe_exp.moe_guiding.config import ARCHITECTURE


def register() -> None:
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE,
            "moe_exp.moe_guiding.model:MoEGuidingMixtralForCausalLM",
        )

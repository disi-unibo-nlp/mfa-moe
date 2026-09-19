"""Expert counts for the experiment checkpoints; workers verify actual configs."""
from moe_exp.correlation_pipeline.model_profiles import model_profile


def expert_defaults(model: str) -> tuple[int, int]:
    family = model_profile(model).family
    defaults = {"qwen3_5_moe": (256, 8), "gpt_oss": (32, 4), "gemma4": (128, 8)}
    if family not in defaults:
        raise ValueError(f"No identity-guiding preset for checkpoint {model}")
    return defaults[family]

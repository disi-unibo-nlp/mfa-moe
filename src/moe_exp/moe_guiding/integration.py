from __future__ import annotations

import inspect
import re
from contextlib import contextmanager
from functools import wraps
from types import ModuleType

import torch

from moe_exp.moe_guiding.config import RoutingConfig
from moe_exp.moe_guiding.routing import MarginRouter


def validate_vllm_config(vllm_config, config: RoutingConfig) -> tuple[int, ...]:
    model = vllm_config.model_config
    hf_config = model.hf_config
    if getattr(hf_config, "model_type", None) != "mixtral":
        raise ValueError(
            "The moe_guiding vLLM adapter currently supports Mixtral checkpoints only."
        )
    if hf_config.num_experts_per_tok != 2 or hf_config.num_local_experts < 3:
        raise ValueError("moe_guiding requires native top-2 routing with at least 3 experts.")
    if (
        vllm_config.quant_config is not None
        or getattr(model, "quantization", None) is not None
        or getattr(hf_config, "quantization_config", None) is not None
    ):
        raise ValueError("Use an unquantized FP16/BF16 checkpoint for this initial experiment.")
    if model.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("moe_guiding requires FP16 or BF16 model precision.")
    kernel = getattr(vllm_config, "kernel_config", None)
    if getattr(kernel, "moe_backend", None) != "triton":
        raise ValueError("Set --moe-backend triton; other backends may bypass custom routing.")
    if not model.enforce_eager:
        raise ValueError("Set --enforce-eager for this instrumented experiment.")
    parallel = vllm_config.parallel_config
    if (
        parallel.enable_expert_parallel
        or parallel.enable_eplb
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or getattr(parallel, "enable_dbo", False)
    ):
        raise ValueError("Use tensor parallelism only; EP, EPLB, PP, DP and DBO are unsupported.")
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise ValueError("Disable speculative decoding for the initial routing experiment.")
    return config.selected_layers(hf_config.num_hidden_layers)


@contextmanager
def routing_construction(
    module: ModuleType, config: RoutingConfig, selected_layers: tuple[int, ...]
):
    """Inject callbacks into Mixtral's factory call before backend construction.

    The temporary binding is local to the worker's Mixtral module and is restored
    even if construction fails. Older FusedMoE constructor names are also accepted
    when they expose the same callback keyword.
    """
    name = "FusedMoEFactory" if hasattr(module, "FusedMoEFactory") else "FusedMoE"
    original = getattr(module, name)
    signature = inspect.signature(original)
    if "custom_routing_function" not in signature.parameters:
        raise RuntimeError("This vLLM MoE constructor does not expose custom_routing_function.")
    routers: dict[int, MarginRouter] = {}
    seen: set[int] = set()

    @wraps(original)
    def construct(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", bound.arguments.get("prefix", ""))
        if match is None:
            raise RuntimeError("Cannot identify the transformer layer in the Mixtral MoE prefix.")
        layer = int(match.group(1))
        if layer in selected_layers:
            seen.add(layer)
            if config.condition != "baseline":
                if bound.arguments.get("custom_routing_function") is not None:
                    raise RuntimeError("Refusing to overwrite an existing custom router.")
                router = MarginRouter(config)
                routers[layer] = router
                bound.arguments["custom_routing_function"] = router
        return original(*bound.args, **bound.kwargs)

    setattr(module, name, construct)
    try:
        yield routers
        missing = set(selected_layers) - seen
        if missing:
            raise RuntimeError(
                f"No MoE constructor was reached for selected layers {sorted(missing)}."
            )
    finally:
        setattr(module, name, original)


def routing_diagnostics(worker, reset: bool = False) -> dict:
    """Small control RPC executed inside each inference worker."""
    model = worker.model_runner.get_model()
    if not hasattr(model, "moe_guiding_routers"):
        raise RuntimeError("Worker did not load the moe_guiding model adapter.")
    routers = model.moe_guiding_routers
    if reset:
        for router in routers.values():
            router.reset()
    return {
        "condition": model.moe_guiding_config.condition,
        "selected_layers": list(model.moe_guiding_layers),
        "layers": {str(layer): router.snapshot() for layer, router in routers.items()},
    }

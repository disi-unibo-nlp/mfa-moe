"""Stream native MoE signals while retaining absolute decoder layer indices."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

ROUTER_CAPTURE_VERSION = 1
PROBABILITY_ROUTERS = {"qwen3_5_moe", "qwen3_5_moe_text", "gemma4", "gemma4_text"}
SIGMOID_ROUTERS = {"nemotron_h", "glm4_moe_lite"}
SUPPORTED_ROUTERS = PROBABILITY_ROUTERS | SIGMOID_ROUTERS | {"qwen3_moe", "gpt_oss"}


def model_family(model: Any) -> str | None:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    return getattr(text_config or config, "model_type", None)


def decoder_layers(model: Any):
    backbone = getattr(model, "model", model)
    decoder = getattr(backbone, "language_model", backbone)
    return getattr(decoder, "layers", None)


def router_module(layer: Any, family: str):
    if family in {"gemma4", "gemma4_text"}:
        return getattr(layer, "router", None)
    block = getattr(layer, "mixer" if family == "nemotron_h" else "mlp", None)
    return getattr(block, "router" if family == "gpt_oss" else "gate", None)


def available_router_layers(model: Any) -> list[int] | None:
    family = model_family(model)
    layers = decoder_layers(model)
    if family not in SUPPORTED_ROUTERS or not isinstance(layers, torch.nn.ModuleList):
        return None
    return [i for i, layer in enumerate(layers) if router_module(layer, family) is not None]


def configured_top_k(config: Any) -> int | None:
    config = getattr(config, "text_config", None) or config
    for name in ("num_experts_per_tok", "top_k_experts", "experts_per_token"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def stream_router_signals(
    model: Any, forward_model: Any, forward_kwargs: dict, prompt_len: int,
    requested_layers: set[int] | None, extract_hidden_states: bool,
    details: dict | None = None,
):
    family = model_family(model)
    layers = decoder_layers(model)
    available = available_router_layers(model)
    if available is None:
        raise ValueError("No streaming router adapter for this model")
    selected = available if requested_layers is None else sorted(requested_layers)
    missing = set(selected) - set(available)
    if missing:
        raise ValueError(f"Selected decoder layers have no MoE router: {sorted(missing)}")
    routers, hidden, experts, weights = {}, {}, {}, {}
    handles = []

    def cpu_slice(value):
        if value.ndim == 2:
            value = value.unsqueeze(0)
        return value[0, prompt_len:].detach().to(device="cpu", copy=True)

    def capture_selection(index, ids, values):
        ids, values = cpu_slice(ids), cpu_slice(values)
        # Some native routers return an unordered top-k. Persist the same set
        # ranked by its actual mixture weights for identity/rank analyses.
        order = values.argsort(dim=-1, descending=True, stable=True)
        experts[index] = ids.gather(-1, order)
        weights[index] = values.gather(-1, order)

    def router_hook(index):
        def capture(module, args, output):
            raw = output[0] if isinstance(output, tuple) else output
            values = cpu_slice(raw).float()
            if family in PROBABILITY_ROUTERS:
                values = values.clamp_min(torch.finfo(torch.float32).tiny).log()
            elif family in SIGMOID_ROUTERS:
                # Downstream softmax now yields normalized sigmoid affinities.
                # Selection still uses the model's correction bias/group mask.
                values = F.logsigmoid(values)
            routers[index] = values
            if isinstance(output, tuple) and len(output) >= 3:
                capture_selection(index, output[2], output[1])
        return capture

    def experts_hook(index):
        def capture(module, args, kwargs):
            ids = args[1] if len(args) > 1 else kwargs.get("top_k_index")
            values = args[2] if len(args) > 2 else kwargs.get("top_k_weights")
            if ids is None or values is None:
                raise RuntimeError("Could not capture the native expert selection")
            capture_selection(index, ids, values)
        return capture

    def hidden_hook(index):
        def capture(module, args, kwargs):
            value = args[0] if args else kwargs["hidden_states"]
            hidden[index] = cpu_slice(value)
        return capture

    try:
        for index in selected:
            layer = layers[index]
            gate = router_module(layer, family)
            handles.append(gate.register_forward_hook(router_hook(index)))
            # Transformers 5.5 GLM/Nemotron gates expose logits only; expert
            # inputs carry the actual bias- and group-adjusted routing decision.
            if family in SIGMOID_ROUTERS:
                block = getattr(layer, "mixer" if family == "nemotron_h" else "mlp")
                handles.append(block.experts.register_forward_pre_hook(
                    experts_hook(index), with_kwargs=True,
                ))
            if extract_hidden_states:
                handles.append(layer.register_forward_pre_hook(hidden_hook(index), with_kwargs=True))
        with torch.inference_mode():
            forward_model(**{
                **forward_kwargs, "output_router_logits": False,
                "output_hidden_states": False, "output_attentions": False,
            })
    finally:
        for handle in handles:
            handle.remove()
    if any(i not in routers or (extract_hidden_states and i not in hidden) for i in selected):
        raise RuntimeError("Forward did not capture every requested router/hidden layer")
    if details is not None:
        if any(i not in experts for i in selected):
            raise RuntimeError("Forward did not expose the native selected experts")
        details.update({
            "selected_experts": torch.stack([experts[i] for i in selected]),
            "expert_weights": torch.stack([weights[i] for i in selected]),
            "layer_indices": selected,
            "router_distribution": "normalized_sigmoid" if family in SIGMOID_ROUTERS else "softmax",
        })
    router_tensor = torch.stack([routers[i] for i in selected]) if selected else torch.empty(0)
    if extract_hidden_states:
        hidden_tensor = torch.stack([hidden[i] for i in selected]) if selected else torch.empty(0)
        return router_tensor, hidden_tensor
    return router_tensor

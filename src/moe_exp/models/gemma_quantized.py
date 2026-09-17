"""Text-only Gemma 4 replay with every gated expert quantized to NF4."""

from copy import deepcopy

import torch
from torch import nn
from transformers import BitsAndBytesConfig, Gemma4ForCausalLM
from transformers.activations import ACT2FN
from transformers.core_model_loading import ConversionOps, WeightConverter
from transformers.integrations.bitsandbytes import Bnb4bitQuantize
from transformers.quantizers.auto import register_quantizer
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer


class GemmaReplayNF4Config(BitsAndBytesConfig):
    def __init__(self, num_experts, num_hidden_layers):
        super().__init__(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head"] + [
                f"model.layers.{index}.router" for index in range(num_hidden_layers)
            ],
        )
        self.quant_method = "gemma-replay-nf4"
        self.num_experts = num_experts


class SplitExperts(ConversionOps):
    def convert(self, input_dict, target_patterns, **kwargs):
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        if tensor.ndim != 3 or tensor.shape[0] != len(target_patterns):
            raise ValueError("Gemma fused expert tensor has an unexpected shape")
        return dict(zip(target_patterns, tensor.unbind(0)))


class QuantizeSplitExperts(ConversionOps):
    """The stock bnb op accepts one tensor; a split produces multiple tensors."""

    def __init__(self, quantizer):
        self.quantize = Bnb4bitQuantize(quantizer)

    def convert(self, input_dict, model=None, **kwargs):
        result = {}
        for name, value in input_dict.items():
            tensors = value if isinstance(value, list) else [value]
            result.update(self.quantize.convert(
                {name: tensors}, model=model, full_layer_name=name,
            ))
        return result


@register_quantizer("gemma-replay-nf4")
class GemmaReplayNF4Quantizer(Bnb4BitHfQuantizer):
    def get_quantize_ops(self):
        return QuantizeSplitExperts(self)

    def get_weight_conversions(self):
        if self.pre_quantized:
            raise ValueError("Gemma NF4 replay requires an unquantized source checkpoint")
        return [
            WeightConverter(
                source_patterns=f"experts.{projection}",
                target_patterns=[
                    f"experts.{projection}s.{index}.weight"
                    for index in range(self.quantization_config.num_experts)
                ],
                operations=[SplitExperts()],
            ) for projection in ("gate_up_proj", "down_proj")
        ]

    def is_serializable(self):
        # This adapter consumes the original fused checkpoint; it does not
        # implement reloading an exported per-expert bitsandbytes checkpoint.
        return False


class GemmaLinearExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.gate_up_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, 2 * config.moe_intermediate_size, bias=False)
            for _ in range(self.num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
            for _ in range(self.num_experts)
        ])
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        for expert in top_k_index.unique().tolist():
            tokens, slots = torch.where(top_k_index == expert)
            gate, up = self.gate_up_projs[expert](hidden_states[tokens]).chunk(2, dim=-1)
            values = self.down_projs[expert](self.act_fn(gate) * up)
            values = values * top_k_weights[tokens, slots, None]
            result.index_add_(0, tokens, values.to(result.dtype))
        return result


class Gemma4ForQuantizedReplay(Gemma4ForCausalLM):
    # Intentionally load only the language model from the multimodal checkpoint.
    # Unexpected language-model weights must still be reported as errors.
    _keys_to_ignore_on_load_unexpected = [
        r"^model\.vision_tower\.", r"^model\.audio_tower\.",
        r"^model\.embed_vision\.", r"^model\.embed_audio\.",
    ]

    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            if hasattr(layer, "experts"):
                layer.experts = GemmaLinearExperts(config)


def load_gemma_4bit(model_id, config, *, offload_folder="offload", revision="main"):
    text_config = getattr(config, "text_config", config)
    if any(getattr(c, "quantization_config", None) is not None for c in (config, text_config)):
        raise ValueError(
            "Gemma bnb-4bit replay requires the unquantized checkpoint; "
            "select the NVFP4 checkpoint with --generation-model only."
        )
    if not text_config.enable_moe_block:
        raise ValueError("Gemma expert NF4 replay requires an MoE checkpoint")
    model, info = Gemma4ForQuantizedReplay.from_pretrained(
        model_id, config=deepcopy(text_config), dtype=torch.bfloat16, revision=revision,
        device_map="auto", offload_folder=offload_folder,
        experts_implementation="eager",
        quantization_config=GemmaReplayNF4Config(text_config.num_experts, text_config.num_hidden_layers),
        key_mapping={r"^model\.language_model\.": "model."},
        output_loading_info=True,
    )
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(name):
            raise RuntimeError(f"Gemma replay checkpoint {name}: {list(info[name])[:8]}")
    validate_quantized_experts(model)
    return model


def validate_quantized_experts(model):
    from bitsandbytes.nn import Linear4bit

    count = 0
    for layer in model.model.layers:
        if not hasattr(layer, "experts"):
            continue
        if not isinstance(layer.experts, GemmaLinearExperts):
            raise RuntimeError("Gemma experts were not converted to quantizable layers")
        if isinstance(layer.router.proj, Linear4bit):
            raise RuntimeError("Gemma replay must keep the router projection in floating point")
        for linear in (*layer.experts.gate_up_projs, *layer.experts.down_projs):
            if not isinstance(linear, Linear4bit) or linear.weight.quant_state is None:
                raise RuntimeError("Gemma expert weights are not materialized in 4-bit")
            if linear.weight.device.type != "cuda":
                raise RuntimeError("Gemma 4-bit replay expects all expert weights on CUDA")
            count += 1
    if not count:
        raise RuntimeError("No quantized Gemma experts were found")
    return count

"""Qwen3.5/3.6 text replay with every routed expert loaded in bitsandbytes."""
from copy import deepcopy

import torch
from torch import nn
from transformers import BitsAndBytesConfig, Qwen3_5MoeForCausalLM
from transformers.activations import ACT2FN
from transformers.core_model_loading import ConversionOps, WeightConverter
from transformers.integrations.bitsandbytes import Bnb4bitQuantize, Bnb8bitQuantize
from transformers.quantizers.auto import register_quantizer
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
from transformers.quantizers.quantizer_bnb_8bit import Bnb8BitHfQuantizer


class QwenReplayConfig(BitsAndBytesConfig):
    def __init__(self, num_experts, bits):
        if bits not in (4, 8):
            raise ValueError("Qwen replay supports 4 or 8 bits")
        super().__init__(
            load_in_4bit=bits == 4, load_in_8bit=bits == 8,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # Recurrent kernels directly access linear-attention projection weights.
            llm_int8_skip_modules=["lm_head", "gate", "shared_expert_gate", r".*\.linear_attn(?:\.|$)"],
        )
        self.quant_method = f"qwen-replay-{bits}bit"
        self.num_experts = num_experts


class SplitExperts(ConversionOps):
    def convert(self, input_dict, target_patterns, **kwargs):
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        if tensor.ndim != 3 or tensor.shape[0] != len(target_patterns):
            raise ValueError("Qwen fused expert tensor has an unexpected shape")
        return dict(zip(target_patterns, tensor.unbind(0)))


class QuantizeSplitExperts(ConversionOps):
    def __init__(self, quantizer):
        operation = Bnb4bitQuantize if quantizer.quantization_config.load_in_4bit else Bnb8bitQuantize
        self.quantize = operation(quantizer)

    def convert(self, input_dict, model=None, **kwargs):
        result = {}
        for name, value in input_dict.items():
            tensors = value if isinstance(value, list) else [value]
            result.update(self.quantize.convert(
                {name: tensors}, model=model, full_layer_name=name,
            ))
        return result


class QwenReplayConversions:
    def get_quantize_ops(self):
        return QuantizeSplitExperts(self)

    def get_weight_conversions(self):
        if self.pre_quantized:
            raise ValueError("Qwen replay requires an unquantized source checkpoint")
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
        return False


@register_quantizer("qwen-replay-4bit")
class QwenReplay4BitQuantizer(QwenReplayConversions, Bnb4BitHfQuantizer):
    pass


@register_quantizer("qwen-replay-8bit")
class QwenReplay8BitQuantizer(QwenReplayConversions, Bnb8BitHfQuantizer):
    pass


class QwenLinearExperts(nn.Module):
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
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        for expert in top_k_index.unique().tolist():
            tokens, slots = torch.where(top_k_index == expert)
            gate, up = self.gate_up_projs[expert](hidden_states[tokens]).chunk(2, dim=-1)
            values = self.down_projs[expert](self.act_fn(gate) * up)
            values = values * top_k_weights[tokens, slots, None]
            result.index_add_(0, tokens, values.to(result.dtype))
        return result


class QwenForQuantizedReplay(Qwen3_5MoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            layer.mlp.experts = QwenLinearExperts(config)


def validate_quantized_experts(model, bits=4):
    from bitsandbytes.nn import Linear4bit, Linear8bitLt

    expected = Linear4bit if bits == 4 else Linear8bitLt
    count = 0
    for layer in model.model.layers:
        if not isinstance(layer.mlp.experts, QwenLinearExperts):
            raise RuntimeError("Qwen experts were not converted to quantizable layers")
        if not layer.mlp.gate.weight.is_floating_point():
            raise RuntimeError("Qwen replay must keep router weights floating point")
        for linear in (*layer.mlp.experts.gate_up_projs, *layer.mlp.experts.down_projs):
            if not isinstance(linear, expected) or linear.weight.device.type != "cuda":
                raise RuntimeError(f"Qwen expert weights are not materialized in {bits}-bit on CUDA")
            if bits == 4 and linear.weight.quant_state is None:
                raise RuntimeError("Qwen NF4 expert quantization state is missing")
            if bits == 8 and linear.weight.dtype != torch.int8:
                raise RuntimeError("Qwen int8 expert storage is missing")
            count += 1
    if not count:
        raise RuntimeError("No quantized Qwen experts were found")
    return count


def load_qwen_quantized(model_id, config, *, offload_folder="offload", bits=4):
    text_config = getattr(config, "text_config", config)
    if any(getattr(c, "quantization_config", None) is not None for c in (config, text_config)):
        raise ValueError("Qwen bitsandbytes replay requires the unquantized checkpoint, not FP8 weights")
    model, info = QwenForQuantizedReplay.from_pretrained(
        model_id, config=deepcopy(text_config), dtype=torch.bfloat16,
        device_map="auto", offload_folder=offload_folder,
        experts_implementation="eager",
        quantization_config=QwenReplayConfig(text_config.num_experts, bits),
        key_mapping={r"^model\.language_model\.": "model."},
        output_loading_info=True,
    )
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(name):
            raise RuntimeError(f"Qwen replay checkpoint {name}: {list(info[name])[:8]}")
    count = validate_quantized_experts(model, bits)
    model._replay_quantization = {
        "method": "bitsandbytes", "bits": bits, "expert_linear_count": count,
        "compute_dtype": "bfloat16", "router_dtype": str(model.model.layers[0].mlp.gate.weight.dtype),
    }
    return model

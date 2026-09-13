"""Load Nemotron's non-gated experts as individually quantizable linear layers.

Transformers 5.5 packs Nemotron experts into bare 3-D parameters, which its
bitsandbytes integration does not quantize. Keep the official checkpoint's
per-expert layout instead. Routing and Mamba computation remain native HF code.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import BitsAndBytesConfig, NemotronHForCausalLM
from transformers.activations import ACT2FN


# Apply these renamings before HF's default MergeModulelist conversions. The
# resulting paths no longer match mixer.experts.*.up_proj.weight (or down_proj).
EXPERT_KEY_MAPPING = {
    r"mixer\.experts\.(\d+)\.up_proj\.weight": r"mixer.experts.up_projs.\1.weight",
    r"mixer\.experts\.(\d+)\.down_proj\.weight": r"mixer.experts.down_projs.\1.weight",
}


class NemotronLinearExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        input_dim = config.moe_latent_size or config.hidden_size
        self.up_projs = nn.ModuleList([
            nn.Linear(input_dim, config.moe_intermediate_size, bias=False)
            for _ in range(self.num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(config.moe_intermediate_size, input_dim, bias=False)
            for _ in range(self.num_experts)
        ])
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        # Avoid a [tokens, top_k, experts] one-hot allocation on long replays.
        for expert in top_k_index.unique().tolist():
            token_indices, slots = torch.where(top_k_index == expert)
            values = self.up_projs[expert](hidden_states[token_indices])
            values = self.down_projs[expert](self.act_fn(values))
            values = values * top_k_weights[token_indices, slots, None]
            result.index_add_(0, token_indices, values.to(result.dtype))
        return result.to(hidden_states.dtype)


class NemotronHForQuantizedReplay(NemotronHForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            if hasattr(layer.mixer, "experts"):
                layer.mixer.experts = NemotronLinearExperts(config)


def load_nemotron_4bit(model_id, config, *, offload_folder="offload"):
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError(
            "Nemotron bnb-4bit replay requires the BF16 checkpoint as input; "
            "select the NVFP4 checkpoint with --generation-model only."
        )
    # Fused Mamba kernels access out_proj.weight directly instead of calling
    # Linear.forward. Keep entire Mamba mixers in BF16, including their states.
    skip_modules = ["lm_head"] + [
        f"model.layers.{index}.mixer"
        for index, kind in enumerate(config.layers_block_type) if kind == "mamba"
    ]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules,
    )
    model, loading_info = NemotronHForQuantizedReplay.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16,
        device_map="auto",
        offload_folder=offload_folder,
        quantization_config=quantization_config,
        experts_implementation="eager",
        key_mapping=EXPERT_KEY_MAPPING,
        output_loading_info=True,
    )
    # A changed upstream mapping must fail rather than run randomly initialized
    # experts and produce plausible but invalid routing measurements.
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(name):
            raise RuntimeError(f"Nemotron replay checkpoint {name}: {list(loading_info[name])[:8]}")
    validate_quantized_experts(model)
    return model


def validate_quantized_experts(model):
    from bitsandbytes.nn import Linear4bit

    count = 0
    for layer in model.model.layers:
        experts = getattr(layer.mixer, "experts", None)
        if experts is None:
            continue
        if not isinstance(experts, NemotronLinearExperts):
            raise RuntimeError("Nemotron experts were not converted to quantizable layers")
        for linear in (*experts.up_projs, *experts.down_projs):
            if not isinstance(linear, Linear4bit) or linear.weight.quant_state is None:
                raise RuntimeError("Nemotron expert weights are not materialized in 4-bit")
            if linear.weight.device.type != "cuda":
                raise RuntimeError("Nemotron 4-bit replay expects all expert weights on CUDA")
            count += 1
    if not count:
        raise RuntimeError("No quantized Nemotron experts were found")
    return count

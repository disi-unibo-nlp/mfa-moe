"""Check real checkpoint conversion, expert math, and quantized router capture."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models
from transformers import NemotronHConfig, NemotronHForCausalLM, PreTrainedTokenizerFast

from moe_exp.models.loader import load_model_and_tokenizer
from moe_exp.models.nemotron_quantized import (
    EXPERT_KEY_MAPPING,
    NemotronHForQuantizedReplay,
    NemotronLinearExperts,
    load_nemotron_4bit,
    validate_quantized_experts,
)
from moe_exp.models.router_adapters import stream_router_signals


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    # Exercise the native torch Mamba reference without downloading optional kernels.
    monkeypatch.setattr(
        "transformers.models.nemotron_h.modeling_nemotron_h.lazy_load_kernel", lambda name: None,
    )
    torch.manual_seed(12)
    config = NemotronHConfig(
        vocab_size=128, hidden_size=64, num_hidden_layers=4,
        layers_block_type=["mamba", "moe", "attention", "moe"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, n_routed_experts=4, num_experts_per_tok=2,
        n_shared_experts=1, moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=64, n_group=1, topk_group=1,
        mamba_num_heads=4, mamba_head_dim=16, n_groups=1,
        ssm_state_size=4, chunk_size=8, use_mamba_kernels=False,
        pad_token_id=0,
    )
    model = NemotronHForCausalLM(config).eval()
    weights = {}
    for name, value in model.state_dict().items():
        # Match NVIDIA's original on-disk keys, not HF's fused representation.
        name = name.replace("model.", "backbone.").replace("embeddings.weight", "embedding.weight")
        if name.endswith(("experts.up_proj", "experts.down_proj")):
            prefix, projection = name.rsplit(".", 1)
            for index, expert in enumerate(value):
                weights[f"{prefix}.{index}.{projection}.weight"] = expert.contiguous()
        else:
            weights[name] = value.contiguous()
    config.save_pretrained(tmp_path)
    save_file(weights, tmp_path / "model.safetensors")
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]", eos_token="[EOS]",
    ).save_pretrained(tmp_path)
    return tmp_path, model


def test_checkpoint_mapping_preserves_native_forward_and_routes(checkpoint):
    path, original = checkpoint
    model, info = NemotronHForQuantizedReplay.from_pretrained(
        path, key_mapping=EXPERT_KEY_MAPPING, output_loading_info=True,
    )
    assert not any(info.values()), info
    ids = torch.tensor([[1, 8, 12, 16, 9, 7, 2, 4]])
    kwargs = dict(input_ids=ids, use_cache=False)
    with torch.inference_mode():
        torch.testing.assert_close(model(**kwargs).logits, original(**kwargs).logits)
    expected_details, actual_details = {}, {}
    expected = stream_router_signals(
        original, original.model, kwargs, 2, {1, 3}, True, expected_details,
    )
    actual = stream_router_signals(model, model.model, kwargs, 2, {1, 3}, True, actual_details)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(actual_details["selected_experts"], expected_details["selected_experts"])
    torch.testing.assert_close(actual_details["expert_weights"], expected_details["expert_weights"])


def test_experts_support_latent_dimension_and_empty_input():
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

    config = NemotronHConfig(
        hidden_size=64, moe_latent_size=32, moe_intermediate_size=16,
        n_routed_experts=4,
    )
    experts = NemotronLinearExperts(config)
    original = NemotronHExperts(config)
    with torch.no_grad():
        for index in range(config.n_routed_experts):
            original.up_proj[index].copy_(experts.up_projs[index].weight)
            original.down_proj[index].copy_(experts.down_projs[index].weight)
        inputs = torch.randn(3, 32)
        indices = torch.tensor([[0, 2], [2, 3], [3, 0]])
        weights = torch.rand(3, 2)
        torch.testing.assert_close(
            experts(inputs, indices, weights), original(inputs, indices, weights),
        )
    result = experts(torch.empty(0, 32), torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2))
    assert result.shape == (0, 32)
    with pytest.raises(RuntimeError, match="4-bit"):
        validate_quantized_experts(SimpleNamespace(model=SimpleNamespace(layers=[
            SimpleNamespace(mixer=SimpleNamespace(experts=experts)),
        ])))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA bitsandbytes kernels")
def test_cuda_4bit_loads_every_expert_and_captures_native_routes(checkpoint):
    from bitsandbytes.nn import Linear4bit

    path, original = checkpoint
    model, tokenizer = load_model_and_tokenizer(path, quantization="bnb-4bit")
    assert tokenizer.padding_side == "left"
    assert not model.training and not model.config.use_cache
    assert validate_quantized_experts(model) == 16
    assert not isinstance(model.model.layers[0].mixer.out_proj, Linear4bit)
    assert not isinstance(model.model.layers[0].mixer.in_proj, Linear4bit)
    ids = torch.tensor([[1, 8, 12, 16, 9, 7, 2, 4]], device="cuda")
    details, native_ids = {}, {}
    handles = [
        model.model.layers[index].mixer.experts.register_forward_pre_hook(
            lambda module, args, index=index: native_ids.update({index: args[1].detach().cpu()})
        ) for index in (1, 3)
    ]
    try:
        routers, hidden = stream_router_signals(
            model, model.model, dict(input_ids=ids, use_cache=False), 2, {1, 3}, True, details,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert routers.shape == (2, 6, 4)
    assert hidden.shape == (2, 6, 64)
    assert torch.isfinite(routers).all() and torch.isfinite(hidden).all()
    for row, index in enumerate((1, 3)):
        torch.testing.assert_close(
            details["selected_experts"][row].sort(-1).values,
            native_ids[index][2:].sort(-1).values,
        )


def test_prequantized_input_is_rejected(checkpoint):
    path, model = checkpoint
    model.config.quantization_config = {"quant_method": "modelopt"}
    with pytest.raises(ValueError, match="BF16 checkpoint"):
        load_nemotron_4bit(path, model.config)

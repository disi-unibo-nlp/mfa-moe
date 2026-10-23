"""Verify Gemma expert math and real fused-checkpoint NF4 loading."""

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models
from transformers import Gemma4Config, Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerFast

from moe_exp.models.gemma_quantized import (
    Gemma4ForQuantizedReplay, load_gemma_4bit, validate_quantized_experts,
)
from moe_exp.models.loader import load_model_and_tokenizer
from moe_exp.models.router_adapters import stream_router_signals


@pytest.fixture
def original():
    torch.manual_seed(41)
    config = Gemma4TextConfig(
        vocab_size=128, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, num_experts=4, top_k_experts=2,
        enable_moe_block=True, moe_intermediate_size=64, global_head_dim=16,
        num_global_key_value_heads=2, hidden_size_per_layer_input=0,
        num_kv_shared_layers=0, sliding_window=8,
        layer_types=["sliding_attention", "full_attention"], pad_token_id=0,
        experts_implementation="eager",
    )
    model = Gemma4ForCausalLM(config).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.router.per_expert_scale.copy_(torch.tensor([0.5, 1., 2., 3.]))
    return model


def write_checkpoint(path, model, multimodal):
    weights = {}
    for name, value in model.state_dict().items():
        if name == "lm_head.weight":
            continue  # Official checkpoint uses tied word embeddings.
        if multimodal:
            name = name.replace("model.", "model.language_model.", 1)
        weights[name] = value.clone().contiguous()
    config = model.config
    if multimodal:
        config = Gemma4Config(text_config=config.to_dict(), vision_config=None, audio_config=None)
        # Unused modality tensors must be ignored without hiding text-loading errors.
        weights["model.vision_tower.test.weight"] = torch.ones(2, 2)
        weights["model.embed_vision.test.weight"] = torch.ones(2, 2)
    config.save_pretrained(path)
    save_file(weights, path / "model.safetensors")
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]", eos_token="[EOS]",
    ).save_pretrained(path)


def test_split_expert_math_and_native_router_capture(original):
    model = Gemma4ForQuantizedReplay(original.config).eval()
    weights = {}
    for name, value in original.state_dict().items():
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            for index, tensor in enumerate(value):
                weights[f"{name}s.{index}.weight"] = tensor
        else:
            weights[name] = value
    model.load_state_dict(weights, strict=True)
    inputs = dict(input_ids=torch.tensor([[1, 9, 8, 6, 4, 2]]), use_cache=False)
    with torch.inference_mode():
        torch.testing.assert_close(model(**inputs).logits, original(**inputs).logits)
    expected_details, actual_details = {}, {}
    expected = stream_router_signals(original, original.model, inputs, 2, {0, 1}, True, expected_details)
    actual = stream_router_signals(model, model.model, inputs, 2, {0, 1}, True, actual_details)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)
    for key in ("selected_experts", "expert_weights"):
        torch.testing.assert_close(expected_details[key], actual_details[key])
    experts = model.model.layers[0].experts
    assert experts(torch.empty(0, 64), torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2)).shape == (0, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA bitsandbytes kernels")
@pytest.mark.parametrize("multimodal", [False, True])
def test_cuda_fused_checkpoint_loads_all_experts_and_preserves_router_hooks(tmp_path, original, multimodal):
    from bitsandbytes.nn import Linear4bit

    write_checkpoint(tmp_path, original, multimodal)
    model, tokenizer = load_model_and_tokenizer(tmp_path, quantization="bnb-4bit")
    assert validate_quantized_experts(model) == 16
    assert model.config.model_type == "gemma4_text"
    assert tokenizer.padding_side == "left" and not model.training
    assert not hasattr(model.model, "vision_tower")
    assert model.lm_head.weight is model.model.embed_tokens.weight
    for layer in model.model.layers:
        assert not isinstance(layer.router.proj, Linear4bit)
    # Compare every packed expert against the source, catching missing/duplicated splits.
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit
    for index, layer in enumerate(model.model.layers):
        for projection in ("gate_up_proj", "down_proj"):
            source = getattr(original.model.layers[index].experts, projection)
            for expert, linear in enumerate(getattr(layer.experts, projection + "s")):
                packed, state = quantize_4bit(
                    source[expert].to(device="cuda", dtype=torch.bfloat16),
                    quant_type="nf4", compress_statistics=True,
                )
                torch.testing.assert_close(
                    dequantize_4bit(linear.weight, linear.weight.quant_state), dequantize_4bit(packed, state),
                )
    captured = {}
    handle = model.model.layers[1].router.register_forward_hook(
        lambda module, args, output: captured.update(ids=output[2].detach().cpu(), weights=output[1].detach().cpu())
    )
    details = {}
    try:
        routers, hidden = stream_router_signals(
            model, model.model,
            dict(input_ids=torch.tensor([[1, 9, 8, 6, 4, 2]], device="cuda"), use_cache=False),
            2, {1}, True, details,
        )
    finally:
        handle.remove()
    order = captured["weights"][2:].argsort(dim=-1, descending=True, stable=True)
    torch.testing.assert_close(details["selected_experts"][0], captured["ids"][2:].gather(-1, order))
    torch.testing.assert_close(details["expert_weights"][0], captured["weights"][2:].gather(-1, order))
    assert routers.shape == (1, 4, 4) and hidden.shape == (1, 4, 64)
    assert torch.isfinite(routers).all() and torch.isfinite(hidden).all()


def test_rejects_native_nvfp4_and_dense_models(original):
    original.config.quantization_config = {"quant_method": "modelopt"}
    with pytest.raises(ValueError, match="unquantized checkpoint"):
        load_gemma_4bit("unused", original.config)
    del original.config.quantization_config
    original.config.enable_moe_block = False
    with pytest.raises(ValueError, match="MoE checkpoint"):
        load_gemma_4bit("unused", original.config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA bitsandbytes kernels")
def test_missing_expert_checkpoint_fails_instead_of_running_random_weights(tmp_path, original):
    write_checkpoint(tmp_path, original, False)
    filename = tmp_path / "model.safetensors"
    weights = load_file(filename)
    del weights["model.layers.0.experts.down_proj"]
    save_file(weights, filename)
    with pytest.raises(RuntimeError, match="Gemma replay checkpoint missing_keys"):
        load_model_and_tokenizer(tmp_path, quantization="bnb-4bit")

"""Small native Qwen fixtures for expert math, loading, and precision enforcement."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models
from transformers import (
    PreTrainedTokenizerFast, Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig,
)

from moe_exp.models.loader import load_model_and_tokenizer
from moe_exp.models.qwen_quantized import (
    QwenForQuantizedReplay, SplitExperts, load_qwen_quantized, validate_quantized_experts,
)
from moe_exp.models.router_adapters import stream_router_signals


@pytest.fixture
def original():
    torch.manual_seed(42)
    config = Qwen3_5MoeTextConfig(
        vocab_size=128, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1]},
        pad_token_id=0, experts_implementation="eager",
    )
    return Qwen3_5MoeForCausalLM(config).eval()


def write_checkpoint(path, original, multimodal):
    path.mkdir(parents=True, exist_ok=True)
    weights = {}
    for name, value in original.state_dict().items():
        if multimodal and name.startswith("model."):
            name = name.replace("model.", "model.language_model.", 1)
        weights[name] = value.clone().contiguous()
    config = original.config
    if multimodal:
        config = Qwen3_5MoeConfig(text_config=config.to_dict())
        weights["model.visual.test.weight"] = torch.ones(2, 2)
    config.save_pretrained(path)
    save_file(weights, path / "model.safetensors")
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]", eos_token="[EOS]",
    ).save_pretrained(path)


def test_split_experts_preserve_native_forward_and_router_logits(original):
    model = QwenForQuantizedReplay(original.config).eval()
    weights = {}
    for name, value in original.state_dict().items():
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            names = [f"{name}s.{i}.weight" for i in range(4)]
            weights.update(SplitExperts().convert({name: [value]}, names))
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
    captured = {}
    handle = original.model.layers[0].mlp.gate.register_forward_hook(
        lambda module, args, output: captured.update(logits=output[0].detach())
    )
    try:
        with torch.inference_mode():
            original.model(**inputs)
    finally:
        handle.remove()
    # Independently compare probabilities to the native router output; negative
    # logits catch the old erroneous clamp+log probability interpretation.
    raw = captured["logits"][2:].float()
    assert (raw < 0).any()
    torch.testing.assert_close(actual[0][0].softmax(-1), raw.softmax(-1))


def test_split_rejects_incorrect_expert_count():
    with pytest.raises(ValueError, match="unexpected shape"):
        SplitExperts().convert({"x": [torch.ones(3, 4, 5)]}, ["a", "b"])


@pytest.mark.parametrize("bits", [4, 8])
def test_reject_prequantized_source(original, bits):
    original.config.quantization_config = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="unquantized checkpoint"):
        load_qwen_quantized("unused", original.config, bits=bits)


@pytest.mark.parametrize("quantization", ["bnb-4bit", "bnb-8bit", "unsloth-4bit"])
def test_loader_rejects_cpu_precision_fallback(quantization):
    with pytest.raises(ValueError, match="requires a CUDA"):
        load_model_and_tokenizer("unused", device="cpu", quantization=quantization)


@pytest.mark.parametrize("quantization", ["bnb-4bit", "bnb-8bit"])
def test_loader_rejects_incompatible_native_quantization(monkeypatch, quantization):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(model_type="qwen3_moe",
                                                       quantization_config={"quant_method": "fp8"}))
    with pytest.raises(ValueError, match="refusing to ignore"):
        load_model_and_tokenizer("unused", quantization=quantization)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA bitsandbytes kernels")
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("multimodal", [False, True])
def test_cuda_checkpoint_loads_every_expert_and_captures_routes(tmp_path, original, bits, multimodal):
    write_checkpoint(tmp_path, original, multimodal)
    model, tokenizer = load_model_and_tokenizer(str(tmp_path), quantization=f"bnb-{bits}bit")
    assert validate_quantized_experts(model, bits) == 16
    assert model._replay_quantization["bits"] == bits
    assert tokenizer.padding_side == "left"
    captured = {}
    handle = model.model.layers[1].mlp.gate.register_forward_hook(
        lambda module, args, output: captured.update(
            logits=output[0].detach().cpu(), weights=output[1].detach().cpu(), ids=output[2].detach().cpu(),
        )
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
    torch.testing.assert_close(routers[0].softmax(-1), captured["logits"][2:].float().softmax(-1))
    assert hidden.shape == (1, 4, 64) and torch.isfinite(hidden).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA bitsandbytes kernels")
def test_cuda_missing_expert_weights_are_rejected(tmp_path, original):
    from safetensors.torch import load_file
    write_checkpoint(tmp_path, original, False)
    filename = tmp_path / "model.safetensors"
    weights = load_file(filename)
    del weights["model.layers.0.mlp.experts.down_proj"]
    save_file(weights, filename)
    with pytest.raises(RuntimeError, match="missing_keys"):
        load_model_and_tokenizer(str(tmp_path), quantization="bnb-4bit")

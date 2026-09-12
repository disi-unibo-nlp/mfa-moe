"""Memory-safe extraction must preserve the existing full-context signals."""

from types import SimpleNamespace

import pytest
import torch

from moe_exp.models.inference import _generated_tokens_to_cpu, extract_logs_single_pass


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT"

    def __call__(self, text, **kwargs):
        encoding = {"input_ids": torch.tensor([[ord(c) % 31 for c in text]])}
        if kwargs.get("return_offsets_mapping"):
            encoding["offset_mapping"] = torch.tensor([[(i, i + 1) for i in range(len(text))]])
        return encoding


@pytest.fixture
def tiny_qwen(monkeypatch):
    modeling = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    # Exercise the same PyTorch linear-attention fallback as the reported OOM,
    # without downloading weights or requiring CUDA/optional fused kernels.
    for name in (
        "FusedRMSNormGated", "causal_conv1d_fn", "causal_conv1d_update",
        "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
    ):
        monkeypatch.setattr(modeling, name, None)
    config = modeling.Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        rope_parameters={
            "rope_type": "default", "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1],
        },
        # Explicit extraction flags must override memory-heavy config defaults.
        output_hidden_states=True,
        output_router_logits=True,
    )
    return modeling.Qwen3_5MoeForCausalLM(config).eval()


def _hook_counts(module):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in module.modules()]


@pytest.mark.parametrize("layer_indices", [None, [2, 0, 2], []])
@pytest.mark.parametrize("extract_hidden_states", [False, True])
@pytest.mark.parametrize("nested_language_model", [False, True])
def test_streamed_qwen_matches_hf_full_capture(
    tiny_qwen, layer_indices, extract_hidden_states, nested_language_model
):
    tokenizer = CharacterTokenizer()
    # Cross multiple linear-attention chunks and include a partial final chunk.
    cot = "a" * 131
    inputs = tokenizer("PROMPT" + cot)
    backbone = tiny_qwen.model
    with torch.inference_mode():
        reference = backbone(
            **inputs, use_cache=False, output_hidden_states=True, output_router_logits=True,
        )
    selected = [i for i in range(3) if layer_indices is None or i in layer_indices]
    expected_router = torch.stack([
        reference.router_logits[i][6:].clamp_min(torch.finfo(torch.float32).tiny).log()
        for i in selected
    ]) if selected else torch.empty(0)
    expected_hidden = torch.stack([
        reference.hidden_states[i][0, 6:] for i in selected
    ]) if selected else torch.empty(0)

    if nested_language_model:
        class MultimodalBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = backbone

            def forward(self, **kwargs):
                return self.language_model(**kwargs)

        tiny_qwen.model = MultimodalBackbone()

    seen_kwargs = []
    seen_outputs = []

    def check_outputs(_module, args, output):
        # Do not retain the output object itself; verify HF did not collect
        # extra outputs or a cache, including when config defaults request them.
        seen_outputs.append((output.hidden_states, output.router_logits, output.past_key_values))

    handle = backbone.register_forward_pre_hook(
        lambda _module, args, kwargs: seen_kwargs.append(kwargs), with_kwargs=True,
    )
    output_handle = backbone.register_forward_hook(check_outputs)
    before = _hook_counts(tiny_qwen)
    try:
        result = extract_logs_single_pass(
            tiny_qwen, tokenizer, "problem", cot,
            extract_hidden_states=extract_hidden_states, layer_indices=layer_indices,
        )
        assert _hook_counts(tiny_qwen) == before
    finally:
        handle.remove()
        output_handle.remove()

    assert len(seen_kwargs) == 1
    assert seen_kwargs[0]["input_ids"].shape == (1, 137)
    for flag in ("use_cache", "output_router_logits", "output_hidden_states", "output_attentions"):
        assert seen_kwargs[0][flag] is False
    assert seen_outputs == [(None, None, None)]
    if extract_hidden_states:
        router, hidden = result
        torch.testing.assert_close(hidden, expected_hidden, rtol=0, atol=0)
        assert hidden.device.type == "cpu"
    else:
        router = result
    torch.testing.assert_close(router, expected_router, rtol=0, atol=0)
    assert router.device.type == "cpu"


def test_generated_cpu_copy_has_independent_continuation_storage():
    full = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4)
    expected = full[0, 6:].clone()
    copied = _generated_tokens_to_cpu(full, 6)
    full.zero_()
    torch.testing.assert_close(copied, expected)
    assert copied.untyped_storage().nbytes() == expected.numel() * expected.element_size()


@pytest.mark.parametrize("fail_forward", [False, True])
def test_streaming_hooks_support_keyword_inputs_and_are_removed_on_failure(fail_forward):
    class Gate(torch.nn.Module):
        def forward(self, hidden):
            probabilities = torch.softmax(hidden.reshape(-1, 4), dim=-1)
            return probabilities, probabilities[:, :2], torch.zeros((hidden.shape[1], 2))

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.gate = Gate()

        def forward(self, *, hidden_states):
            self.mlp.gate(hidden_states)
            if fail_forward:
                raise RuntimeError("simulated forward failure")
            return hidden_states + 1

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(model_type="qwen3_5_moe_text")
            self.layers = torch.nn.ModuleList([Layer(), Layer()])

        def forward(self, input_ids, **kwargs):
            hidden = torch.zeros((1, input_ids.shape[1], 4))
            for layer in self.layers:
                hidden = layer(hidden_states=hidden)

    model = Backbone()
    before = _hook_counts(model)
    if fail_forward:
        with pytest.raises(RuntimeError, match="simulated forward failure"):
            extract_logs_single_pass(model, CharacterTokenizer(), "p", "cot", True)
    else:
        router, hidden = extract_logs_single_pass(model, CharacterTokenizer(), "p", "cot", True)
        assert router.shape == (2, 3, 4)
        torch.testing.assert_close(hidden[0], torch.zeros((3, 4)))
        torch.testing.assert_close(hidden[1], torch.ones((3, 4)))
    assert _hook_counts(model) == before

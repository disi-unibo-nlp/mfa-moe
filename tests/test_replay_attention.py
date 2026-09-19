"""Numerical and memory checks for the CUDA replay backend."""
from types import SimpleNamespace

import pytest
import torch

from moe_exp.models.replay_attention import replay_attention, replay_mask, register_replay_attention


def test_replay_mask_rejects_padding_and_cache():
    with pytest.raises(ValueError, match="padding"):
        replay_mask(1, 4, 4, attention_mask=torch.tensor([[1, 1, 0, 0]]))
    with pytest.raises(ValueError, match="KV cache"):
        replay_mask(1, 1, 4)
    assert replay_mask(1, 4, 4, attention_mask=torch.ones(1, 4)) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
@pytest.mark.parametrize("head_dim", [64, 256, 512])
@pytest.mark.parametrize("window", [None, 8])
@pytest.mark.parametrize("sinks", [False, True])
def test_attention_matches_dense_reference(head_dim, window, sinks):
    torch.manual_seed(42)
    n = 137
    q = torch.randn(1, 4, n, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, n, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sink = torch.randn(4, device="cuda", dtype=torch.bfloat16) if sinks else None
    scale = head_dim ** -0.5
    actual, _ = replay_attention(SimpleNamespace(training=False), q, k, v, None,
                                 scaling=scale, sliding_window=window, s_aux=sink)
    scores = q.float() @ k.repeat_interleave(2, dim=1).float().transpose(-1, -2) * scale
    indices = torch.arange(n, device="cuda")
    mask = indices[:, None] >= indices[None, :]
    if window is not None:
        mask &= indices[:, None] - indices[None, :] < window
    scores.masked_fill_(~mask, float("-inf"))
    if sinks:
        scores = torch.cat([scores, sink.float().view(1, 4, 1, 1).expand(1, 4, n, 1)], -1)
    probs = scores.softmax(-1)[..., :n]
    expected = (probs @ v.repeat_interleave(2, dim=1).float()).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, atol=0.012, rtol=0.025)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
@pytest.mark.parametrize("family", ["gemma", "gpt"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_model_routing_matches_eager(family, dtype):
    from moe_exp.models.router_adapters import stream_router_signals
    torch.manual_seed(17)
    common = dict(vocab_size=128, hidden_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                  intermediate_size=64, pad_token_id=0, experts_implementation="eager")
    if family == "gemma":
        from transformers import Gemma4TextConfig, Gemma4ForCausalLM
        config = Gemma4TextConfig(**common, global_head_dim=32,
            num_global_key_value_heads=2, hidden_size_per_layer_input=0,
            num_kv_shared_layers=0, sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
            num_experts=4, top_k_experts=2, enable_moe_block=True, moe_intermediate_size=64)
        cls = Gemma4ForCausalLM
    else:
        from transformers import GptOssConfig, GptOssForCausalLM
        config = GptOssConfig(**common, num_local_experts=4, num_experts_per_tok=2,
                             sliding_window=8, initial_context_length=128)
        cls = GptOssForCausalLM
    config._attn_implementation = "eager"
    model = cls(config).eval().to(device="cuda", dtype=dtype)
    ids = torch.randint(1, 128, (1, 37), device="cuda")
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    old, new = {}, {}
    expected = stream_router_signals(model, model.model, kwargs, 7, None, True, old)
    model.set_attn_implementation(register_replay_attention())
    actual = stream_router_signals(model, model.model, kwargs, 7, None, True, new)
    for a, b in zip(actual, expected):
        if dtype == torch.float32:
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
        else:
            # Fused FP32 accumulation differs from eager's rounded BF16 scores.
            relative_error = (a.float() - b.float()).norm() / b.float().norm()
            assert relative_error < 0.01, relative_error
    torch.testing.assert_close(new['expert_weights'], old['expert_weights'], atol=0.015, rtol=0.03)
    # Decisions can differ only at near-ties; report the actual agreement.
    agreement = (old['selected_experts'] == new['selected_experts']).float().mean().item()
    assert agreement >= 0.98, agreement


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
def test_long_gemma_attention_has_no_quadratic_allocation():
    torch.cuda.empty_cache()
    q = torch.randn(1, 16, 8192, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 8192, 512, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out, _ = replay_attention(SimpleNamespace(training=False), q, k, k, None, scaling=512**-0.5)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - start
    assert out.shape == (1, 8192, 16, 512)
    assert extra < 1024**3, extra  # Dense BF16 scores alone would require 2 GiB.

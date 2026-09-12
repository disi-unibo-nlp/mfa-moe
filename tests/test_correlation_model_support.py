"""Exercise real, tiny HF architectures and exact replay without model downloads."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from moe_exp.correlation_pipeline.features import compute_layer_features
from moe_exp.correlation_pipeline.spans import sentence_spans, token_layout, trace_digest
from moe_exp.models.inference import extract_logs_single_pass
from moe_exp.models.router_adapters import (
    SIGMOID_ROUTERS,
    available_router_layers,
    configured_top_k,
    decoder_layers,
    model_family,
    router_module,
)
from moe_exp.models.token_replay import make_token_replay, validate_token_replay
from moe_exp.schemas import TraceRecord


@pytest.fixture
def byte_tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {char: i for i, char in enumerate(alphabet)}
    vocab["ab"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab, [("a", "b")]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.parametrize("raw", [
    "First λ.\n</think>\nFinal.",  # Qwen3.5/3.6, Nemotron, GLM prompt prefills <think>
    "<think>First λ.\n</think>\nFinal.",  # Qwen3
    "<|channel>thought\nFirst λ.<channel|>Final.<turn|>",  # Gemma4
    ("<|channel|>analysis<|message|>First λ.<|end|>"
     "<|start|>assistant<|channel|>final<|message|>Final.<|return|>"),
])
def test_native_chat_markers_and_unicode_align_with_exact_token_ids(byte_tokenizer, raw):
    prompt_ids = byte_tokenizer.encode("prompt<think>\n", add_special_tokens=False)
    completion_ids = byte_tokenizer.encode(raw, add_special_tokens=False)
    text, replay = make_token_replay(byte_tokenizer, prompt_ids, completion_ids)
    assert text == raw
    assert validate_token_replay(replay, byte_tokenizer, text) == (prompt_ids, completion_ids)
    trace = TraceRecord(
        dataset="math500", problem_id="p", prompt="question", gold_answer="Final.",
        model_id="test", model_answer="Final.", cot_text=text,
        metadata={"token_replay": replay, "reasoning_content": "First λ.",
                  "assistant_content": "Final."},
    )
    assert [unit["text"] for unit in sentence_spans(trace)] == ["First λ."]
    layout = token_layout(trace, byte_tokenizer)
    assert layout["token_count"] == len(completion_ids)
    assert layout["reasoning_tokens"] == layout["unit_tokens"][0]
    for index in layout["reasoning_tokens"]:
        start, end = replay["completion_offsets"][index]
        assert "<" not in text[start:end]
    before = trace_digest(trace)
    replay["prompt_token_ids"] = prompt_ids + [prompt_ids[-1]]
    assert trace_digest(trace) != before


@pytest.mark.parametrize("ending", [
    "", "<|end|><|start|>assistant<|channel|>final<|message|>Answer.<|return|>",
])
def test_multiple_harmony_analysis_messages(byte_tokenizer, ending):
    raw = ("<|channel|>analysis<|message|>First."
           "<|end|><|start|>assistant<|channel|>analysis<|message|>Second. " + ending)
    text, replay = make_token_replay(
        byte_tokenizer, [0], byte_tokenizer.encode(raw, add_special_tokens=False),
    )
    trace = TraceRecord(
        dataset="math500", problem_id="p", prompt="question", gold_answer="Answer.",
        model_id="test", model_answer="Answer.", cot_text=text,
        metadata={"token_replay": replay, "reasoning_content": "First.\nSecond."},
    )
    units = sentence_spans(trace)
    assert [u["text"] for u in units] == ["First.", "Second."]
    assert [u["index"] for u in units] == [0, 1]
    assert all(text[u["start"]:u["end"]] == u["text"] for u in units)
    layout = token_layout(trace, byte_tokenizer)
    expected = [i for i, (left, right) in enumerate(replay["completion_offsets"])
                if any(right > text.index(body) and left < text.index(body) + len(body)
                       for body in ("First.", "Second. "))]
    assert layout["reasoning_tokens"] == expected
    assert sorted(i for group in layout["unit_tokens"] for i in group) == expected
    trace.metadata["reasoning_content"] = "Unrelated reasoning."
    with pytest.raises(ValueError, match="does not occur"):
        sentence_spans(trace)


def test_noncanonical_bpe_tokens_are_not_reencoded(byte_tokenizer):
    ids = [byte_tokenizer.convert_tokens_to_ids(c) for c in ("a", "b")]
    assert len(byte_tokenizer.encode("ab", add_special_tokens=False)) == 1
    text, replay = make_token_replay(byte_tokenizer, [0], ids)
    assert text == "ab"
    assert replay["completion_offsets"] == [[0, 1], [1, 2]]
    seen = {}

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace()

        def forward(self, input_ids, **kwargs):
            seen["input_ids"] = input_ids.tolist()
            return SimpleNamespace(router_logits=(torch.zeros(input_ids.shape[1], 4),))

    result = extract_logs_single_pass(Backbone(), byte_tokenizer, "ignored", text, token_replay=replay)
    assert seen["input_ids"] == [[0] + ids]
    assert result.shape == (1, 2, 4)
    with pytest.raises(ValueError, match="do not decode"):
        validate_token_replay(replay, byte_tokenizer, "different")
    replay["tokenizer_sha256"] = "wrong"
    with pytest.raises(ValueError, match="different vocabularies"):
        validate_token_replay(replay, byte_tokenizer, text)


def tiny_model(family):
    common = {
        "vocab_size": 512, "hidden_size": 32, "num_hidden_layers": 3,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
        "intermediate_size": 32, "pad_token_id": 0,
    }
    if family == "qwen3_moe":
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        return Qwen3MoeForCausalLM(Qwen3MoeConfig(
            **common, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        )).eval()
    if family == "gpt_oss":
        from transformers import GptOssConfig, GptOssForCausalLM
        return GptOssForCausalLM(GptOssConfig(
            **common, num_local_experts=4, num_experts_per_tok=2,
            sliding_window=8, initial_context_length=32,
        )).eval()
    if family == "glm4_moe_lite":
        from transformers import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM
        common.pop("head_dim")  # GLM aliases head_dim to qk_rope_head_dim.
        return Glm4MoeLiteForCausalLM(Glm4MoeLiteConfig(
            **common, n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
            moe_intermediate_size=16, first_k_dense_replace=1, n_group=1, topk_group=1,
            q_lora_rank=16, kv_lora_rank=8, qk_rope_head_dim=8, qk_nope_head_dim=8,
            v_head_dim=8,
        )).eval()
    if family == "nemotron_h":
        from transformers import NemotronHConfig, NemotronHForCausalLM
        common["num_hidden_layers"] = 4
        return NemotronHForCausalLM(NemotronHConfig(
            **common, layers_block_type=["mamba", "moe", "attention", "moe"],
            n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
            moe_intermediate_size=16, moe_shared_expert_intermediate_size=16,
            n_group=1, topk_group=1, mamba_num_heads=4, mamba_head_dim=8,
            n_groups=1, ssm_state_size=4, chunk_size=8, use_mamba_kernels=False,
        )).eval()
    if family == "gemma4_text":
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig
        return Gemma4ForCausalLM(Gemma4TextConfig(
            **common, num_experts=4, top_k_experts=2, enable_moe_block=True,
            moe_intermediate_size=16, global_head_dim=16, num_global_key_value_heads=1,
            hidden_size_per_layer_input=0, num_kv_shared_layers=0, sliding_window=8,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        )).eval()
    raise AssertionError(family)


@pytest.mark.parametrize("family", [
    "qwen3_moe", "gpt_oss", "glm4_moe_lite", "nemotron_h", "gemma4_text",
])
def test_real_tiny_architectures_capture_native_routes_and_decoder_inputs(family, byte_tokenizer):
    model = tiny_model(family)
    assert model_family(model) == family
    assert configured_top_k(model.config) == 2
    layers = decoder_layers(model)
    indices = available_router_layers(model)
    if family == "glm4_moe_lite":
        assert indices == [1, 2]  # The dense first block must not shift layer labels.
    if family == "nemotron_h":
        assert indices == [1, 3]  # Preserve the hybrid model's absolute indices.
    prompt_ids = byte_tokenizer.encode("Prompt", add_special_tokens=False)
    completion_ids = byte_tokenizer.encode("First. Second.", add_special_tokens=False)
    text, replay = make_token_replay(byte_tokenizer, prompt_ids, completion_ids)
    ids = torch.tensor([prompt_ids + completion_ids])
    reference_router, reference_hidden, reference_experts = {}, {}, {}
    handles = []

    def gate_hook(index):
        def hook(module, args, output):
            reference_router[index] = (output[0] if isinstance(output, tuple) else output).detach()
            if isinstance(output, tuple):
                reference_experts[index] = output[2].detach()
        return hook

    def experts_hook(index):
        def hook(module, args):
            reference_experts[index] = args[1].detach()
        return hook

    def hidden_hook(index):
        def hook(module, args, kwargs):
            reference_hidden[index] = (args[0] if args else kwargs["hidden_states"]).detach()
        return hook

    for index in indices:
        gate = router_module(layers[index], family)
        if family in SIGMOID_ROUTERS:
            # Selection must follow the native correction bias, not raw top-k.
            with torch.no_grad():
                gate.weight.zero_()
                gate.e_score_correction_bias.copy_(torch.tensor([10., 10., 0., 0.]))
            block = getattr(layers[index], "mixer" if family == "nemotron_h" else "mlp")
            handles.append(block.experts.register_forward_pre_hook(experts_hook(index)))
        handles.append(gate.register_forward_hook(gate_hook(index)))
        handles.append(layers[index].register_forward_pre_hook(hidden_hook(index), with_kwargs=True))
    with torch.inference_mode():
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for handle in handles:
        handle.remove()
    details = {}
    # Nonconsecutive selection tests that returned tensor rows retain real layer IDs.
    selected_layers = indices[-1:]
    router, hidden = extract_logs_single_pass(
        model, byte_tokenizer, "unused", text, True, layer_indices=selected_layers,
        token_replay=replay, routing_details=details,
    )
    index = selected_layers[0]
    raw = reference_router[index].reshape(1, -1, 4)[0, len(prompt_ids):].float()
    if family in SIGMOID_ROUTERS:
        expected = raw.sigmoid() / raw.sigmoid().sum(-1, keepdim=True)
        assert set(details["selected_experts"].flatten().tolist()) == {0, 1}
    elif family == "gemma4_text":
        expected = raw
    else:
        expected = raw.softmax(-1)
    torch.testing.assert_close(router[0].softmax(-1), expected)
    torch.testing.assert_close(hidden[0], reference_hidden[index][0, len(prompt_ids):])
    actual_set = details["selected_experts"][0].sort(-1).values
    expected_set = reference_experts[index].reshape(1, -1, 2)[0, len(prompt_ids):].sort(-1).values
    torch.testing.assert_close(actual_set, expected_set)
    assert details["layer_indices"] == selected_layers
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_confidence_features_use_actual_selected_set_even_with_selection_bias():
    probabilities = torch.tensor([[[.7, .2, .08, .02]]])
    values = compute_layer_features(
        probabilities.log(), None, torch.tensor([[[2, 3]]]), max_geometry_tokens=3,
    )
    assert values["router_selected_mass_l00"] == pytest.approx(.1)
    assert values["router_boundary_margin_l00"] == pytest.approx(.02 - .7)


def test_generation_persists_exact_ids_and_reuses_them_on_resume(byte_tokenizer, monkeypatch, tmp_path):
    from moe_exp.correlation_pipeline import generate
    from moe_exp.correlation_pipeline.benchmarks import BenchmarkSpec
    from moe_exp.correlation_pipeline.client import Completion
    from moe_exp.jsonl import iter_jsonl

    spec = BenchmarkSpec(
        name="synthetic", source="test", split="test", default_samples=1,
        answer_type="choice", loader=lambda limit: [
            {"problem_id": "1", "prompt": "Choose.", "gold_answer": "B",
             "options": ["wrong", "right"]},
        ],
    )
    monkeypatch.setitem(generate.BENCHMARKS, "synthetic", spec)
    monkeypatch.setattr(generate, "_replay_tokenizer", lambda _: byte_tokenizer)
    raw = "<|channel|>analysis<|message|>Reason.<|end|>" \
          "<|start|>assistant<|channel|>final<|message|>\\boxed{B}<|return|>"
    ids = byte_tokenizer.encode(raw, add_special_tokens=False)
    prompt_ids = byte_tokenizer.encode("<|start|>assistant", add_special_tokens=False)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return Completion(
            text="<think>Reason.</think>\\boxed{B}", content="\\boxed{B}",
            reasoning_content="Reason.", finish_reason="stop", usage={},
            prompt_token_ids=prompt_ids, token_ids=ids,
        )

    monkeypatch.setattr(generate, "generate_completion", completion)
    args = generate.build_parser().parse_args([
        "--model", "openai/gpt-oss-20b", "--target-model-id", "openai/gpt-oss-20b",
        "--draft-model-id", "", "--save-token-ids", "--output-dir", str(tmp_path),
    ])
    for _ in range(2):
        summary = generate.generate_dataset("synthetic", args)
        assert summary["accuracy"] == 1.0
    assert len(calls) == 1
    assert calls[0]["return_token_ids"] is True
    row = next(iter(iter_jsonl(tmp_path / "openai--gpt-oss-20b/synthetic/traces.jsonl")))
    assert row["cot_text"] == raw
    assert row["metadata"]["token_replay"]["completion_token_ids"] == ids
    assert row["metadata"]["token_replay"]["prompt_token_ids"] == prompt_ids
    assert row["metadata"]["scoring_input"] == "assistant_content"


def test_client_requests_and_preserves_vllm_token_ids(monkeypatch):
    from moe_exp.correlation_pipeline.client import generate_completion

    def response(url, key, payload, timeout):
        assert payload["return_token_ids"] is True
        return {"prompt_token_ids": [10, 11], "choices": [{
            "message": {"content": "Answer", "reasoning": "Think"},
            "token_ids": [12, 13], "finish_reason": "stop",
        }]}

    monkeypatch.setattr("moe_exp.correlation_pipeline.client._post_json", response)
    result = generate_completion(
        base_url="http://localhost/v1", api_key="test", model="test", messages=[],
        max_tokens=20, temperature=.6, top_p=.95, top_k=0, seed=1, return_token_ids=True,
    )
    assert result.prompt_token_ids == [10, 11]
    assert result.token_ids == [12, 13]


@pytest.mark.parametrize("prompt,completion", [(None, [1]), ([1], None), ([True], [1]), ([1], [])])
def test_missing_or_invalid_server_token_ids_are_rejected(byte_tokenizer, prompt, completion):
    with pytest.raises(ValueError, match="requires non-empty"):
        make_token_replay(byte_tokenizer, prompt, completion)

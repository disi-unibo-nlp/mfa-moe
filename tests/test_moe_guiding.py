from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from moe_exp.moe_guiding import plugin
from moe_exp.moe_guiding import run as run_module
from moe_exp.moe_guiding.config import ARCHITECTURE, RoutingConfig, parse_layers
from moe_exp.moe_guiding.integration import (
    routing_construction,
    routing_diagnostics,
    validate_vllm_config,
)
from moe_exp.moe_guiding.routing import MarginRouter, margin_route


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("condition", ["baseline", "selected", "transfer"])
def test_per_token_routing_and_both_weighting_conditions(condition, renormalize):
    probabilities = torch.tensor([[0.50, 0.25, 0.15, 0.10], [0.40, 0.25, 0.20, 0.15]])
    router = MarginRouter(RoutingConfig(condition=condition))
    weights, ids = router(torch.empty(2, 1), probabilities.log(), 2, renormalize)
    expected_ids = [[0, 1], [0, 1]] if condition == "baseline" else [[0, 3], [0, 1]]
    expected_weights = torch.tensor([[0.5, 0.1 if condition == "selected" else 0.25], [0.4, 0.25]])
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    assert ids.tolist() == expected_ids
    torch.testing.assert_close(weights, expected_weights)
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    assert ids.is_contiguous() and weights.is_contiguous()
    assert router.snapshot()["interventions"] == (0 if condition == "baseline" else 1)


def test_strict_threshold_and_disabled_extremes():
    logits = torch.tensor([[0.5, 0.25, 0.15, 0.10]]).log()
    # Use the same representable margin so this checks strict > without decimal rounding noise.
    margin = float(2 * logits.softmax(-1).topk(2).values.sum() - 1)
    hidden = torch.empty(1, 1)
    for threshold in (margin, 1.0):
        router = MarginRouter(RoutingConfig(threshold=threshold))
        assert router(hidden, logits, 2, True)[1].tolist() == [[0, 1]]
    router = MarginRouter(RoutingConfig(threshold=margin - 1e-4))
    assert router(hidden, logits, 2, True)[1].tolist() == [[0, 3]]


def test_ties_never_duplicate_top1_and_extreme_logits_remain_finite():
    router = MarginRouter(RoutingConfig(threshold=-1))
    logits = torch.zeros(8, 5)
    _, ids = router(torch.empty(8, 1), logits, 2, True)
    assert torch.all(ids[:, 0] != ids[:, 1])
    weights, ids = margin_route(
        torch.empty(1, 1), torch.tensor([[10000.0, 0.0, -10000.0]]), 2, True
    )
    assert ids.tolist() == [[0, 2]]
    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_noncontiguous_logits_and_empty_batches(dtype):
    logits = torch.tensor([[4, 3, 2, 1], [1, 2, 3, 4]], dtype=dtype).T.contiguous().T
    weights, ids = margin_route(torch.empty(2, 1), logits, 2, True)
    assert weights.dtype == torch.float32
    assert weights.shape == ids.shape == (2, 2)
    assert weights.is_contiguous() and ids.is_contiguous()
    weights, ids = margin_route(torch.empty(0, 1), logits[:0], 2, True)
    assert weights.shape == ids.shape == (0, 2)


@pytest.mark.parametrize("shape,topk", [((2, 2), 2), ((2, 4), 3), ((4,), 2), ((2, 2, 4), 2)])
def test_invalid_routing_shapes_are_rejected(shape, topk):
    with pytest.raises(ValueError, match="top-2"):
        margin_route(torch.empty(2, 1), torch.zeros(shape), topk, True)


def test_counters_can_exclude_warmup():
    router = MarginRouter(RoutingConfig())
    logits = torch.tensor([[0.5, 0.25, 0.15, 0.1]]).log()
    for _ in range(2):
        router(torch.empty(1, 1), logits, 2, True)
    assert router.snapshot() == {
        "calls": 2,
        "token_evaluations": 2,
        "interventions": 2,
        "intervention_rate": 1.0,
    }
    router.reset()
    assert router.snapshot()["token_evaluations"] == 0
    assert router.snapshot()["intervention_rate"] is None


@pytest.mark.parametrize("value", ["", "-1", "1,1", "12,", "one", "1.5"])
def test_invalid_layer_arguments(value):
    with pytest.raises(ValueError):
        parse_layers(value)


def test_config_round_trip_and_layer_bounds():
    assert parse_layers("all") is None
    assert parse_layers("12, 0") == (0, 12)
    config = RoutingConfig(layers=parse_layers("all"))
    assert config.selected_layers(3) == (0, 1, 2)
    assert RoutingConfig(**config.to_dict()) == config
    assert RoutingConfig(**RoutingConfig().to_dict()) == RoutingConfig()
    with pytest.raises(ValueError, match="outside"):
        RoutingConfig().selected_layers(12)
    for threshold in (float("nan"), float("inf"), -1.01, 1.01):
        with pytest.raises(ValueError):
            RoutingConfig(threshold=threshold)


@pytest.mark.parametrize("constructor_name", ["FusedMoEFactory", "FusedMoE"])
def test_injection_happens_during_construction_and_only_at_selected_layers(constructor_name):
    module = ModuleType("fake_mixtral")
    callbacks = []

    def original(num_experts, top_k, prefix="", custom_routing_function=None):
        callbacks.append(custom_routing_function)
        if custom_routing_function is not None:
            # Simulate a backend capturing and invoking the callback at initialization.
            logits = torch.tensor([[0.5, 0.25, 0.15, 0.1]]).log()
            return custom_routing_function(torch.empty(1, 1), logits, top_k, True)
        return None

    setattr(module, constructor_name, original)
    with routing_construction(module, RoutingConfig(), (12,)) as routers:
        constructor = getattr(module, constructor_name)
        assert constructor(4, 2, "model.layers.112.experts") is None
        weights, ids = constructor(4, 2, "model.layers.12.experts")
        assert ids.tolist() == [[0, 3]]
        assert list(routers) == [12]
    assert getattr(module, constructor_name) is original
    assert callbacks[0] is None and isinstance(callbacks[1], MarginRouter)


def test_native_baseline_and_factory_restoration_on_failure():
    module = ModuleType("fake_mixtral")

    def original(prefix="", custom_routing_function=None):
        assert custom_routing_function is None

    module.FusedMoEFactory = original
    with routing_construction(module, RoutingConfig(condition="baseline"), (12,)) as routers:
        module.FusedMoEFactory(prefix="layers.12.experts")
    assert routers == {}
    with pytest.raises(RuntimeError, match="selected layers"):
        with routing_construction(module, RoutingConfig(), (12,)):
            module.FusedMoEFactory(prefix="layers.112.experts")
    assert module.FusedMoEFactory is original
    with pytest.raises(RuntimeError, match="construction failed"):
        with routing_construction(module, RoutingConfig(), (12,)):
            raise RuntimeError("construction failed")
    assert module.FusedMoEFactory is original


def fake_vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="mixtral",
                num_experts_per_tok=2,
                num_local_experts=8,
                num_hidden_layers=32,
            ),
            dtype=torch.bfloat16,
            enforce_eager=True,
        ),
        quant_config=None,
        kernel_config=SimpleNamespace(moe_backend="triton"),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=False,
            enable_eplb=False,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
    )


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("hf", "model_type", "qwen3_moe"),
        ("hf", "num_experts_per_tok", 8),
        ("hf", "quantization_config", {}),
        ("model_config", "enforce_eager", False),
        ("model_config", "dtype", torch.float32),
        ("kernel_config", "moe_backend", "auto"),
        ("parallel_config", "enable_expert_parallel", True),
        ("parallel_config", "enable_eplb", True),
        ("parallel_config", "pipeline_parallel_size", 2),
    ],
)
def test_unsupported_model_or_backend_fails_before_loading(target, key, value):
    vllm_config = fake_vllm_config()
    assert validate_vllm_config(vllm_config, RoutingConfig()) == (12,)
    obj = vllm_config.model_config.hf_config if target == "hf" else getattr(vllm_config, target)
    setattr(obj, key, value)
    with pytest.raises(ValueError):
        validate_vllm_config(vllm_config, RoutingConfig())


def test_plugin_registration_is_lazy_idempotent_and_packaged(monkeypatch):
    registered = {}
    fake = ModuleType("vllm")
    fake.ModelRegistry = SimpleNamespace(
        get_supported_archs=lambda: list(registered),
        register_model=lambda name, target: registered.setdefault(name, target),
    )
    monkeypatch.setitem(sys.modules, "vllm", fake)
    plugin.register()
    plugin.register()
    assert registered == {ARCHITECTURE: "moe_exp.moe_guiding.model:MoEGuidingMixtralForCausalLM"}
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["entry-points"]["vllm.general_plugins"]["moe_guiding"] == (
        "moe_exp.moe_guiding.plugin:register"
    )


def test_prompt_validation_and_unicode_line_separators(tmp_path):
    path = tmp_path / "prompts.jsonl"
    row = {"id": "example", "prompt": "first\u2028second"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert run_module.load_prompts(path) == [row]
    path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        run_module.load_prompts(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        run_module.load_prompts(path)


@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("chat", [False, True])
def test_generation_records_condition_and_detects_callback_bypass(
    tmp_path, monkeypatch, bypass, chat
):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"one","prompt":"Calculate 2+2."}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    args = run_module.build_parser().parse_args(
        [
            "generate",
            "--model",
            "test-mixtral",
            "--prompts",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--condition",
            "transfer",
            "--layers",
            "0",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "2048",
        ]
    )

    args.chat = chat

    class FakeLLM:
        def __init__(self, **kwargs):
            assert kwargs["moe_backend"] == "triton" and kwargs["enforce_eager"]
            assert kwargs["hf_overrides"]["architectures"] == [ARCHITECTURE]
            assert kwargs["max_num_seqs"] == 1
            assert kwargs["max_num_batched_tokens"] == 2048
            config = RoutingConfig(**kwargs["additional_config"]["moe_guiding"])
            self.router = MarginRouter(config)
            model = SimpleNamespace(
                moe_guiding_routers={0: self.router},
                moe_guiding_config=config,
                moe_guiding_layers=(0,),
            )
            self.worker = SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda: model))

        def collective_rpc(self, method, kwargs=None):
            return [method(self.worker, **(kwargs or {}))]

        def get_tokenizer(self):
            def encode(prompt, add_special_tokens):
                assert prompt == "<s>[INST]Calculate 2+2.[/INST]"
                assert add_special_tokens is False
                return [1, 2]

            return SimpleNamespace(
                apply_chat_template=lambda *args, **kwargs: "<s>[INST]Calculate 2+2.[/INST]",
                encode=encode,
            )

        def generate(self, prompts, sampling, use_tqdm):
            assert prompts == ([{"prompt_token_ids": [1, 2]}] if chat else ["Calculate 2+2."])
            if not bypass:
                self.router(
                    torch.empty(1, 1), torch.tensor([[0.5, 0.25, 0.15, 0.1]]).log(), 2, True
                )
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(text="4", token_ids=[3], finish_reason="stop")],
                )
            ]

    fake = ModuleType("vllm")
    fake.LLM = FakeLLM
    fake.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(
        run_module, "entry_points", lambda **kwargs: [SimpleNamespace(name="moe_guiding")]
    )
    monkeypatch.setattr(run_module, "version", lambda name: "test-version")
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    if bypass:
        with pytest.raises(RuntimeError, match="did not execute"):
            run_module.generate(args)
    else:
        assert run_module.generate(args) == output_dir
        row = json.loads((output_dir / "generations.jsonl").read_text(encoding="utf-8"))
        assert row["condition"] == "transfer" and row["text"] == "4"
        with pytest.raises(FileExistsError):
            run_module.generate(args)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == ("failed" if bypass else "complete")
    assert manifest["routing"]["condition"] == "transfer"
    assert manifest["engine_args"]["max_num_seqs"] == 1
    assert manifest["engine_args"]["max_num_batched_tokens"] == 2048
    assert manifest["routing_diagnostics"][0]["layers"]["0"]["token_evaluations"] == int(not bypass)


def test_worker_diagnostics_reject_missing_adapter():
    worker = SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda: object()))
    with pytest.raises(RuntimeError, match="adapter"):
        routing_diagnostics(worker)


def test_sanity_examples():
    assert len(run_module.sanity_check()) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_callback_matches_cpu():
    cpu = run_module.sanity_check("cpu")
    cuda = run_module.sanity_check("cuda")
    for expected, actual in zip(cpu, cuda, strict=True):
        assert actual["expert_ids"] == expected["expert_ids"]
        torch.testing.assert_close(
            torch.tensor(actual["weights"]), torch.tensor(expected["weights"])
        )

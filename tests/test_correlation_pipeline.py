from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from moe_exp.models import loader as loader_module
from moe_exp.correlation_pipeline import generate as generation_module
from moe_exp.correlation_pipeline.analyze import (
    _repeated_problem_analysis,
    extract_trace_features,
)
from moe_exp.correlation_pipeline.analyze import build_parser as build_analysis_parser
from moe_exp.correlation_pipeline.benchmarks import (
    BENCHMARKS,
    BenchmarkSpec,
    _normalize_math_rows,
    format_user_prompt,
    sample_variant,
)
from moe_exp.correlation_pipeline.client import Completion, generate_completion
from moe_exp.correlation_pipeline.defaults import (
    DEFAULT_FORWARD_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MTP_MODEL,
)
from moe_exp.correlation_pipeline.extract import (
    build_parser as build_extract_parser,
    load_probe_layers,
)
from moe_exp.correlation_pipeline.scoring import score_completion
from moe_exp.experiment2.run import process_file
from moe_exp.models.inference import _find_prompt_length, _format_prompt, extract_logs_single_pass
from moe_exp.models.loader import _is_conditional_generation_config, load_model_and_tokenizer
from moe_exp.schemas import ModelLogs, TraceRecord


def test_spiral_benchmark_defaults_and_prompts() -> None:
    assert BENCHMARKS["aime24"].default_samples == 32
    assert BENCHMARKS["aime25"].default_samples == 32
    assert BENCHMARKS["amc23"].default_samples == 32
    assert BENCHMARKS["math500"].default_samples == 1
    assert BENCHMARKS["gpqa_diamond"].default_samples == 10
    assert set(BENCHMARKS) >= {
        "math500",
        "aime24",
        "aime25",
        "olympiad",
        "amc23",
        "minerva",
        "gpqa_diamond",
        "mmlu_pro",
        "gsm8k",
        "math",
        "prm800k",
        "processbench",
    }
    math_prompt = format_user_prompt({"prompt": "Compute 1+1."})
    choice_prompt = format_user_prompt({"prompt": "Choose.", "options": ["one", "two"]})
    assert "\\boxed{}" in math_prompt
    assert "\\boxed{LETTER}" in choice_prompt
    assert "A) one" in choice_prompt and "B) two" in choice_prompt


def test_pipeline_defaults_to_unsloth_qwen35_mtp_pair() -> None:
    assert DEFAULT_GENERATION_MODEL == "qwen3.5-35b-a3b-mtp-ud-q4-k-xl"
    assert DEFAULT_FORWARD_MODEL == "unsloth/Qwen3.5-35B-A3B"
    assert DEFAULT_MTP_MODEL == "unsloth/Qwen3.5-35B-A3B-MTP-GGUF:UD-Q4_K_XL"
    generation_args = generation_module.build_parser().parse_args([])
    assert generation_args.model == DEFAULT_GENERATION_MODEL
    assert generation_args.target_model_id == DEFAULT_FORWARD_MODEL
    assert generation_args.draft_model_id == DEFAULT_MTP_MODEL
    extraction_args = build_extract_parser().parse_args([])
    assert extraction_args.model_id == DEFAULT_FORWARD_MODEL
    assert extraction_args.generation_model == DEFAULT_GENERATION_MODEL
    assert extraction_args.quantization == "unsloth-4bit"
    assert build_analysis_parser().parse_args([]).model_id == DEFAULT_FORWARD_MODEL


def test_qwen36_uses_conditional_generation_loader() -> None:
    config = SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"])
    assert _is_conditional_generation_config(config) is True
    assert _is_conditional_generation_config(SimpleNamespace(architectures=["LlamaForCausalLM"])) is False


def test_unsloth_loader_quantizes_with_fast_model(monkeypatch, tmp_path: Path) -> None:
    calls = {}
    monkeypatch.delenv("UNSLOTH_DISABLE_STATISTICS", raising=False)
    monkeypatch.delenv("UNSLOTH_DISABLE_VENDORED_FLA", raising=False)
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising=False)
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    for variable in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        loader_module.importlib,
        "import_module",
        lambda name: calls.setdefault("preimport", name),
    )
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    model.eval = lambda: calls.setdefault("eval", True)
    tokenizer = SimpleNamespace(padding_side="right", pad_token_id=None, eos_token_id=42)

    class FastModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["kwargs"] = kwargs
            return model, tokenizer

    unsloth_module = ModuleType("unsloth")
    unsloth_module.FastModel = FastModel
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_module)

    loaded_model, loaded_tokenizer = load_model_and_tokenizer(
        "unsloth/Qwen3.5-35B-A3B",
        quantization="unsloth-4bit",
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert calls["kwargs"]["load_in_4bit"] is True
    assert calls["kwargs"]["load_in_8bit"] is False
    assert calls["kwargs"]["full_finetuning"] is False
    assert calls["kwargs"]["text_only"] is True
    assert calls["preimport"] == "torch._inductor.async_compile"
    assert os.environ["UNSLOTH_DISABLE_STATISTICS"] == "1"
    assert os.environ["UNSLOTH_DISABLE_VENDORED_FLA"] == "1"
    assert os.environ["UNSLOTH_COMPILE_DISABLE"] == "1"
    assert os.environ["USER"] == f"uid-{os.getuid()}"
    assert Path(os.environ["UNSLOTH_COMPILE_LOCATION"]) == tmp_path / "unsloth_compiled_cache"
    assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) == tmp_path / "torchinductor"
    assert (tmp_path / "unsloth_compiled_cache").is_dir()
    assert (tmp_path / "torchinductor").is_dir()
    assert model.config.use_cache is False
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token_id == 42


def test_single_pass_avoids_full_vocabulary_logits() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return "PROMPT"

        def __call__(self, text, *, return_tensors, return_offsets_mapping=False):
            payload = {"input_ids": torch.arange(len(text)).unsqueeze(0)}
            if return_offsets_mapping:
                payload["offset_mapping"] = torch.tensor(
                    [[(index, index + 1) for index in range(len(text))]]
                )
            return payload

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(model_type="qwen3_5_moe_text")
            self.logits_to_keep = None

        def forward(
            self,
            *,
            input_ids,
            use_cache,
            output_router_logits,
            output_hidden_states,
            return_dict,
            logits_to_keep,
        ):
            self.logits_to_keep = logits_to_keep
            token_count = input_ids.shape[1]
            probabilities = torch.tensor([0.7, 0.2, 0.1]).expand(token_count, -1)
            router = tuple(probabilities for _ in range(2))
            hidden = tuple(torch.zeros(1, token_count, 4) for _ in range(3))
            return SimpleNamespace(router_logits=router, hidden_states=hidden)

    model = Model()
    router, hidden = extract_logs_single_pass(
        model,
        Tokenizer(),
        problem="ignored",
        cot_text="answer",
        extract_hidden_states=True,
    )

    assert model.logits_to_keep == 1
    assert router.shape == (2, 6, 3)
    assert hidden.shape == (2, 6, 4)
    expected_probabilities = torch.tensor([0.7, 0.2, 0.1]).expand_as(router)
    assert torch.allclose(router.softmax(dim=-1), expected_probabilities)

    selected_router, selected_hidden = extract_logs_single_pass(
        model,
        Tokenizer(),
        problem="ignored",
        cot_text="answer",
        extract_hidden_states=True,
        layer_indices=[1],
    )
    assert selected_router.shape == (1, 6, 3)
    assert selected_hidden.shape == (1, 6, 4)


def test_single_pass_bypasses_causal_lm_auxiliary_router_loss() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return "PROMPT"

        def __call__(self, text, *, return_tensors, return_offsets_mapping=False):
            payload = {"input_ids": torch.arange(len(text)).unsqueeze(0)}
            if return_offsets_mapping:
                payload["offset_mapping"] = torch.tensor(
                    [[(index, index + 1) for index in range(len(text))]]
                )
            return payload

    class Backbone(torch.nn.Module):
        def forward(self, *, input_ids, **kwargs):
            token_count = input_ids.shape[1]
            router = tuple(torch.zeros(token_count, 3) for _ in range(2))
            hidden = tuple(torch.zeros(1, token_count, 4) for _ in range(3))
            return SimpleNamespace(router_logits=router, hidden_states=hidden)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(model_type="qwen3_5_moe_text")
            self.model = Backbone()

        def forward(self, **kwargs):
            raise AssertionError("CausalLM wrapper must not run during extraction")

    router, hidden = extract_logs_single_pass(
        Model(),
        Tokenizer(),
        problem="ignored",
        cot_text="answer",
        extract_hidden_states=True,
    )

    assert router.shape == (2, 6, 3)
    assert hidden.shape == (2, 6, 4)


def test_dflash_launcher_uses_native_llamacpp_draft_backend() -> None:
    script = (
        Path(__file__).parents[1]
        / "src"
        / "common"
        / "llamacpp"
        / "serve_qwen3_6_35b_a3b_dflash.sh"
    )
    environment = os.environ.copy()
    environment.update({"HOME": "/tmp", "DRY_RUN": "true"})
    result = subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    command = result.stdout
    assert "--spec-type draft-dflash" in command
    assert "--spec-draft-n-max 6" in command
    assert "--model-draft /models/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf" in command
    assert "qwen3.6-35b-a3b-dflash" in command
    script_text = script.read_text(encoding="utf-8")
    assert "--continue-at -" in script_text
    assert 'DRAFT_MODEL_REPO="${DRAFT_MODEL_REPO:-Alittlehammmer/' in script_text
    assert 'MODEL_REPO="${MODEL_REPO:-unsloth/' in script_text


def test_qwen35_launcher_uses_integrated_native_mtp() -> None:
    script = (
        Path(__file__).parents[1]
        / "src"
        / "common"
        / "llamacpp"
        / "serve_qwen3_5_35b_a3b_mtp.sh"
    )
    environment = os.environ.copy()
    environment.update({"HOME": "/tmp", "DRY_RUN": "true"})
    result = subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    command = result.stdout
    assert "--spec-type draft-mtp" in command
    assert "--spec-draft-n-max 6" in command
    assert "--model-draft" not in command
    assert "--parallel 1" in command
    assert "/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf" in command
    script_text = script.read_text(encoding="utf-8")
    assert "--continue-at -" in script_text
    assert 'MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-35B-A3B-MTP-GGUF}"' in script_text


def test_correlation_docker_launcher_routes_stages() -> None:
    script = (
        Path(__file__).parents[1]
        / "src"
        / "moe_exp"
        / "correlation_pipeline"
        / "run_docker.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert "--network host" in text
    assert 'MODULE="moe_exp.correlation_pipeline.generate"' in text
    assert 'MODULE="moe_exp.correlation_pipeline.extract"' in text
    assert 'MODULE="moe_exp.correlation_pipeline.analyze"' in text
    assert 'PYTHONPATH=/workspace/src' in text


def test_one_command_orchestrator_orders_server_and_stages() -> None:
    script = (
        Path(__file__).parents[1]
        / "src"
        / "moe_exp"
        / "correlation_pipeline"
        / "run_all.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert "DATASETS=(math500 aime24 minerva)" in text
    assert "Waiting for llama.cpp health" in text
    assert text.index('STAGE_LAUNCHER\" generate') < text.index('STAGE_LAUNCHER\" forward')
    assert text.index('STAGE_LAUNCHER\" forward') < text.index('STAGE_LAUNCHER\" analyze')
    assert "docker stop --timeout 30" in text
    assert "Removing stale stopped container" in text
    assert "SERVER_PID=$!" in text
    assert 'kill -0 "$SERVER_PID"' in text
    assert "trap cleanup EXIT INT TERM" in text
    assert "serve_qwen3_5_35b_a3b_mtp.sh" in text
    assert 'QUANTIZATION="${QUANTIZATION:-unsloth-4bit}"' in text


def test_gpqa_attempts_permute_choices_and_recompute_gold_letter() -> None:
    example = {
        "problem_id": "gpqa_diamond_record",
        "prompt": "Question",
        "gold_answer": "A",
        "options": ["correct", "wrong 1", "wrong 2", "wrong 3"],
        "_canonical_options": ["correct", "wrong 1", "wrong 2", "wrong 3"],
        "metadata": {},
    }
    variants = [sample_variant("gpqa_diamond", example, sample_id) for sample_id in range(10)]
    assert len({tuple(variant["options"]) for variant in variants}) > 1
    assert all(
        variant["options"]["ABCD".index(variant["gold_answer"])] == "correct"
        for variant in variants
    )


def test_math_row_normalization_preserves_source_metadata() -> None:
    rows = [
        {
            "unique_id": "test/algebra/1.json",
            "problem": "Solve x=1.",
            "answer": "1",
            "subject": "Algebra",
            "level": 1,
        }
    ]
    normalized = _normalize_math_rows(rows, dataset="math500")
    assert normalized[0]["problem_id"] == "math500_test/algebra/1.json"
    assert normalized[0]["metadata"]["source_id"] == "test/algebra/1.json"


def test_llamacpp_client_preserves_reasoning_content(monkeypatch) -> None:
    def fake_post(url, api_key, payload, timeout):
        assert url.endswith("/chat/completions")
        assert payload["temperature"] == 0.6
        return {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "First reason.",
                        "content": "\\boxed{B}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 4},
        }

    monkeypatch.setattr("moe_exp.correlation_pipeline.client._post_json", fake_post)
    completion = generate_completion(
        base_url="http://localhost:8080/v1",
        api_key="key",
        model="test-model",
        messages=[{"role": "user", "content": "Question"}],
        max_tokens=16,
        temperature=0.6,
        top_p=0.95,
        top_k=0,
        seed=42,
    )
    assert completion.text == "<think>\nFirst reason.\n</think>\n\\boxed{B}"
    assert completion.reasoning_content == "First reason."


def test_multiple_choice_scoring_requires_boxed_letter() -> None:
    answer, correct, method = score_completion(
        {"gold_answer": "C"},
        answer_type="choice",
        model_text="Reasoning. Therefore \\boxed{C}.",
    )
    assert (answer, correct, method) == ("C", True, "exact_multiple_choice")
    _, incorrect, _ = score_completion(
        {"gold_answer": "C"},
        answer_type="choice",
        model_text="Reasoning without the required final format.",
    )
    assert incorrect is False


def test_exact_generation_messages_are_replayed() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages == [{"role": "user", "content": "Exact prompt"}]
            return "FORMATTED"

    prompt = _format_prompt(
        Tokenizer(),
        "raw problem",
        messages=[{"role": "user", "content": "Exact prompt"}],
    )
    assert prompt == "FORMATTED"


def test_prompt_boundary_uses_offsets_for_boundary_merged_token() -> None:
    class Tokenizer:
        def __call__(self, text, *, return_tensors, return_offsets_mapping=False):
            if return_offsets_mapping:
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "offset_mapping": torch.tensor([[[0, 1], [1, 3]]]),
                }
            return {"input_ids": torch.tensor([[1, 9]])}

    assert _find_prompt_length(Tokenizer(), "AB", "ABC") == 1


def test_generation_resumes_from_atomic_sample_shards(tmp_path, monkeypatch) -> None:
    calls = 0

    def fake_completion(**kwargs):
        nonlocal calls
        calls += 1
        return Completion(
            text="Reasoning. \\boxed{B}",
            content="Reasoning. \\boxed{B}",
            reasoning_content="",
            usage={"completion_tokens": 3},
            finish_reason="stop",
        )

    spec = BenchmarkSpec(
        name="synthetic",
        source="unit-test",
        split="test",
        default_samples=2,
        answer_type="choice",
        loader=lambda limit: [
            {
                "problem_id": "problem/1",
                "prompt": "Choose.",
                "gold_answer": "B",
                "options": ["wrong", "right"],
                "metadata": {},
            }
        ],
    )
    monkeypatch.setitem(generation_module.BENCHMARKS, "synthetic", spec)
    monkeypatch.setattr(generation_module, "generate_completion", fake_completion)
    args = SimpleNamespace(
        max_items=None,
        samples_per_problem=None,
        output_dir=tmp_path,
        model="unit/model",
        target_model_id="hf/target",
        draft_model_id="hf/draft",
        base_url="http://localhost:8080/v1",
        api_key="key",
        max_tokens=32,
        temperature=0.6,
        top_p=0.95,
        top_k=0,
        seed=42,
        timeout=10,
        max_retries=0,
        workers=1,
    )
    first = generation_module.generate_dataset("synthetic", args)
    assert first["traces"] == 2
    assert first["accuracy"] == 1.0
    assert calls == 2
    second = generation_module.generate_dataset("synthetic", args)
    assert second["traces"] == 2
    assert calls == 2


def test_tensor_features_include_router_hidden_and_geometry(tmp_path) -> None:
    router = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [0.0, 2.0, 1.0], [2.0, 0.0, 1.0]],
            [[0.0, 1.0, 2.0], [0.0, 2.0, 1.0], [2.0, 1.0, 0.0], [1.0, 0.0, 2.0]],
        ]
    )
    hidden = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5) + 1
    selected = torch.topk(router, k=2, dim=-1).indices.to(torch.int16)
    router_path = tmp_path / "router.pt"
    hidden_path = tmp_path / "hidden.pt"
    selected_path = tmp_path / "selected.pt"
    torch.save(router, router_path)
    torch.save(hidden, hidden_path)
    torch.save(selected, selected_path)
    trace = TraceRecord(
        dataset="synthetic",
        problem_id="problem__sample_00",
        source_problem_id="problem",
        prompt="Prompt",
        gold_answer="1",
        model_id="model",
        model_answer="1",
        is_correct=True,
        cot_text="One. Two.",
        steps=["One.", "Two."],
        model_logs=ModelLogs(
            router_logits=router_path.as_posix(),
            hidden_states=hidden_path.as_posix(),
            selected_experts=selected_path.as_posix(),
        ),
    )
    input_path = tmp_path / "traces_with_routing.jsonl"
    input_path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    row = extract_trace_features(trace, input_path=input_path, max_geometry_tokens=4)
    assert row["token_count"] == 4
    assert np.isfinite(row["router_entropy_l00"])
    assert np.isfinite(row["router_topk_overlap_l01"])
    assert np.isfinite(row["hidden_step_distance_l00"])
    assert np.isfinite(row["hidden_router_geometry_l01"])
    assert np.isfinite(row["hidden_router_geometry_mean_layers"])


def test_forward_extraction_reuses_content_addressed_tensor_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    trace = TraceRecord(
        dataset="synthetic",
        problem_id="problem__sample_00",
        source_problem_id="problem",
        prompt="Prompt",
        generation_messages=[{"role": "user", "content": "Exact prompt"}],
        gold_answer="1",
        model_id="generator",
        model_answer="1",
        is_correct=True,
        cot_text="Reasoning. \\boxed{1}",
        steps=["Reasoning.", "\\boxed{1}"],
    )
    input_path = tmp_path / "traces.jsonl"
    input_path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    output_path = tmp_path / "forward" / "traces_with_routing.jsonl"
    calls = 0

    def fake_extract(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["messages"] == trace.generation_messages
        router = torch.randn(2, 4, 3)
        hidden = torch.randn(2, 4, 5)
        return router, hidden

    monkeypatch.setattr("moe_exp.experiment2.run.extract_logs_single_pass", fake_extract)
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(num_experts_per_tok=2),
        )
    )
    tokenizer = object()
    for _ in range(2):
        process_file(
            input_path=input_path,
            model_id="hf/model",
            output_path=output_path,
            top_k=None,
            extract_hidden_states=True,
            model=model,
            tokenizer=tokenizer,
        )
    assert calls == 1
    assert output_path.is_file()
    assert len(list((output_path.parent / "tensors").glob("*_extraction.json"))) == 1


def test_probe_layers_are_loaded_as_a_unique_router_compatible_union(tmp_path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "best_by_target": {
                    "Read": {"layer_idx": 34},
                    "Plan": {"layer_idx": 39},
                    "Explore": {"layer_idx": 40},
                    "Monitor": {"layer_idx": 34},
                }
            }
        ),
        encoding="utf-8",
    )
    assert load_probe_layers(path, 40) == ([34, 39], [40])


def test_forward_extraction_saves_only_requested_layers(tmp_path, monkeypatch) -> None:
    trace = TraceRecord(
        dataset="synthetic",
        problem_id="problem",
        prompt="Prompt",
        gold_answer="1",
        model_id="generator",
        model_answer="1",
        cot_text="Reasoning.",
    )
    input_path = tmp_path / "traces.jsonl"
    input_path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    output_path = tmp_path / "forward" / "traces_with_routing.jsonl"

    def fake_extract(**kwargs):
        router = torch.arange(4 * 3 * 2).reshape(4, 3, 2).to(torch.float32)
        hidden = torch.arange(4 * 3 * 5).reshape(4, 3, 5).to(torch.float32)
        return router, hidden

    monkeypatch.setattr("moe_exp.experiment2.run.extract_logs_single_pass", fake_extract)
    model = SimpleNamespace(config=SimpleNamespace(num_experts_per_tok=1))
    process_file(
        input_path=input_path,
        model_id="hf/model",
        output_path=output_path,
        extract_hidden_states=True,
        layer_indices=[1, 3],
        save_expert_weights=False,
        model=model,
        tokenizer=object(),
    )

    saved = TraceRecord(**json.loads(output_path.read_text(encoding="utf-8")))
    assert saved.model_logs.layer_indices == [1, 3]
    assert saved.model_logs.expert_weights is None
    router = torch.load(saved.model_logs.router_logits, weights_only=True)
    hidden = torch.load(saved.model_logs.hidden_states, weights_only=True)
    assert router.shape == (2, 3, 2)
    assert hidden.shape == (2, 3, 5)
    assert not list((output_path.parent / "tensors").glob("*_weights.pt"))


def test_correlation_forward_rejects_wrong_top_k(tmp_path) -> None:
    trace = TraceRecord(
        dataset="synthetic",
        problem_id="problem",
        prompt="Prompt",
        gold_answer="1",
        model_id="generator",
        model_answer="1",
        cot_text="Answer.",
    )
    input_path = tmp_path / "traces.jsonl"
    input_path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    model = SimpleNamespace(config=SimpleNamespace(num_experts_per_tok=4))
    with pytest.raises(ValueError, match="differs from model config"):
        process_file(
            input_path=input_path,
            model_id="hf/model",
            output_path=tmp_path / "output.jsonl",
            top_k=2,
            model=model,
            tokenizer=object(),
            strict_top_k=True,
        )


def test_repeated_problem_analysis_uses_mixed_outcome_contrasts() -> None:
    rows = []
    for problem in range(5):
        for sample, outcome in enumerate((0, 1, 0, 1)):
            rows.append(
                {
                    "dataset": "aime24",
                    "source_problem_id": f"p{problem}",
                    "sample_id": sample,
                    "is_correct": outcome,
                    "feature": float(problem + outcome * 2 + sample / 10),
                }
            )
    result = _repeated_problem_analysis(
        pd.DataFrame(rows),
        feature_columns=["feature"],
        bootstrap_samples=30,
        rng=np.random.default_rng(42),
    )
    assert result["within_problem"][0]["n_mixed_outcome_problems"] == 5
    assert result["within_problem"][0]["mean_correct_minus_incorrect"] > 1.0


def test_gpqa_is_excluded_from_within_problem_correctness_contrasts() -> None:
    rows = [
        {
            "dataset": "gpqa_diamond",
            "source_problem_id": "question",
            "sample_id": sample,
            "is_correct": sample % 2,
            "feature": float(sample),
        }
        for sample in range(10)
    ]
    result = _repeated_problem_analysis(
        pd.DataFrame(rows),
        feature_columns=["feature"],
        bootstrap_samples=30,
        rng=np.random.default_rng(42),
    )
    assert result["within_problem"] == []
    assert "gpqa_diamond" in result["within_problem_exclusions"]

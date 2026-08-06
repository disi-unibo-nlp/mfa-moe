from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from moe_exp.probeTest.data import EPISODE_LABELS, load_gold_responses
from moe_exp.probeTest.extract import (
    PreparedResponse,
    extract_boundary_activations,
    prepare_response,
    token_containing_char,
)
from moe_exp.probeTest.probe import train_binary_probe, train_layerwise_probes


class _FakeTokenizer:
    is_fast = True

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        assert messages[-1]["role"] == "user"
        return [100, 101]

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        matches = list(re.finditer(r"\S+", text))
        payload = {"input_ids": list(range(10, 10 + len(matches)))}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return payload


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(20, 4)
        self.logits_to_keep = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        output_hidden_states,
        use_cache,
        return_dict,
        logits_to_keep,
    ):
        assert attention_mask.shape == input_ids.shape
        assert output_hidden_states is True
        assert use_cache is False
        assert return_dict is True
        self.logits_to_keep = logits_to_keep
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32).view(1, -1, 1)
        hidden_states = tuple(positions.expand(-1, -1, 4) + layer * 100 for layer in range(3))
        return SimpleNamespace(hidden_states=hidden_states)


def _write_synthetic_gold(root: Path) -> None:
    originals_dir = root / "responses_original"
    labels_dir = root / "responses_labeled"
    originals_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    originals = [
        {
            "Question ID": "abc",
            "Instruction": "Compute something.",
            "deepseek-reasoner (response)": "Alpha sentence.\n\nBeta sentence.",
        }
    ]
    (originals_dir / "SAT_deepseekR1_results.json").write_text(
        json.dumps(originals), encoding="utf-8"
    )
    labeled = {
        "Question ID": "abc",
        "data": [
            {"text": "Alpha sentence.", "gt-class-1": "General", "gt-class-2": "Read"},
            {"text": "Beta sentence.", "gt-class-1": "General", "gt-class-2": "Analyze"},
        ],
    }
    (labels_dir / "1.json").write_text(json.dumps(labeled), encoding="utf-8")


def test_gold_units_align_to_original_response(tmp_path: Path) -> None:
    _write_synthetic_gold(tmp_path)
    responses = load_gold_responses(tmp_path)
    assert len(responses) == 1
    assert [(unit.char_start, unit.char_end) for unit in responses[0].units] == [
        (0, 15),
        (17, 31),
    ]


def test_think_boundary_artifact_is_excluded_by_default(tmp_path: Path) -> None:
    _write_synthetic_gold(tmp_path)
    originals_path = tmp_path / "responses_original" / "SAT_deepseekR1_results.json"
    originals = json.loads(originals_path.read_text(encoding="utf-8"))
    originals[0]["deepseek-reasoner (response)"] += "\n\n</think>"
    originals_path.write_text(json.dumps(originals), encoding="utf-8")
    labels_path = tmp_path / "responses_labeled" / "1.json"
    labeled = json.loads(labels_path.read_text(encoding="utf-8"))
    labeled["data"].append(
        {"text": "</think>", "gt-class-1": "General", "gt-class-2": "Monitor"}
    )
    labels_path.write_text(json.dumps(labeled), encoding="utf-8")

    filtered = load_gold_responses(tmp_path)
    unfiltered = load_gold_responses(tmp_path, include_think_boundary_units=True)
    assert len(filtered[0].units) == 2
    assert len(unfiltered[0].units) == 3


def test_prepare_response_uses_pre_sentence_token(tmp_path: Path) -> None:
    _write_synthetic_gold(tmp_path)
    response = load_gold_responses(tmp_path)[0]
    prepared = prepare_response(_FakeTokenizer(), response)
    assert prepared.prompt_tokens == 2
    assert prepared.response_token_indices == (0, 2)
    # Alpha is predicted by the last prompt token; Beta by the token "sentence."
    # from Alpha. No token from either target sentence is visible at its boundary.
    assert prepared.boundary_positions == (1, 3)
    assert len(prepared.input_ids) == 6


def test_token_alignment_keeps_token_with_leading_whitespace() -> None:
    offsets = [(0, 5), (5, 12), (12, 18)]
    assert token_containing_char(offsets, 6) == 1


def test_forward_selects_only_pre_unit_positions() -> None:
    prepared = PreparedResponse(
        input_ids=(1, 2, 3, 4, 5),
        boundary_positions=(1, 3),
        response_token_indices=(0, 2),
        prompt_tokens=2,
        response_tokens=3,
    )
    model = _FakeModel()
    activations = extract_boundary_activations(model, prepared)
    assert activations.shape == (2, 3, 4)
    assert activations[:, 0, 0].tolist() == [1.0, 3.0]
    assert activations[:, 2, 0].tolist() == [201.0, 203.0]
    assert model.logits_to_keep == 1


def test_binary_probe_uses_paper_configuration() -> None:
    rng = np.random.default_rng(4)
    y = np.asarray([0, 1] * 20, dtype=np.int8)
    x = rng.normal(size=(40, 5)).astype(np.float32)
    x[:, 0] += y * 5
    classifier, metrics = train_binary_probe(x[:30], y[:30], x[30:], y[30:])
    assert classifier.solver == "lbfgs"
    assert classifier.penalty == "l2"
    assert classifier.C == 1.0
    assert classifier.class_weight == "balanced"
    assert classifier.max_iter == 2000
    assert metrics["n_train_samples"] == 30
    assert metrics["n_test_samples"] == 10


def test_layerwise_probe_writes_all_seven_targets(tmp_path: Path) -> None:
    activation_dir = tmp_path / "activations"
    shard_dir = activation_dir / "shards"
    shard_dir.mkdir(parents=True)
    labels = [label for label in EPISODE_LABELS for _ in range(10)]
    activations = torch.randn(len(labels), 3, 8)
    for sample_index, label in enumerate(labels):
        activations[sample_index, :, EPISODE_LABELS.index(label)] += 6
    torch.save(activations, shard_dir / "01-synthetic.pt")
    shard = {
        "activation_file": "01-synthetic.pt",
        "response_id": "synthetic",
        "n_units": len(labels),
        "labels": labels,
        "texts": [f"unit {index}" for index in range(len(labels))],
        "char_spans": [[index, index + 1] for index in range(len(labels))],
        "boundary_token_positions": list(range(len(labels))),
    }
    manifest = {
        "status": "complete",
        "model_id": "synthetic/model",
        "model_revision": "main",
        "quantization": "none",
        "boundary_definition": "pre-unit",
        "shards": [shard],
    }
    manifest_path = activation_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    results_path = train_layerwise_probes(
        manifest_path=manifest_path,
        output_dir=tmp_path / "probes",
        make_plot=False,
    )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(results["results"]) == len(EPISODE_LABELS) * 3
    assert set(results["best_by_target"]) == set(EPISODE_LABELS)
    assert len(list((tmp_path / "probes" / "classifiers").glob("*.pkl"))) == 21
    assert results["config"]["standardization"] is False
    assert results["config"]["pca"] is False

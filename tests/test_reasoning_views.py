from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from moe_exp.correlation_pipeline.annotate import annotate_trace
from moe_exp.correlation_pipeline.features import compute_layer_features
from moe_exp.correlation_pipeline.sample_tagging import sample, select_solutions, sentence_quotas
from moe_exp.correlation_pipeline.spans import (
    position_windows,
    sentence_spans,
    token_layout,
    validate_annotation,
)
from moe_exp.correlation_pipeline.view_analysis import analyze_views, collect_view_rows
from moe_exp.correlation_pipeline.views import compute_views, position_reference
from moe_exp.models import routing_extraction
from moe_exp.schemas import TraceRecord


class CharacterTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "PROMPT"

    def __call__(self, text, **kwargs):
        result = {"input_ids": torch.tensor([[ord(c) for c in text]])}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = torch.tensor([[(i, i + 1) for i in range(len(text))]])
        return result


def trace(text="First. Second. Third.", **kwargs):
    return TraceRecord(
        dataset="math500",
        problem_id="p0",
        source_problem_id="p0",
        sample_id=0,
        prompt="Compute nine",
        gold_answer="9",
        model_id="test",
        model_answer="9",
        is_correct=True,
        cot_text=text,
        metadata={"evaluation_metric": "pass@1"},
        **kwargs,
    )


def annotated(item, tmp_path, labels=None):
    iterator = iter(labels or ["Plan"] * len(sentence_spans(item)))
    annotation = annotate_trace(item, lambda **kw: next(iterator), {"program": "frozen"}, tmp_path)
    item.metadata["reasoning_annotation"] = annotation
    return annotation


def test_sampling_from_generations_uses_all_attempts_and_resumes(tmp_path):
    generation = tmp_path / "generation"
    # Both selected attempts are correct; weights must include the other attempts.
    for dataset, outcomes in {"aime24": [True, False], "aime25": [True, True, True, False]}.items():
        directory = generation / "test" / dataset
        directory.mkdir(parents=True)
        rows = []
        for index, correct in enumerate(outcomes):
            item = trace("One. Two. Three. Four. Five. Six.")
            item.dataset = dataset
            item.problem_id = f"p0-s{index}"
            item.sample_id = index
            item.is_correct = correct
            rows.append(item.model_dump_json())
        (directory / "traces.jsonl").write_text("\n".join(rows) + "\n")
        (directory / "manifest.json").write_text(json.dumps({"samples_per_problem": len(outcomes)}))
    args = SimpleNamespace(analysis=None, generation_dir=generation, generation_model="test",
                           datasets=["aime24", "aime25"], sample_id=0, seed=42,
                           max_sentences=6, weight_by="error-rate", output_dir=tmp_path / "sampled")
    result = sample(args)
    assert result["weights"] == {"aime24": 0.5, "aime25": 0.25}
    assert result["sentence_quotas"] == {"aime24": 4, "aime25": 2}
    assert result["selected_sentences"] == 6
    for dataset, quota in result["sentence_quotas"].items():
        rows = (args.output_dir / "generation" / "test" / dataset / "traces.jsonl").read_text().splitlines()
        assert len(rows) == 1
        row = json.loads(rows[0])
        assert row["sample_id"] == 0
        assert row["cot_text"] == "One. Two. Three. Four. Five. Six."
        assert len(row["metadata"]["sentence_selection"]["indices"]) == quota
    assert sample(args) == result
    args.max_sentences = 3
    with pytest.raises(ValueError, match="different sampling plan"):
        sample(args)


def test_sentences_keep_offsets_math_decimals_and_exclude_final_answer():
    item = trace("<think>1. Use 3.14. Try $x = 2. 0$.\nVerify it!</think>\nFinal answer: 9.")
    units = sentence_spans(item)
    assert [u["text"] for u in units] == ["1. Use 3.14.", "Try $x = 2. 0$.", "Verify it!"]
    assert all(item.cot_text[u["start"] : u["end"]] == u["text"] for u in units)
    layout = token_layout(item, CharacterTokenizer())
    assert layout["token_count"] == len(item.cot_text)
    owned = [token for group in layout["unit_tokens"] for token in group]
    assert owned == layout["reasoning_tokens"]
    assert len(set(owned)) == len(owned)
    assert max(owned) < item.cot_text.index("</think>")


def test_boundary_merged_token_is_owned_once():
    class BoundaryTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            spans = (
                [(i, i + 1) for i in range(5)]
                + [(5, 7)]
                + [(i, i + 1) for i in range(7, len(text))]
            )
            return {
                "input_ids": torch.tensor([[1] * len(spans)]),
                "offset_mapping": torch.tensor([spans]),
            }

    layout = token_layout(trace("Hi. Bye."), BoundaryTokenizer())
    assert layout["token_count"] == 8
    assert layout["unit_tokens"][0][0] == 0
    assert sorted(token for group in layout["unit_tokens"] for token in group) == list(range(8))


def test_annotation_resumes_units_rejects_bad_labels_and_invalidates_program(tmp_path):
    item = trace()
    path = tmp_path / "annotation.json"
    calls = []

    def interrupted(**kwargs):
        calls.append(kwargs)
        return "Plan" if len(calls) == 1 else "not-a-class"

    with pytest.raises(ValueError, match="invalid label"):
        annotate_trace(item, interrupted, {"program": "one"}, path)
    assert len(json.loads(path.read_text())["units"]) == 1
    assert set(calls[0]) == {"problem_statement", "previous_sentence", "sentence", "next_sentence"}
    assert "9" not in str(calls[0])  # Gold answer is not exposed to the classifier.
    resumed = []
    result = annotate_trace(
        item, lambda **kw: resumed.append(kw) or "Verify", {"program": "one"}, path
    )
    assert len(resumed) == 2
    assert [u["label"] for u in result["units"]] == ["Plan", "Verify", "Verify"]
    annotate_trace(
        item, lambda **kw: pytest.fail("Completed units must be cached"), {"program": "one"}, path
    )
    resumed.clear()
    annotate_trace(item, lambda **kw: resumed.append(kw) or "Analyze", {"program": "two"}, path)
    assert len(resumed) == 3
    item.cot_text += " Changed."
    with pytest.raises(ValueError, match="Stale"):
        validate_annotation(item, result)


def test_sampled_annotations_keep_original_context_and_only_pool_selected_tokens(tmp_path):
    item = trace("<think>A. B. C.</think>answer")
    item.metadata["sentence_selection"] = {
        "schema_version": 1, "indices": [0, 2], "manifest_sha256": "fixed",
    }
    calls = []
    annotation = annotate_trace(
        item, lambda **kw: calls.append(kw) or "Plan", {"program": "frozen"},
        tmp_path / "sampled.json",
    )
    assert [unit["index"] for unit in annotation["units"]] == [0, 2]
    assert [call["sentence"] for call in calls] == ["A.", "C."]
    assert calls[0]["next_sentence"] == calls[1]["previous_sentence"] == "B."
    item.metadata["reasoning_annotation"] = annotation
    layout = token_layout(item, CharacterTokenizer())
    logits = torch.randn(1, len(item.cot_text), 3)
    views = compute_views(
        item, CharacterTokenizer(), logits, None, logits.argmax(-1).unsqueeze(-1), [0],
        modes=["full", "class", "position"],
        reference={"mean_reasoning_tokens": 6, "bins": 3}, max_geometry_tokens=8,
    )
    selected_tokens = len(layout["unit_tokens"][0]) + len(layout["unit_tokens"][2])
    assert views["reasoning_token_count"] == selected_tokens
    for mode in ["full", "class", "position"]:
        assert sum(s["token_count"] for s in views["scopes"] if s["view"] == mode) == selected_tokens
    assert "label" not in views["sentence_spans"][1]
    plan = next(s for s in views["scopes"] if s["view"] == "class" and s["name"] == "Plan")
    assert plan["transition_count"] == selected_tokens - 2
    item.metadata["sentence_selection"] = {**item.metadata["sentence_selection"], "indices": [1]}
    with pytest.raises(ValueError, match="different sentence selection"):
        validate_annotation(item, annotation)


def test_sentence_quotas_preserve_error_rate_weights_when_supply_is_limited():
    weights = {"aime24": 140 / 960, "aime25": 171 / 960, "amc23": 40 / 1280}
    capacity = {"aime24": 26254, "aime25": 30610, "amc23": 17392}
    quotas = sentence_quotas(capacity, weights, 100000)
    assert sum(quotas.values()) < 100000
    assert quotas["aime25"] == 30610
    for name, count in quotas.items():
        assert count <= capacity[name]
        assert abs(count - sum(quotas.values()) * weights[name] / sum(weights.values())) < 1
    capped = sentence_quotas(capacity, weights, 1000)
    assert sum(capped.values()) == 1000
    with pytest.raises(ValueError, match="not all zero"):
        sentence_quotas({"x": 10}, {"x": 0}, 5)


def test_solution_selection_preserves_single_attempts_and_covers_repeated_problems():
    single = trace().model_dump()
    single["sample_id"] = 7
    assert select_solutions([single], sample_id=0, repeated=False)[0].sample_id == 7
    repeated = []
    for problem in ["p0", "p1"]:
        for index in [0, 1]:
            repeated.append({**single, "source_problem_id": problem,
                             "problem_id": f"{problem}_{index}", "sample_id": index})
    chosen = select_solutions(repeated, sample_id=0, repeated=True)
    assert [row.problem_id for row in chosen] == ["p0_0", "p1_0"]
    with pytest.raises(ValueError, match="every source problem"):
        select_solutions(repeated[:-2] + repeated[-1:], sample_id=0, repeated=True)


def test_empty_sentence_sample_retains_trace_with_missing_class_metrics(tmp_path):
    item = trace()
    item.metadata["sentence_selection"] = {"schema_version": 1, "indices": []}
    annotation = annotate_trace(
        item, lambda **kw: pytest.fail("An unsampled trace must not call the judge"),
        {"program": "frozen"}, tmp_path / "empty.json",
    )
    assert annotation["status"] == "complete" and annotation["units"] == []
    item.metadata["reasoning_annotation"] = annotation
    logits = torch.randn(1, len(item.cot_text), 3)
    item.metadata["correlation_views"] = compute_views(
        item, CharacterTokenizer(), logits, None, logits.argmax(-1).unsqueeze(-1), [0],
        modes=["class"], reference=None, max_geometry_tokens=8,
    )
    rows, _ = collect_view_rows(item, {"dataset": "math500", "is_correct": 1}, ["class"])
    assert len(rows) == 7
    assert all(row["is_correct"] == 1 and np.isnan(row["token_count"]) for _, row in rows)


def test_position_windows_are_fixed_non_cumulative_and_cover_overflow():
    short = position_windows(list(range(7)), mean_length=10, bins=5)
    long = position_windows(list(range(15)), mean_length=10, bins=5)
    assert [w["tokens"] for w in short] == [[0, 1], [2, 3], [4, 5], [6], [], []]
    assert long[1]["tokens"] == [2, 3]
    assert long[-1]["tokens"] == list(range(10, 15))
    assert [token for window in long for token in window["tokens"]] == list(range(15))


def test_transition_metrics_do_not_bridge_disconnected_or_different_sentences():
    logits = torch.tensor([[[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]]])
    hidden = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    experts = logits.argmax(-1).unsqueeze(-1)
    full = compute_layer_features(logits, hidden, experts, max_geometry_tokens=8)
    assert full["router_switch_rate_l00"] == pytest.approx(1 / 3)
    subset = compute_layer_features(
        logits, hidden, experts, max_geometry_tokens=8, token_indices=[0, 1, 3]
    )
    assert subset["router_switch_rate_l00"] == 0
    assert subset["hidden_step_distance_l00"] == 0
    assert subset["router_topk_overlap_l00"] == 1
    segmented = compute_layer_features(
        logits,
        hidden,
        experts,
        max_geometry_tokens=8,
        token_indices=[0, 1, 2, 3],
        segment_ids=[0, 0, 1, 1],
    )
    assert segmented["router_switch_rate_l00"] == 0
    no_edges = compute_layer_features(
        logits, hidden, experts, max_geometry_tokens=8, token_indices=[0, 3]
    )
    assert np.isnan(no_edges["router_switch_rate_l00"])


def test_views_pool_classes_and_exclude_final_tokens(tmp_path):
    item = trace("<think>A. B. C.</think>answer")
    annotated(item, tmp_path / "ann.json", ["Plan", "Verify", "Plan"])
    logits = torch.randn(1, len(item.cot_text), 3, generator=torch.Generator().manual_seed(7))
    experts = logits.argmax(-1).unsqueeze(-1)
    result = compute_views(
        item,
        CharacterTokenizer(),
        logits,
        None,
        experts,
        [5],
        modes=["full", "class", "position"],
        reference={"mean_reasoning_tokens": 6, "bins": 3},
        max_geometry_tokens=8,
    )
    scopes = {(s["view"], s["name"]): s for s in result["scopes"]}
    assert scopes[("full", "reasoning")]["token_count"] == 8
    assert sum(s["token_count"] for s in result["scopes"] if s["view"] == "class") == 8
    assert sum(s["token_count"] for s in result["scopes"] if s["view"] == "position") == 8
    assert scopes[("class", "Read")]["values"] == {}
    assert "router_margin_l05" in scopes[("class", "Plan")]["values"]
    item.metadata["correlation_views"] = result
    rows, _ = collect_view_rows(item, {"dataset": "math500", "is_correct": 1}, ["class"])
    assert np.isnan(dict(rows)[("class", "Read")]["token_count"])
    item.metadata["reasoning_annotation"]["units"][0]["label"] = "Monitor"
    with pytest.raises(ValueError, match="do not match"):
        collect_view_rows(item, {}, ["class"])


def test_reference_uses_model_corpus_mean(tmp_path):
    items = [trace("AB"), trace("ABCDEF")]
    items[1].problem_id = "p1"
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(item.model_dump_json() + "\n" for item in items))
    reference = position_reference([path], CharacterTokenizer(), "model", bins=2)
    assert reference["mean_reasoning_tokens"] == 4
    assert reference["traces"] == 2
    assert reference["limit_applied_to_reference"] is False


def test_forward_views_checkpoint_resume_and_annotation_invalidation(tmp_path, monkeypatch):
    item = trace("A. B.")
    annotation = annotated(item, tmp_path / "ann.json", ["Plan", "Verify"])
    input_path = tmp_path / "input.jsonl"
    # Source generations do not contain the separately stored annotations.
    item.metadata.pop("reasoning_annotation")
    input_path.write_text(item.model_dump_json() + "\n")
    calls = []

    def extract(**kwargs):
        calls.append(kwargs)
        return torch.tensor([[[3.0, 1.0]] * len(item.cot_text)])

    monkeypatch.setattr(routing_extraction, "extract_logs_single_pass", extract)
    output = tmp_path / "forward" / "traces.jsonl"
    tokenizer = CharacterTokenizer()
    settings = {"version": 1}

    def run():
        routing_extraction.process_file(
            input_path,
            "test",
            output,
            model=SimpleNamespace(config=SimpleNamespace(num_experts_per_tok=1)),
            tokenizer=tokenizer,
            save_raw_tensors=False,
            save_expert_weights=False,
            feature_reducer=lambda r, h, e, layers: compute_layer_features(
                r, h, e, max_geometry_tokens=8
            ),
            feature_schema_version=2,
            view_reducer=lambda t, r, h, e, layers: compute_views(
                t,
                tokenizer,
                r,
                h,
                e,
                layers,
                modes=["full", "class"],
                reference=None,
                max_geometry_tokens=8,
            ),
            view_config=settings,
            trace_annotations={item.problem_id: annotation},
        )

    run()
    run()
    assert len(calls) == 1
    record = json.loads(output.read_text())
    assert record["metadata"]["correlation_views"]["scopes"]
    assert record["model_logs"]["router_logits"] is None
    annotation["units"][0]["label"] = "Monitor"
    run()
    assert len(calls) == 2
    settings["version"] = 2
    run()
    assert len(calls) == 3


def test_analysis_keeps_attempt_units_missingness_and_all_metric_pairs(tmp_path):
    rows = []
    for problem in range(6):
        for sample in range(2):
            missing = problem == 0 and sample == 0
            rows.append(
                {
                    "dataset": "aime24",
                    "problem_id": f"{problem}-{sample}",
                    "source_problem_id": str(problem),
                    "sample_id": sample,
                    "evaluation_metric": "avg@2",
                    "generation_sha256": f"hash-{problem}-{sample}",
                    "is_correct": (problem + sample) % 2,
                    "has_backtracking": 0,
                    "has_contradiction": 0,
                    "has_self_correction": 0,
                    "token_count": np.nan if missing else 5,
                    "selected_token_count": 0 if missing else 5,
                    "transition_count": 3,
                    "router_margin_mean_layers": np.nan if missing else problem + sample / 2,
                    "router_confidence_mean_layers": np.nan
                    if missing
                    else problem / 10 + sample / 20,
                }
            )
    result = analyze_views(
        {("position", "bin_00"): rows}, tmp_path, bootstrap_samples=5, seed=42, contract={}
    )
    path = tmp_path / "views-v1/position/bin_00/correlations.json"
    report = json.loads(path.read_text())
    assert result["views"][0]["coverage"]["traces"] == 12
    assert report["coverage"]["traces_with_tokens"] == 11
    assert all(row["feature"] != "token_count" for row in report["binary_correlations"])
    pairs = report["cross_feature_correlations"]
    assert pairs["trace_level"][0]["n_traces"] == 11
    assert pairs["repeated_problem_level"][0]["n_problems"] == 5
    # The missing window excludes a feature aggregate, not a saved attempt.
    import pandas as pd

    problems = pd.read_csv(path.parent / "problem_features.csv")
    assert problems.observed_rows.tolist() == [2] * 6
    assert problems.loc[0, "router_margin_mean_layers__n_valid"] == 1


@pytest.mark.parametrize("tagged", [True, False])
@pytest.mark.parametrize("truncated", [False, True])
def test_all_stages_roundtrip_with_a_fixed_reference_and_separate_outputs(
    tmp_path, monkeypatch, tagged, truncated
):
    from moe_exp.correlation_pipeline import analyze, annotate, extract

    generation = tmp_path / "generation/test/math500"
    generation.mkdir(parents=True)
    records = []
    for index in range(6):
        item = trace(f"<think>Plan {index}. Verify {'x' * (index + 1)}.</think>9")
        item.problem_id = item.source_problem_id = f"p{index}"
        item.is_correct = index % 2 == 0
        if truncated and index == 0:
            item.metadata["finish_reason"] = "length"
        if truncated and index == 1:
            item.metadata["finish_reason"] = "stop"
            item.metadata["usage"] = {"completion_tokens": 10}
            item.metadata["generation_config"] = {"max_tokens": 10}
        item.generation_messages = [{"role": "user", "content": "Choose A) one B) two"}]
        records.append(item)
    source = "".join(item.model_dump_json() + "\n" for item in records)
    (generation / "traces.jsonl").write_text(source)
    (generation / "manifest.json").write_text(json.dumps({"target_model_id": "test"}))
    program = tmp_path / "program.json"
    program.write_text(json.dumps({"classify": {"signature": {"instructions": "Frozen"}}}))
    modes = ["full", "class", "position"] if tagged else ["full", "position"]
    annotation_root = tmp_path / "annotations"
    annotation_args = annotate.build_parser().parse_args(
        [
            "--generation-dir",
            str(tmp_path / "generation"),
            "--generation-model",
            "test",
            "--output-dir",
            str(annotation_root),
            "--datasets",
            "math500",
            "--limit",
            "4",
            "--judge-program",
            str(program),
            "--judge-model",
            "test",
        ]
    )
    visible_questions = []

    def predict(**kwargs):
        visible_questions.append(kwargs["problem_statement"])
        return "Plan" if kwargs["sentence"].startswith("Plan") else "Verify"

    if tagged:
        annotate.annotate_all(annotation_args, predict=predict)
        assert visible_questions == ["Choose A) one B) two"] * 8
    else:
        assert not annotation_root.exists()
    tokenizer = CharacterTokenizer()
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1, num_experts_per_tok=1))
    monkeypatch.setattr(extract, "load_model_and_tokenizer", lambda *a, **kw: (model, tokenizer))
    monkeypatch.setattr(
        routing_extraction,
        "extract_logs_single_pass",
        lambda **kw: torch.randn(
            1, len(kw["cot_text"]), 2, generator=torch.Generator().manual_seed(len(kw["cot_text"]))
        ),
    )
    probes = tmp_path / "probes.json"
    probes.write_text(json.dumps({"best_by_target": {"accuracy": {"layer_idx": 0}}}))
    forward_root = tmp_path / "forward"
    forward_args = extract.build_parser().parse_args(
        [
            "--generation-dir",
            str(tmp_path / "generation"),
            "--generation-model",
            "test",
            "--model-id",
            "test",
            "--output-dir",
            str(forward_root),
            "--datasets",
            "math500",
            *(["--annotation-dir", str(annotation_root)] if tagged else []),
            "--views",
            *modes,
            "--probe-results",
            str(probes),
            "--router-only",
            "--limit",
            "4",
        ]
    )
    extract.extract_all(forward_args)
    reference = json.loads((forward_root / "test/position_reference.json").read_text())
    assert reference["traces"] == 6  # A smoke limit must not renormalize position bins.
    analysis_args = analyze.build_parser().parse_args(
        [
            "--forward-dir",
            str(forward_root),
            "--model-id",
            "test",
            "--datasets",
            "math500",
            "--output-dir",
            str(tmp_path / "analysis"),
            "--bootstrap-samples",
            "5",
            "--views",
            *modes,
            "--skip-expert-identity",
        ]
    )
    result = analyze.analyze(analysis_args)
    assert result["n_traces"] == (2 if truncated else 4)
    assert result["datasets"] == {"math500": 2 if truncated else 4}
    assert result["excluded_truncated_generations"] == {"math500": 2 if truncated else 0}
    assert result["generation_budget_audit"]["scopes"][0]["n_traces"] == 4
    import pandas as pd

    for table in (tmp_path / "analysis/test").rglob("trace_features.csv"):
        features = pd.read_csv(table)
        assert len(features) == (2 if truncated else 4)
        if truncated:
            assert set(features.source_problem_id) == {"p2", "p3"}
    assert len(result["reasoning_view_analysis"]["views"]) == (19 if tagged else 12)
    assert (tmp_path / "analysis/test/views-v1/class/Plan/correlations.json").is_file() == tagged
    assert (tmp_path / "analysis/test/views-v1/full/reasoning/correlations.json").is_file()
    assert (tmp_path / "analysis/test/views-v1/position/bin_00/correlations.json").is_file()
    if not tagged:
        assert not annotation_root.exists()
        assert not (tmp_path / "analysis/test/views-v1/class").exists()
    assert (generation / "traces.jsonl").read_text() == source


def test_frozen_dspy_program_loads_and_predicts_without_optimization(tmp_path, monkeypatch):
    import dspy

    from moe_exp.correlation_pipeline.annotate import build_parser, make_predictor
    from moe_exp.gepaLLMAsJudge import run as judge_module

    program_path = tmp_path / "selected.json"
    judge_module.EpisodeJudge("Frozen instruction for this test.").save(str(program_path))
    lm = dspy.utils.DummyLM(answers=[{"label": "Plan"}])
    monkeypatch.setattr(judge_module, "_make_lm", lambda *a, **kw: lm)
    args = build_parser().parse_args(
        ["--judge-program", str(program_path), "--judge-model", "test"]
    )
    result = make_predictor(args)(
        problem_statement="Question",
        previous_sentence="Before",
        sentence="I will calculate.",
        next_sentence="After",
    )
    assert result.label == "Plan"
    assert "Frozen instruction for this test." in str(lm.history)


def test_parallel_annotation_uses_eight_requests_and_restores_source_order(tmp_path):
    import threading

    item = trace(" ".join(f"Unit {i}." for i in range(16)))
    barrier = threading.Barrier(8, timeout=5)
    lock = threading.Lock()
    running = maximum = 0

    def predict(**kwargs):
        nonlocal running, maximum
        index = int(kwargs["sentence"].split()[1].rstrip("."))
        with lock:
            running += 1
            maximum = max(maximum, running)
        try:
            if index < 8:
                barrier.wait()
            assert kwargs["problem_statement"] == item.prompt
            return "Plan" if index % 2 == 0 else "Verify"
        finally:
            with lock:
                running -= 1

    path = tmp_path / "parallel.json"
    result = annotate_trace(item, predict, {"model": "fixed"}, path, workers=8)
    assert maximum == 8
    assert [u["index"] for u in result["units"]] == list(range(16))
    assert [u["label"] for u in result["units"]] == ["Plan", "Verify"] * 8
    validate_annotation(item, result)
    annotate_trace(
        item,
        lambda **kw: pytest.fail("Cached labels must survive worker changes"),
        {"model": "fixed"},
        path,
        workers=1,
    )


def test_parallel_annotation_resumes_sparse_out_of_order_checkpoints(tmp_path):
    item = trace()
    path = tmp_path / "sparse.json"
    saved = annotate_trace(item, lambda **kw: "Read", {"model": "fixed"}, path)
    saved["units"] = [saved["units"][2], saved["units"][0]]
    saved["status"] = "partial"
    path.write_text(json.dumps(saved))
    calls = []
    result = annotate_trace(
        item,
        lambda **kw: calls.append(kw["sentence"]) or "Plan",
        {"model": "fixed"},
        path,
        workers=8,
    )
    assert calls == ["Second."]
    assert [u["label"] for u in result["units"]] == ["Read", "Plan", "Read"]
    validate_annotation(item, result)


def test_unclosed_latex_does_not_stall_sentence_splitting():
    import signal

    def timed_out(*args):
        raise AssertionError("Sentence splitting stalled on unclosed math")

    previous = signal.signal(signal.SIGALRM, timed_out)
    try:
        signal.alarm(3)
        text = "$" + r"\alpha " * 2000 + "\nNext."
        assert [u["text"] for u in sentence_spans(trace(text))] == [
            text.split("\n")[0].strip(),
            "Next.",
        ]
        assert [u["text"] for u in sentence_spans(trace(r"Cost $a. b\$. After."))] == [
            r"Cost $a. b\$.",
            "After.",
        ]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

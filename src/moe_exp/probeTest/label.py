"""Apply trained Schoenfeld probes to benchmark reference reasoning traces."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from moe_exp.datasets.loaders import load_dataset_by_name
from moe_exp.probeTest.data import EPISODE_LABELS
from moe_exp.probeTest.extract import (
    DEFAULT_QUANTIZATION,
    PreparedResponse,
    extract_boundary_activations,
    load_model_and_tokenizer,
    prepare_text_boundaries,
)

logger = logging.getLogger(__name__)

BENCHMARK_DATASETS = ("gsm8k", "math", "prm800k", "processbench")
_UNIT_SEPARATOR = re.compile(r"(?<=[.!?])[ \t]+|(?:\r?\n)+")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ReasoningUnit:
    unit_index: int
    text: str
    char_start: int
    char_end: int
    source_step_index: int | None = None


@dataclass(frozen=True)
class BenchmarkTrace:
    dataset: str
    problem_id: str
    prompt: str
    gold_answer: str
    reasoning_text: str
    trace_source: str
    units: tuple[ReasoningUnit, ...]
    source_steps: tuple[str, ...]
    first_error_step: int | None
    solution_is_correct: bool | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProbeSuite:
    model_id: str
    model_revision: str
    quantization: str
    boundary_definition: str
    feature_size: int
    layers: dict[str, int]
    classifiers: dict[str, Any]
    validation_metrics: dict[str, dict[str, float | int]]
    results_sha256: str


def segment_reasoning_text(text: str) -> tuple[ReasoningUnit, ...]:
    """Split reasoning into exact, sentence-like spans without changing its text."""
    units: list[ReasoningUnit] = []

    def append_span(raw_start: int, raw_end: int) -> None:
        start = raw_start
        end = raw_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            return
        units.append(
            ReasoningUnit(
                unit_index=len(units),
                text=text[start:end],
                char_start=start,
                char_end=end,
            )
        )

    cursor = 0
    for match in _UNIT_SEPARATOR.finditer(text):
        append_span(cursor, match.start())
        cursor = match.end()
    append_span(cursor, len(text))
    return tuple(units)


def _join_steps(steps: Sequence[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for step in steps:
        if pieces:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(step)
        cursor += len(step)
        spans.append((start, cursor))
    return "".join(pieces), tuple(spans)


def _source_step_index(char_start: int, step_spans: Sequence[tuple[int, int]]) -> int | None:
    for step_index, (start, end) in enumerate(step_spans):
        if start <= char_start < end:
            return step_index
    return None


def build_benchmark_trace(dataset: str, example: dict[str, Any]) -> BenchmarkTrace:
    """Normalize one loader record into sentence-level reasoning units."""
    raw_steps = example.get("solution_steps") or []
    source_steps = tuple(str(step).strip() for step in raw_steps if str(step).strip())
    if source_steps:
        reasoning_text, step_spans = _join_steps(source_steps)
        trace_source = (
            "reconstructed_rated_path" if dataset == "prm800k" else "provided_solution"
        )
    else:
        reasoning_text = str(example.get("reference_solution") or "").strip()
        step_spans = ()
        trace_source = "reference_solution"
    if not reasoning_text:
        raise ValueError(f"{dataset}/{example.get('problem_id')}: no reasoning trace available")

    segmented = segment_reasoning_text(reasoning_text)
    if not segmented:
        raise ValueError(f"{dataset}/{example.get('problem_id')}: reasoning trace has no units")
    units = tuple(
        ReasoningUnit(
            unit_index=unit.unit_index,
            text=unit.text,
            char_start=unit.char_start,
            char_end=unit.char_end,
            source_step_index=_source_step_index(unit.char_start, step_spans),
        )
        for unit in segmented
    )
    return BenchmarkTrace(
        dataset=dataset,
        problem_id=str(example["problem_id"]),
        prompt=str(example["prompt"]),
        gold_answer=str(example.get("gold_answer") or ""),
        reasoning_text=reasoning_text,
        trace_source=trace_source,
        units=units,
        source_steps=source_steps,
        first_error_step=example.get("first_error_step"),
        solution_is_correct=example.get("solution_is_correct"),
        metadata=dict(example.get("metadata") or {}),
    )


def load_benchmark_traces(dataset: str, examples_per_dataset: int) -> list[BenchmarkTrace]:
    if dataset not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown benchmark {dataset!r}; choose from {BENCHMARK_DATASETS}")
    examples = load_dataset_by_name(dataset, max_items=examples_per_dataset)
    traces = [
        build_benchmark_trace(dataset, example)
        for example in examples
        if example.get("prompt")
    ]
    if len(traces) != examples_per_dataset:
        raise RuntimeError(
            f"{dataset} produced {len(traces)} usable traces; expected {examples_per_dataset}"
        )
    return traces


def load_probe_suite(results_path: Path) -> ProbeSuite:
    """Load and cross-check each target's saved best-layer classifier."""
    with results_path.open(encoding="utf-8") as handle:
        results = json.load(handle)
    config = results.get("config") or {}
    best_by_target = results.get("best_by_target") or {}
    missing_targets = [target for target in EPISODE_LABELS if target not in best_by_target]
    if missing_targets:
        raise ValueError(f"Probe results are missing targets: {', '.join(missing_targets)}")

    classifiers: dict[str, Any] = {}
    layers: dict[str, int] = {}
    validation_metrics: dict[str, dict[str, float | int]] = {}
    expected_feature_size = int(config["hidden_size"])
    for target in EPISODE_LABELS:
        best = best_by_target[target]
        layer_index = int(best["layer_idx"])
        classifier_path = (
            results_path.parent / "classifiers" / f"{target.lower()}_layer_{layer_index:02d}.pkl"
        )
        if not classifier_path.is_file():
            raise FileNotFoundError(f"Missing trained classifier: {classifier_path}")
        with classifier_path.open("rb") as handle:
            # The path points to classifiers produced locally by train_layerwise_probes.
            payload = pickle.load(handle)
        if payload.get("target") != target or int(payload.get("layer_idx", -1)) != layer_index:
            raise ValueError(f"Classifier metadata does not match {target} layer {layer_index}")
        if payload.get("model_id") != config.get("model_id"):
            raise ValueError(f"Classifier model metadata does not match {results_path}")
        if int(payload.get("feature_size", -1)) != expected_feature_size:
            raise ValueError(f"Classifier feature size does not match {results_path}")
        classifier = payload.get("classifier")
        if classifier is None or not hasattr(classifier, "predict_proba"):
            raise TypeError(f"Invalid classifier payload: {classifier_path}")
        classifiers[target] = classifier
        layers[target] = layer_index
        validation_metrics[target] = {
            "layer_idx": layer_index,
            "test_accuracy": float(best["test_accuracy"]),
            "test_f1": float(best["test_f1"]),
            "test_auc": float(best["test_auc"]),
        }

    return ProbeSuite(
        model_id=str(config["model_id"]),
        model_revision=str(config.get("model_revision") or "main"),
        quantization=str(config.get("quantization") or DEFAULT_QUANTIZATION),
        boundary_definition=str(config["boundary_definition"]),
        feature_size=expected_feature_size,
        layers=layers,
        classifiers=classifiers,
        validation_metrics=validation_metrics,
        results_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
    )


def predict_probe_labels(suite: ProbeSuite, activations: Any) -> list[dict[str, Any]]:
    """Score every unit with all seven best-layer one-vs-rest probes."""
    if activations.ndim != 3:
        raise ValueError(f"Expected (units, hidden_states, hidden_size), got {activations.shape}")
    if int(activations.shape[2]) != suite.feature_size:
        raise ValueError(
            f"Activation feature size {activations.shape[2]} != probe size {suite.feature_size}"
        )
    if max(suite.layers.values()) >= int(activations.shape[1]):
        raise ValueError("Probe layer index exceeds available hidden states")

    scores_by_target: dict[str, np.ndarray] = {}
    for target in EPISODE_LABELS:
        layer = activations[:, suite.layers[target], :].numpy()
        scores_by_target[target] = suite.classifiers[target].predict_proba(layer)[:, 1]

    predictions: list[dict[str, Any]] = []
    for unit_index in range(int(activations.shape[0])):
        scores = {
            target: float(scores_by_target[target][unit_index]) for target in EPISODE_LABELS
        }
        ranked = sorted(EPISODE_LABELS, key=lambda target: (-scores[target], target))
        predictions.append(
            {
                "predicted_label": ranked[0],
                "top_score": scores[ranked[0]],
                "score_margin": scores[ranked[0]] - scores[ranked[1]],
                "positive_probes": [target for target in EPISODE_LABELS if scores[target] >= 0.5],
                "scores": scores,
            }
        )
    return predictions


def _truncate_prepared(
    prepared: PreparedResponse,
    *,
    max_input_tokens: int,
) -> tuple[PreparedResponse, int]:
    if len(prepared.input_ids) <= max_input_tokens:
        return prepared, len(prepared.boundary_positions)
    kept_units = sum(position < max_input_tokens for position in prepared.boundary_positions)
    if kept_units == 0:
        raise ValueError(
            f"Prompt uses at least {prepared.prompt_tokens} tokens, exceeding the "
            f"{max_input_tokens}-token inspection limit"
        )
    truncated = PreparedResponse(
        input_ids=prepared.input_ids[:max_input_tokens],
        boundary_positions=prepared.boundary_positions[:kept_units],
        response_token_indices=prepared.response_token_indices[:kept_units],
        prompt_tokens=prepared.prompt_tokens,
        response_tokens=max(0, max_input_tokens - prepared.prompt_tokens),
    )
    return truncated, kept_units


def _trace_digest(trace: BenchmarkTrace) -> str:
    payload = {
        "dataset": trace.dataset,
        "problem_id": trace.problem_id,
        "prompt": trace.prompt,
        "reasoning_text": trace.reasoning_text,
        "units": [
            [unit.text, unit.char_start, unit.char_end, unit.source_step_index]
            for unit in trace.units
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_dump_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _jsonl_dump_atomic(records: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _reusable_prediction(
    path: Path,
    *,
    trace_sha256: str,
    suite: ProbeSuite,
    max_input_tokens: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    expected = {
        "trace_sha256": trace_sha256,
        "probe_results_sha256": suite.results_sha256,
        "model_id": suite.model_id,
        "model_revision": suite.model_revision,
        "quantization": suite.quantization,
        "max_input_tokens": max_input_tokens,
    }
    if all(record.get(key) == value for key, value in expected.items()):
        return record
    return None


def _prediction_record(
    trace: BenchmarkTrace,
    *,
    trace_sha256: str,
    suite: ProbeSuite,
    prepared: PreparedResponse,
    original_input_tokens: int,
    kept_units: int,
    predictions: Sequence[dict[str, Any]],
    max_input_tokens: int,
) -> dict[str, Any]:
    units: list[dict[str, Any]] = []
    for unit, boundary, prediction in zip(
        trace.units[:kept_units],
        prepared.boundary_positions,
        predictions,
        strict=True,
    ):
        units.append(
            {
                "unit_index": unit.unit_index,
                "text": unit.text,
                "char_span": [unit.char_start, unit.char_end],
                "source_step_index": unit.source_step_index,
                "at_or_after_first_error": (
                    trace.first_error_step is not None
                    and unit.source_step_index is not None
                    and unit.source_step_index >= trace.first_error_step
                ),
                "boundary_token_position": boundary,
                **prediction,
            }
        )
    return {
        "schema_version": 1,
        "dataset": trace.dataset,
        "problem_id": trace.problem_id,
        "prompt": trace.prompt,
        "gold_answer": trace.gold_answer,
        "reasoning_text": trace.reasoning_text,
        "trace_source": trace.trace_source,
        "source_steps": list(trace.source_steps),
        "first_error_step": trace.first_error_step,
        "solution_is_correct": trace.solution_is_correct,
        "metadata": trace.metadata,
        "trace_sha256": trace_sha256,
        "probe_results_sha256": suite.results_sha256,
        "model_id": suite.model_id,
        "model_revision": suite.model_revision,
        "quantization": suite.quantization,
        "boundary_definition": suite.boundary_definition,
        "probe_layers": suite.layers,
        "max_input_tokens": max_input_tokens,
        "original_input_tokens": original_input_tokens,
        "input_tokens": len(prepared.input_ids),
        "total_units": len(trace.units),
        "labeled_units": kept_units,
        "omitted_units": len(trace.units) - kept_units,
        "units": units,
    }


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _write_inspection_markdown(
    dataset: str,
    records: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        f"# {dataset.upper()} probe labels",
        "",
        (
            "The predicted label is the largest of seven independently trained, class-balanced "
            "one-vs-rest scores. Scores are useful for ranking but are not calibrated "
            "multiclass probabilities."
        ),
        "",
    ]
    for example_index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## {example_index}. {record['problem_id']}",
                "",
                f"- Trace source: `{record['trace_source']}`",
                f"- Gold answer: `{_markdown_cell(record['gold_answer'])}`",
                f"- First error step: `{record['first_error_step']}`",
                (
                    f"- Units: `{record['labeled_units']}/{record['total_units']}`; "
                    f"input tokens: `{record['input_tokens']}/{record['original_input_tokens']}`"
                ),
                "",
                "### Problem",
                "",
                record["prompt"],
                "",
                "### Probe-labelled reasoning",
                "",
                "| # | Label | Top score | Positive probes | Reasoning unit |",
                "|---:|---|---:|---|---|",
            ]
        )
        for unit in record["units"]:
            positive = ", ".join(unit["positive_probes"]) or "—"
            lines.append(
                f"| {unit['unit_index']} | {unit['predicted_label']} | "
                f"{unit['top_score']:.3f} | {_markdown_cell(positive)} | "
                f"{_markdown_cell(unit['text'])} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def _manifest_payload(
    *,
    suite: ProbeSuite,
    dataset_records: dict[str, list[dict[str, Any]]],
    examples_per_dataset: int,
    max_input_tokens: int,
) -> dict[str, Any]:
    label_counts = {label: 0 for label in EPISODE_LABELS}
    datasets: dict[str, Any] = {}
    for dataset, records in dataset_records.items():
        for record in records:
            for unit in record["units"]:
                label_counts[unit["predicted_label"]] += 1
        datasets[dataset] = {
            "examples": len(records),
            "labeled_units": sum(record["labeled_units"] for record in records),
            "omitted_units": sum(record["omitted_units"] for record in records),
            "predictions": f"{dataset}/predictions.jsonl",
            "inspection_markdown": f"{dataset}/inspection.md",
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "model_id": suite.model_id,
        "model_revision": suite.model_revision,
        "quantization": suite.quantization,
        "boundary_definition": suite.boundary_definition,
        "probe_results_sha256": suite.results_sha256,
        "probe_layers": suite.layers,
        "probe_validation_metrics": suite.validation_metrics,
        "score_interpretation": (
            "Independent class-balanced one-vs-rest scores; argmax supplies the inspection label, "
            "but scores are not calibrated multiclass probabilities."
        ),
        "examples_per_dataset": examples_per_dataset,
        "max_input_tokens": max_input_tokens,
        "datasets": datasets,
        "predicted_label_counts": label_counts,
    }


def label_benchmark_examples(
    *,
    probe_results_path: Path,
    output_dir: Path,
    datasets: Sequence[str] = BENCHMARK_DATASETS,
    examples_per_dataset: int = 20,
    max_input_tokens: int = 4096,
    trust_remote_code: bool = False,
) -> Path:
    """Label benchmark reasoning units and write resumable inspection artifacts."""
    if examples_per_dataset < 1:
        raise ValueError("examples_per_dataset must be positive")
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive")
    dataset_names = tuple(dict.fromkeys(datasets))
    unknown = [dataset for dataset in dataset_names if dataset not in BENCHMARK_DATASETS]
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    if not dataset_names:
        raise ValueError("Select at least one benchmark")

    suite = load_probe_suite(probe_results_path)
    traces_by_dataset = {
        dataset: load_benchmark_traces(dataset, examples_per_dataset)
        for dataset in dataset_names
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    model: Any | None = None
    tokenizer: Any | None = None
    dataset_records: dict[str, list[dict[str, Any]]] = {}
    for dataset, traces in traces_by_dataset.items():
        records: list[dict[str, Any]] = []
        dataset_dir = output_dir / dataset
        for example_index, trace in enumerate(traces, start=1):
            safe_id = _SAFE_FILENAME.sub("_", trace.problem_id)
            prediction_path = dataset_dir / "records" / f"{example_index:02d}-{safe_id}.json"
            trace_sha256 = _trace_digest(trace)
            reused = _reusable_prediction(
                prediction_path,
                trace_sha256=trace_sha256,
                suite=suite,
                max_input_tokens=max_input_tokens,
            )
            if reused is not None:
                logger.info(
                    "[%s %d/%d] Reusing %s",
                    dataset,
                    example_index,
                    len(traces),
                    trace.problem_id,
                )
                records.append(reused)
                continue

            if model is None:
                model, tokenizer = load_model_and_tokenizer(
                    suite.model_id,
                    revision=suite.model_revision,
                    quantization=suite.quantization,
                    trust_remote_code=trust_remote_code,
                    offload_dir=output_dir / "offload",
                )
            prepared_full = prepare_text_boundaries(
                tokenizer,
                instruction=trace.prompt,
                response_text=trace.reasoning_text,
                unit_char_starts=[unit.char_start for unit in trace.units],
                response_id=f"{dataset}/{trace.problem_id}",
            )
            original_input_tokens = len(prepared_full.input_ids)
            prepared, kept_units = _truncate_prepared(
                prepared_full,
                max_input_tokens=max_input_tokens,
            )
            logger.info(
                "[%s %d/%d] Forward %s: %d tokens, %d/%d units",
                dataset,
                example_index,
                len(traces),
                trace.problem_id,
                len(prepared.input_ids),
                kept_units,
                len(trace.units),
            )
            activations = extract_boundary_activations(model, prepared)
            predictions = predict_probe_labels(suite, activations)
            record = _prediction_record(
                trace,
                trace_sha256=trace_sha256,
                suite=suite,
                prepared=prepared,
                original_input_tokens=original_input_tokens,
                kept_units=kept_units,
                predictions=predictions,
                max_input_tokens=max_input_tokens,
            )
            _json_dump_atomic(record, prediction_path)
            records.append(record)
            del activations

        _jsonl_dump_atomic(records, dataset_dir / "predictions.jsonl")
        _write_inspection_markdown(dataset, records, dataset_dir / "inspection.md")
        dataset_records[dataset] = records

    manifest = _manifest_payload(
        suite=suite,
        dataset_records=dataset_records,
        examples_per_dataset=examples_per_dataset,
        max_input_tokens=max_input_tokens,
    )
    manifest_path = output_dir / "manifest.json"
    _json_dump_atomic(manifest, manifest_path)
    return manifest_path

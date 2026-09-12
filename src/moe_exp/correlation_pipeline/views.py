"""Compact full-reasoning, class and absolute-position feature views."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

from moe_exp.correlation_pipeline.features import compute_layer_features, json_safe_features
from moe_exp.correlation_pipeline.spans import (
    SENTENCE_LABELS,
    SPAN_SCHEMA_VERSION,
    digest,
    position_windows,
    selected_sentence_indices,
    token_layout,
    trace_digest,
    validate_annotation,
)
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord


def position_reference(
    inputs: list[Path], tokenizer: Any, model_id: str, bins: int
) -> dict[str, Any]:
    """Use every supplied generation, independent of labels and extraction --limit."""
    total = count = 0
    corpus = []
    for path in inputs:
        for row in iter_jsonl(path):
            trace = TraceRecord(**row)
            length = len(token_layout(trace, tokenizer)["reasoning_tokens"])
            total += length
            count += 1
            corpus.append((trace.dataset, trace.problem_id, trace_digest(trace), length))
    if not total or not count:
        raise ValueError("Cannot define position windows for empty reasoning")
    return {
        "model_id": model_id,
        "mean_reasoning_tokens": total / count,
        "traces": count,
        "bins": bins,
        "corpus_sha256": digest(sorted(corpus)),
        "rule": "ceil(model_corpus_mean * i / bins); noncumulative; overflow after mean",
        "limit_applied_to_reference": False,
    }


def compute_views(
    trace: TraceRecord,
    tokenizer: Any,
    router_logits: Any,
    hidden_states: Any,
    selected_experts: Any,
    layer_indices: list[int] | None,
    *,
    modes: list[str],
    reference: dict[str, Any] | None,
    max_geometry_tokens: int,
) -> dict[str, Any]:
    layout = token_layout(trace, tokenizer)
    if layout["token_count"] != router_logits.shape[1]:
        raise ValueError("Tokenizer span offsets do not align with extracted routing tokens")
    selected = selected_sentence_indices(trace, layout["units"])
    selected_tokens = {token for index in selected for token in layout["unit_tokens"][index]}
    tokens = [token for token in layout["reasoning_tokens"] if token in selected_tokens]
    scopes = []
    if "full" in modes:
        scopes.append({"view": "full", "name": "reasoning", "tokens": tokens})
    annotation = trace.metadata.get("reasoning_annotation")
    if "class" in modes:
        if annotation is None:
            raise ValueError("Class views require GEPA sentence annotations")
        validate_annotation(trace, annotation)
        for label in SENTENCE_LABELS:
            indices, segments = [], []
            for unit in annotation["units"]:
                owned = layout["unit_tokens"][unit["index"]]
                if unit["label"] == label:
                    indices.extend(owned)
                    segments.extend([unit["index"]] * len(owned))
            scopes.append({"view": "class", "name": label, "tokens": indices, "segments": segments})
    if "position" in modes:
        if reference is None:
            raise ValueError("Position views require a fixed model-corpus reference")
        for window in position_windows(
            layout["reasoning_tokens"], reference["mean_reasoning_tokens"], reference["bins"]
        ):
            window["tokens"] = [token for token in window["tokens"] if token in selected_tokens]
            scopes.append({"view": "position", **window})
    features = []
    for scope in scopes:
        indices = scope.pop("tokens")
        segments = scope.pop("segments", None)
        values = (
            compute_layer_features(
                router_logits,
                hidden_states,
                selected_experts,
                max_geometry_tokens=max_geometry_tokens,
                layer_indices=layer_indices,
                token_indices=indices,
                segment_ids=segments,
            )
            if indices
            else {}
        )
        transitions = sum(
            right == left + 1 and (segments is None or segments[i] == segments[i + 1])
            for i, (left, right) in enumerate(pairwise(indices))
        )
        features.append(
            {
                **scope,
                "token_count": len(indices),
                "transition_count": transitions,
                "values": json_safe_features(values),
            }
        )
    labels = {unit["index"]: unit["label"] for unit in annotation["units"]} if annotation else {}
    return {
        "schema_version": SPAN_SCHEMA_VERSION,
        "trace_sha256": trace_digest(trace),
        "reasoning_token_count": len(tokens),
        "sentence_selection": trace.metadata.get("sentence_selection"),
        "position_reference": reference,
        "annotation_sha256": digest(annotation) if annotation else None,
        "token_assignment": "max_character_overlap; preceding unit for whitespace; earlier ties",
        "sentence_spans": [
            {
                **unit,
                "token_ranges": _ranges(owned),
                **({"label": labels[i]} if i in labels else {}),
            }
            for i, (unit, owned) in enumerate(zip(layout["units"], layout["unit_tokens"]))
        ],
        "scopes": features,
    }


def _ranges(tokens: list[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for token in tokens:
        if ranges and ranges[-1][1] == token:
            ranges[-1][1] = token + 1
        else:
            ranges.append([token, token + 1])
    return ranges

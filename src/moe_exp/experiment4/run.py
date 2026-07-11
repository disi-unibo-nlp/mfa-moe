"""Prospective prefix-only probes for reasoning failure.

The experiment consumes Experiment 2 traces produced with
``--extract-hidden-states``.  At fixed fractions of each generated trace it
pools only tokens available up to that point, then compares structure-only,
router-only, hidden-only, and hidden+router linear probes.  Every evaluation
uses one sample per trace and stratified trace-level cross-validation.

Targets
-------
``final_incorrect``
    Whether the completed trace is incorrect, predicted from an earlier prefix.
``future_first_error``
    Whether a gold first-error step occurs after the prefix. Traces whose first
    error is already inside the observed prefix are excluded at that fraction.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from moe_exp.analysis.event_routing import _map_steps_to_token_ranges
from moe_exp.analysis.linear_probe import evaluate_layer, hidden_features, router_features
from moe_exp.models.inference import _format_prompt
from moe_exp.schemas import TraceRecord


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOURCES = ("structure", "router", "hidden", "combined")
TARGETS = ("final_incorrect", "future_first_error")


def _first_error_token(trace: TraceRecord, tokenizer, seq_len: int) -> int | None:
    step = trace.step_labels.first_error_step
    if step is None or step < 0 or step >= len(trace.steps):
        return None
    formatted_prompt = _format_prompt(tokenizer, trace.prompt, trace.system_prompt)
    ranges = _map_steps_to_token_ranges(
        trace.steps, trace.cot_text, tokenizer, formatted_prompt
    )
    start, end = ranges[step]
    if start < 0 or end <= start or start >= seq_len:
        return None
    return start


def _labels_for_prefix(
    trace: TraceRecord, prefix_end: int, first_error_token: int | None
) -> dict[str, int | None]:
    final_incorrect = None if trace.is_correct is None else int(not trace.is_correct)
    if trace.step_labels.first_error_step is None:
        future_first_error: int | None = 0
    elif first_error_token is None:
        future_first_error = None
    elif first_error_token >= prefix_end:
        future_first_error = 1
    else:
        # The gold error has already occurred, so using this prefix would leak
        # the event that the target asks us to predict.
        future_first_error = None
    return {
        "final_incorrect": final_incorrect,
        "future_first_error": future_first_error,
    }


def _evaluate_source(
    x: np.ndarray,
    y: np.ndarray,
    source: str,
    folds: int,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if source == "structure":
        result = evaluate_layer(x, y, folds, seed, bootstrap_samples)
        result["status"] = "complete"
        return result
    layer_rows: list[dict[str, Any]] = []
    for layer in range(x.shape[1]):
        logger.info("source=%s layer=%d", source, layer)
        row = evaluate_layer(x[:, layer, :], y, folds, seed, bootstrap_samples)
        row["layer"] = layer
        layer_rows.append(row)
    return {"status": "complete", "layers": layer_rows}


def run_experiment(
    input_path: Path,
    output_path: Path,
    model_id: str,
    fractions: list[float],
    sources: list[str],
    targets: list[str],
    folds: int,
    bootstrap_samples: int,
    minimum_class_count: int,
    seed: int,
    limit: int | None = None,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    traces: list[TraceRecord] = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                traces.append(TraceRecord.model_validate_json(line))
    if limit is not None:
        traces = traces[:limit]

    # Compact pooled features only: no raw token-level tensors are copied.
    rows: dict[float, dict[str, list[Any]]] = {
        fraction: {
            "router": [],
            "hidden": [],
            "structure": [],
            "labels": [],
        }
        for fraction in fractions
    }
    n_missing_hidden = 0
    n_missing_router = 0

    for index, trace in enumerate(traces, start=1):
        router_path = trace.model_logs.router_logits
        hidden_path = trace.model_logs.hidden_states
        if router_path is None or not Path(router_path).exists():
            n_missing_router += 1
            continue
        if hidden_path is None or not Path(hidden_path).exists():
            n_missing_hidden += 1
            continue
        router = torch.load(router_path, map_location="cpu", weights_only=True)
        hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
        if router.ndim != 3 or hidden.ndim != 3:
            continue
        if router.shape[:2] != hidden.shape[:2] or router.shape[1] < 1:
            continue
        seq_len = int(router.shape[1])
        first_error = _first_error_token(trace, tokenizer, seq_len)

        for fraction in fractions:
            prefix_end = min(seq_len, max(1, int(np.floor(seq_len * fraction))))
            rows[fraction]["router"].append(router_features(router[:, :prefix_end]))
            rows[fraction]["hidden"].append(hidden_features(hidden[:, :prefix_end]))
            rows[fraction]["structure"].append(
                np.asarray(
                    [
                        np.log1p(prefix_end),
                        np.log1p(seq_len),
                        np.log1p(len(trace.steps)),
                        np.log1p(len(trace.cot_text)),
                    ],
                    dtype=np.float32,
                )
            )
            rows[fraction]["labels"].append(
                _labels_for_prefix(trace, prefix_end, first_error)
            )
        if index % 250 == 0:
            logger.info("Loaded prospective features for %d/%d traces", index, len(traces))

    results: dict[str, Any] = {}
    for fraction in fractions:
        fraction_rows = rows[fraction]
        if not fraction_rows["router"]:
            continue
        router_x = np.stack(fraction_rows["router"])
        hidden_x = np.stack(fraction_rows["hidden"])
        structure_x = np.stack(fraction_rows["structure"])
        labels = fraction_rows["labels"]
        fraction_result: dict[str, Any] = {}

        for target in targets:
            raw_y = [row[target] for row in labels]
            mask = np.asarray([value is not None for value in raw_y])
            y = np.asarray([value for value in raw_y if value is not None], dtype=np.int8)
            counts = np.bincount(y, minlength=2)
            target_result: dict[str, Any] = {
                "n_samples": int(len(y)),
                "class_counts": {"negative": int(counts[0]), "positive": int(counts[1])},
                "sources": {},
            }
            if counts.min() < minimum_class_count:
                target_result.update(
                    status="skipped",
                    reason="insufficient examples in at least one class",
                )
                fraction_result[target] = target_result
                continue
            target_result["status"] = "complete"
            n_folds = min(folds, int(counts.min()))
            target_result["n_folds"] = n_folds
            for source in sources:
                logger.info(
                    "fraction=%.2f target=%s source=%s n=%d",
                    fraction,
                    target,
                    source,
                    len(y),
                )
                if source == "structure":
                    source_x = structure_x[mask]
                elif source == "router":
                    source_x = router_x[mask]
                elif source == "hidden":
                    source_x = hidden_x[mask]
                else:
                    source_x = np.concatenate((hidden_x[mask], router_x[mask]), axis=-1)
                target_result["sources"][source] = _evaluate_source(
                    source_x,
                    y,
                    source,
                    n_folds,
                    seed,
                    bootstrap_samples,
                )
            fraction_result[target] = target_result
        results[f"{fraction:.2f}"] = fraction_result

    output = {
        "config": {
            "input": input_path.as_posix(),
            "model_id": model_id,
            "fractions": fractions,
            "sources": sources,
            "targets": targets,
            "folds_requested": folds,
            "bootstrap_samples": bootstrap_samples,
            "minimum_class_count": minimum_class_count,
            "seed": seed,
            "n_input_traces": len(traces),
            "n_missing_router": n_missing_router,
            "n_missing_hidden": n_missing_hidden,
            "prospective": True,
            "uses_only_prefix_tokens": True,
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    logger.info("Experiment 4 results saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--sources", choices=SOURCES, nargs="+", default=list(SOURCES))
    parser.add_argument("--targets", choices=TARGETS, nargs="+", default=list(TARGETS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--minimum-class-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if any(not 0.0 < fraction < 1.0 for fraction in args.fractions):
        parser.error("all --fractions must be strictly between 0 and 1")
    run_experiment(
        input_path=args.input,
        output_path=args.output,
        model_id=args.model_id,
        fractions=args.fractions,
        sources=args.sources,
        targets=args.targets,
        folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        minimum_class_count=args.minimum_class_count,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

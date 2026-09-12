"""Select sentences from one saved solution per problem for frozen tagging."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from moe_exp.correlation_pipeline.generate import _model_slug, _write_json_atomic
from moe_exp.correlation_pipeline.spans import digest, sentence_spans, trace_digest
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord


DEFAULT_DATASETS = ["math500", "aime24", "aime25", "olympiad", "amc23", "minerva"]


def select_solutions(records: list[dict], *, sample_id: int, repeated: bool) -> list[TraceRecord]:
    """Keep single-attempt datasets intact; select one attempt only for repeats."""
    chosen = [TraceRecord(**row) for row in records
              if not repeated or row.get("sample_id", 0) == sample_id]
    expected = {row.get("source_problem_id") or row["problem_id"] for row in records}
    actual = [trace.source_problem_id or trace.problem_id for trace in chosen]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Selection must retain exactly one solution for every source problem")
    return chosen


def sentence_quotas(capacities: dict[str, int], weights: dict[str, float], cap: int) -> dict[str, int]:
    """Use the largest proportional allocation allowed by supply and the cap.

    Largest-remainder rounding differs from exact proportional shares by less
    than one sentence. We do not redistribute a benchmark's unavailable supply
    into the others, which would change the requested benchmark weighting.
    """
    if cap < 1 or set(weights) != set(capacities):
        raise ValueError("A positive cap and matching benchmark weights are required")
    if any(not math.isfinite(w) or w < 0 for w in weights.values()) or not sum(weights.values()):
        raise ValueError("Weights must be finite, nonnegative, and not all zero")
    if any(n < 0 for n in capacities.values()):
        raise ValueError("Sentence capacities cannot be negative")
    total_weight = sum(weights.values())
    total = min(cap, math.floor(min(
        capacities[name] * total_weight / weight
        for name, weight in weights.items() if weight > 0
    )))
    exact = {name: total * weight / total_weight for name, weight in weights.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(quotas.values())
    for name in sorted(exact, key=lambda name: (-(exact[name] - quotas[name]), name)):
        if remaining and quotas[name] < capacities[name] and weights[name] > 0:
            quotas[name] += 1
            remaining -= 1
    if remaining or not total:
        raise ValueError("Insufficient sentence supply for the requested allocation")
    return quotas


def sample(args: argparse.Namespace) -> dict:
    analysis = json.loads(args.analysis.read_text()) if args.analysis else None
    budget = ({row["scope"]: row for row in analysis["generation_budget_audit"]["scopes"]}
              if analysis else None)
    model_slug = _model_slug(args.generation_model)
    traces = {}
    units = {}
    weights = {}
    inputs = {}
    source_manifests = {}
    for dataset in args.datasets:
        path = args.generation_dir / model_slug / dataset / "traces.jsonl"
        inputs[dataset] = str(path)
        source_manifests[dataset] = json.loads((path.parent / "manifest.json").read_text())
        records = list(iter_jsonl(path))
        chosen = select_solutions(
            records, sample_id=args.sample_id,
            repeated=source_manifests[dataset].get("samples_per_problem", 1) > 1,
        )
        if not chosen:
            raise ValueError(f"No sample ID {args.sample_id} in {dataset}")
        problem_ids = [trace.source_problem_id or trace.problem_id for trace in chosen]
        if len(set(problem_ids)) != len(problem_ids):
            raise ValueError(f"More than one chosen solution per problem in {dataset}")
        traces[dataset] = chosen
        units[dataset] = [sentence_spans(trace) for trace in chosen]
        scored = [row["is_correct"] for row in records if row.get("is_correct") is not None]
        accuracy = (budget[dataset]["accuracy_all"] if budget is not None
                    else sum(scored) / len(scored) if scored else None)
        if accuracy is None or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"No valid benchmark accuracy for {dataset}")
        weights[dataset] = 1 - accuracy if args.weight_by == "error-rate" else accuracy
    capacities = {name: sum(map(len, spans)) for name, spans in units.items()}
    quotas = sentence_quotas(capacities, weights, args.max_sentences)
    selections = {}
    for dataset in args.datasets:
        rng = random.Random(f"{args.seed}:{dataset}")
        picked = set(rng.sample(range(capacities[dataset]), quotas[dataset]))
        selected = defaultdict(list)
        offset = 0
        for trace, spans in zip(traces[dataset], units[dataset], strict=True):
            selected[trace.problem_id]  # Retain outcomes even if no sentence is sampled.
            for index in range(len(spans)):
                if offset + index in picked:
                    selected[trace.problem_id].append(index)
            offset += len(spans)
        selections[dataset] = dict(selected)
    plan = {
        "schema_version": 1,
        "source_analysis": str(args.analysis) if args.analysis else None,
        "source_analysis_sha256": hashlib.sha256(args.analysis.read_bytes()).hexdigest() if args.analysis else None,
        "generation_inputs": inputs,
        "generation_model": args.generation_model,
        "sample_id": args.sample_id,
        "seed": args.seed,
        "max_sentences": args.max_sentences,
        "weight_by": args.weight_by,
        "weights": weights,
        "available_sentences": capacities,
        "sentence_quotas": quotas,
        "selected_sentences": sum(quotas.values()),
        "selection_rule": "uniform sentences without replacement within each benchmark; proportional benchmark quotas capped by supply",
        "correctness_rule": "weights use all-attempt benchmark accuracy; individual sentences are sampled without conditioning on correctness",
        "statistical_unit": "one original attempt per problem; sampled sentences are not independent correctness trials",
        "trace_fingerprints": {name: {trace.problem_id: trace_digest(trace) for trace in rows} for name, rows in traces.items()},
        "selected_indices": selections,
        "selected_traces": {name: sum(bool(indices) for indices in selected.values())
                            for name, selected in selections.items()},
        "eligible_traces": {name: len(rows) for name, rows in traces.items()},
    }
    plan_hash = digest(plan)
    manifest_path = args.output_dir / "sampling_manifest.json"
    manifest = {**plan, "manifest_sha256": plan_hash}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output contains a different sampling plan; choose a new output directory")
    # Preserve original continuations, sample IDs, and avg@32 provenance. The
    # analyzer must report incomplete repeated groups rather than invent avg@1.
    for dataset, rows in traces.items():
        directory = args.output_dir / "generation" / model_slug / dataset
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / ".traces.jsonl.tmp"
        with temporary.open("w") as handle:
            for trace in rows:
                indices = selections[dataset].get(trace.problem_id)
                trace.metadata["sentence_selection"] = {
                    "schema_version": 1, "manifest_sha256": plan_hash,
                    "weight_by": args.weight_by, "sample_id": args.sample_id,
                    "indices": indices,
                }
                handle.write(trace.model_dump_json() + "\n")
        temporary.replace(directory / "traces.jsonl")
        original_manifest = source_manifests[dataset]
        selected_rows = [trace for trace in rows if trace.problem_id in selections[dataset]]
        scored = [trace for trace in selected_rows if trace.is_correct is not None]
        correct = sum(bool(trace.is_correct) for trace in scored)
        _write_json_atomic({
            **original_manifest,
            "status": "complete",
            "problems": len(selected_rows), "traces": len(selected_rows),
            "samples_per_problem": 1,
            "scored_traces": len(scored), "correct_traces": correct,
            "accuracy": correct / len(scored) if scored else None,
            "trace_path": str(directory / "traces.jsonl"),
            "sampling_manifest_sha256": plan_hash,
            "source_samples_per_problem": original_manifest.get("samples_per_problem"),
            "source_manifest": str(args.generation_dir / model_slug / dataset / "manifest.json"),
        }, directory / "manifest.json")
    _write_json_atomic(manifest, manifest_path)
    return {key: value for key, value in manifest.items() if key not in {"selected_indices", "trace_fingerprints"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path,
                        help="Optional accuracy audit; otherwise use all scored generation attempts")
    parser.add_argument("--generation-dir", type=Path, default=Path("results/correlation_pipeline/generation"))
    parser.add_argument("--generation-model", default="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sentences", type=int, default=100000)
    parser.add_argument("--weight-by", choices=["accuracy", "error-rate"], default="error-rate")
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(sample(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()

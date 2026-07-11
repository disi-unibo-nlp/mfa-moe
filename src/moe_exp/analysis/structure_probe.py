"""Non-routing trace-structure baseline for the layerwise routing probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from moe_exp.analysis.linear_probe import TARGETS, _target_value, evaluate_layer
from moe_exp.schemas import TraceRecord


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=["correctness"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--minimum-class-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    traces: list[TraceRecord] = []
    features: list[list[float]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = TraceRecord.model_validate_json(line)
            tensor_path = trace.model_logs.selected_experts or trace.model_logs.router_logits
            if tensor_path is None or not Path(tensor_path).exists():
                continue
            tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
            seq_len = int(tensor.shape[1])
            traces.append(trace)
            features.append(
                [
                    float(np.log1p(seq_len)),
                    float(np.log1p(len(trace.steps))),
                    float(np.log1p(len(trace.cot_text))),
                ]
            )
    x_all = np.asarray(features, dtype=np.float32)

    target_results: dict[str, dict] = {}
    for target in args.targets:
        labels = [_target_value(trace, target) for trace in traces]
        mask = np.asarray([label is not None for label in labels])
        y = np.asarray([label for label in labels if label is not None], dtype=np.int8)
        counts = np.bincount(y, minlength=2)
        if counts.min() < args.minimum_class_count:
            target_results[target] = {
                "status": "skipped",
                "reason": "insufficient examples in at least one class",
                "class_counts": {"negative": int(counts[0]), "positive": int(counts[1])},
            }
            continue
        folds = min(args.folds, int(counts.min()))
        result = evaluate_layer(
            x_all[mask], y, folds, args.seed, args.bootstrap_samples
        )
        result.update(
            {
                "status": "complete",
                "n_samples": int(len(y)),
                "class_counts": {"negative": int(counts[0]), "positive": int(counts[1])},
                "n_folds": folds,
            }
        )
        target_results[target] = result

    output = {
        "config": {
            "input": args.input.as_posix(),
            "features": ["log_token_count", "log_step_count", "log_character_count"],
            "unit": "one sample per trace",
            "uses_full_trace": True,
            "n_traces": len(traces),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "targets": target_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Structure baseline saved to {args.output}")


if __name__ == "__main__":
    main()

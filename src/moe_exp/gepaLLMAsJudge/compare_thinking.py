"""Benchmark the frozen correlation tagger across judge thinking levels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import load_documents
from .metrics import compute_classification_metrics, parse_sentence_label


DEFAULT_RUN = Path("results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3")
DEFAULT_PROGRAM = DEFAULT_RUN / "selected_program_20260827_173300.json"
DEFAULT_SPLIT = DEFAULT_RUN / "results_20260827_173300.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--judge-model", default="unsloth/Qwen3.8-27B-NVFP4")
    parser.add_argument("--base-url", default="http://127.0.0.1:41800/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local-vllm-key"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/Schoenfeld_Reasoning"))
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT,
                        help="GEPA results JSON containing split_question_ids.validation")
    parser.add_argument("--levels", nargs="+", choices=("off", "low", "medium", "high"),
                        default=["off", "low", "medium", "high"])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Seeded sample of validation units for a smoke run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        help="New directory; defaults to a timestamped results directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print request counts without loading the model")
    return parser


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    ids = split["split_question_ids"]["validation"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Validation question IDs must be nonempty and unique")
    documents = load_documents(args.dataset_dir)
    missing = set(ids) - {doc.question_id for doc in documents}
    if missing:
        raise ValueError(f"Validation responses missing from dataset: {sorted(missing)}")
    cases = []
    for doc in documents:
        if doc.question_id not in ids:
            continue
        for index, unit in enumerate(doc.units):
            cases.append({
                "question_id": doc.question_id,
                "unit_id": index,
                "gold_label": unit.sentence_label,
                "inputs": {
                    "problem_statement": doc.problem_statement or "<PROBLEM UNAVAILABLE>",
                    "previous_sentence": doc.units[index - 1].text if index else
                    "<START OF RESPONSE>",
                    "sentence": unit.text,
                    "next_sentence": doc.units[index + 1].text if index + 1 < len(doc.units)
                    else "<END OF RESPONSE>",
                },
            })
    if args.limit is not None and args.limit < len(cases):
        indices = sorted(random.Random(args.seed).sample(range(len(cases)), args.limit))
        cases = [cases[index] for index in indices]
    return cases


def evaluate_case(predict: Any, case: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    raw_label = None
    label = None
    error = None
    try:
        prediction = predict(**case["inputs"])
        raw_label = str(getattr(prediction, "label", prediction))
        label = parse_sentence_label(prediction)
    except Exception as exc:
        # Invalid outputs and transport failures count as wrong, never disappear.
        error = f"{type(exc).__name__}: {exc}"
    return {
        "question_id": case["question_id"], "unit_id": case["unit_id"],
        "gold_label": case["gold_label"], "label": label, "raw_label": raw_label,
        "error": error, "latency_seconds": time.perf_counter() - start,
    }


def summarize(rows: list[dict[str, Any]], levels: list[str]) -> dict[str, Any]:
    report = {}
    baseline = {(r["repeat"], r["question_id"], r["unit_id"]): r["label"]
                for r in rows if r["level"] == levels[0]}
    for level in levels:
        selected = [r for r in rows if r["level"] == level]
        metrics = compute_classification_metrics(
            [r["gold_label"] for r in selected], [r["label"] for r in selected]
        )
        latency = sorted(r["latency_seconds"] for r in selected)
        metrics.update({
            "requests": len(selected),
            "errors": sum(r["error"] is not None for r in selected),
            "valid_output_coverage": sum(r["label"] is not None for r in selected) / len(selected),
            "mean_latency_seconds": statistics.mean(latency),
            "median_latency_seconds": statistics.median(latency),
            "p95_latency_seconds": latency[max(0, (95 * len(latency) + 99) // 100 - 1)],
            "total_request_seconds": sum(latency),
            "disagreement_vs_baseline": sum(
                r["label"] != baseline[(r["repeat"], r["question_id"], r["unit_id"])]
                for r in selected
            ) / len(selected),
        })
        report[level] = metrics
    for metrics in report.values():
        metrics["accuracy_delta_vs_baseline"] = metrics["accuracy"] - report[levels[0]]["accuracy"]
    return report


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--max-tokens, --repeats, and --limit must be positive")
    if len(set(args.levels)) != len(args.levels):
        parser.error("--levels must not contain duplicates")
    program_bytes = args.judge_program.read_bytes()
    json.loads(program_bytes)
    cases = load_cases(args)
    total = len(cases) * len(args.levels) * args.repeats
    print(f"Validation units: {len(cases)}; levels: {', '.join(args.levels)}; requests: {total}",
          flush=True)
    if args.dry_run:
        return

    # Lazy import keeps --help and --dry-run usable without the DSPy extra.
    from moe_exp.correlation_pipeline.annotate import make_predictor

    output = args.output_dir or Path("results/gepaLLMAsJudge") / (
        "thinking-comparison-" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key != "api_key"}
    config.update({"program_sha256": hashlib.sha256(program_bytes).hexdigest(),
                   "baseline": args.levels[0], "cache": False, "workers": 1,
                   "evaluation": "Previously used validation set; exploratory comparison",
                   "output_dir": str(output)})
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "program.json").write_bytes(program_bytes)
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    predictors = {}
    for level in args.levels:
        level_args = argparse.Namespace(**vars(args))
        level_args.enable_thinking = level != "off"
        level_args.reasoning_effort = level if level != "off" else "low"
        predictors[level] = make_predictor(level_args)

    rows = []
    rng = random.Random(args.seed)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for repeat in range(args.repeats):
            order = list(cases)
            rng.shuffle(order)
            for case in order:
                levels = list(args.levels)
                rng.shuffle(levels)
                for level in levels:
                    row = {"level": level, "repeat": repeat,
                           **evaluate_case(predictors[level], case)}
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(f"[{len(rows)}/{total}] {level}: {row['label'] or row['error']} "
                          f"({row['latency_seconds']:.2f}s)", flush=True)

    report = summarize(rows, args.levels)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    flat = [{"level": level, **{k: v for k, v in metrics.items() if k != "per_class"}}
            for level, metrics in report.items()]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print("\nLevel      Accuracy  Balanced  Macro-F1  Valid     Mean(s)   Errors")
    for level, m in report.items():
        print(f"{level:<10} {m['accuracy']:.3f}     {m['balanced_accuracy']:.3f}     "
              f"{m['macro_f1']:.3f}     {m['valid_output_coverage']:.3f}     "
              f"{m['mean_latency_seconds']:.2f}      {m['errors']}")
    print(f"\nSaved comparison to {output}")
    if any(m["valid_output_coverage"] == 0 for m in report.values()):
        raise SystemExit("At least one level produced no valid labels; inspect predictions.jsonl")


if __name__ == "__main__":
    main()

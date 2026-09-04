from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from moe_exp.analysis.classifier import classify_trace
from moe_exp.analysis.step_splitter import split_steps
from moe_exp.correlation_pipeline.benchmarks import (
    BENCHMARKS,
    DEFAULT_BENCHMARKS,
    generation_messages,
    sample_variant,
)
from moe_exp.correlation_pipeline.client import generate_completion
from moe_exp.correlation_pipeline.defaults import (
    DEFAULT_FORWARD_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MTP_MODEL,
)
from moe_exp.correlation_pipeline.scoring import score_completion
from moe_exp.schemas import TraceRecord
from moe_exp.utils import write_jsonl

logger = logging.getLogger(__name__)
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _model_slug(model: str) -> str:
    return _SAFE_FILENAME.sub("--", model).strip("-") or "local-llamacpp"


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_reusable_shard(
    path: Path,
    *,
    example_sha256: str,
    generation_sha256: str,
) -> TraceRecord | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        trace = TraceRecord(**payload)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if (
        trace.metadata.get("example_sha256") != example_sha256
        or trace.metadata.get("generation_sha256") != generation_sha256
    ):
        return None
    return trace


def _generate_trace(
    *,
    dataset: str,
    example: dict[str, Any],
    example_index: int,
    sample_id: int,
    samples_per_problem: int,
    spec: Any,
    args: argparse.Namespace,
    shard_path: Path,
) -> TraceRecord:
    example = sample_variant(dataset, example, sample_id)
    messages = generation_messages(example)
    example_sha256 = _digest(
        {
            "dataset": dataset,
            "problem_id": example["problem_id"],
            "prompt": example["prompt"],
            "gold_answer": example.get("gold_answer"),
            "messages": messages,
        }
    )
    sample_seed = args.seed + example_index * 1000 + sample_id
    generation_config = {
        "base_url": args.base_url,
        "served_model": args.model,
        "target_model_id": args.target_model_id,
        "draft_model_id": args.draft_model_id,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": sample_seed,
    }
    generation_sha256 = _digest(generation_config)
    reusable = _load_reusable_shard(
        shard_path,
        example_sha256=example_sha256,
        generation_sha256=generation_sha256,
    )
    if reusable is not None:
        return reusable

    completion = generate_completion(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=sample_seed,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    model_answer, is_correct, scoring_method = score_completion(
        example,
        answer_type=spec.answer_type,
        model_text=completion.text,
    )
    steps = split_steps(completion.text)
    source_problem_id = str(example["problem_id"])
    problem_id = f"{source_problem_id}__sample_{sample_id:02d}"
    source_metadata = dict(example.get("metadata") or {})
    source_metadata.update(
        {
            "benchmark_source": spec.source,
            "benchmark_split": spec.split,
            "evaluation_metric": (
                f"avg@{samples_per_problem}" if samples_per_problem > 1 else "pass@1"
            ),
            "answer_type": spec.answer_type,
            "example_sha256": example_sha256,
            "generation_sha256": generation_sha256,
            "generation_config": generation_config,
            "finish_reason": completion.finish_reason,
            "usage": completion.usage,
            "assistant_content": completion.content,
            "reasoning_content": completion.reasoning_content,
        }
    )
    if example.get("first_error_step") is not None:
        source_metadata["reference_first_error_step"] = example["first_error_step"]
    if example.get("solution_is_correct") is not None:
        source_metadata["reference_solution_is_correct"] = example["solution_is_correct"]

    trace = TraceRecord(
        dataset=dataset,
        problem_id=problem_id,
        source_problem_id=source_problem_id,
        sample_id=sample_id,
        prompt=str(example["prompt"]),
        generation_messages=messages,
        gold_answer=str(example.get("gold_answer") or ""),
        model_id=args.model,
        model_answer=model_answer,
        is_correct=is_correct,
        cot_text=completion.text,
        steps=steps,
        step_labels=classify_trace(steps, completion.text),
        scoring_method=scoring_method,
        metadata=source_metadata,
        task_type="reasoning",
    )
    _write_json_atomic(trace.model_dump(mode="json"), shard_path)
    return trace


def _dataset_summary(
    dataset: str,
    traces: list[TraceRecord],
    *,
    samples_per_problem: int,
) -> dict[str, Any]:
    scored = [trace for trace in traces if trace.is_correct is not None]
    correct = sum(trace.is_correct is True for trace in scored)
    problem_ids = {trace.source_problem_id or trace.problem_id for trace in traces}
    return {
        "dataset": dataset,
        "status": "complete",
        "problems": len(problem_ids),
        "samples_per_problem": samples_per_problem,
        "traces": len(traces),
        "scored_traces": len(scored),
        "correct_traces": correct,
        "accuracy": (correct / len(scored)) if scored else None,
    }


def generate_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    spec = BENCHMARKS[dataset]
    examples = [example for example in spec.loader(args.max_items) if example.get("prompt")]
    if not examples:
        raise RuntimeError(f"Benchmark {dataset!r} produced zero usable examples")
    samples_per_problem = args.samples_per_problem or spec.default_samples
    dataset_dir = args.output_dir / _model_slug(args.model) / dataset
    shard_dir = dataset_dir / "generation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        safe_id = _SAFE_FILENAME.sub("_", str(example["problem_id"]))
        for sample_id in range(samples_per_problem):
            jobs.append(
                {
                    "dataset": dataset,
                    "example": example,
                    "example_index": example_index,
                    "sample_id": sample_id,
                    "samples_per_problem": samples_per_problem,
                    "spec": spec,
                    "args": args,
                    "shard_path": shard_dir / f"{safe_id}__sample_{sample_id:02d}.json",
                }
            )

    if args.workers == 1:
        traces = [
            _generate_trace(**job)
            for job in tqdm(jobs, desc=f"Generating {dataset}")
        ]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            traces = list(
                tqdm(
                    executor.map(lambda job: _generate_trace(**job), jobs),
                    total=len(jobs),
                    desc=f"Generating {dataset}",
                )
            )

    traces.sort(key=lambda trace: (trace.source_problem_id or trace.problem_id, trace.sample_id))
    trace_path = dataset_dir / "traces.jsonl"
    write_jsonl(traces, trace_path)
    summary = _dataset_summary(
        dataset,
        traces,
        samples_per_problem=samples_per_problem,
    )
    summary.update(
        {
            "model": args.model,
            "target_model_id": args.target_model_id,
            "draft_model_id": args.draft_model_id,
            "source": spec.source,
            "split": spec.split,
            "trace_path": trace_path.as_posix(),
        }
    )
    _write_json_atomic(summary, dataset_dir / "manifest.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate correlation-pipeline benchmark traces with llama.cpp"
    )
    parser.add_argument("--datasets", nargs="+", choices=tuple(BENCHMARKS), default=DEFAULT_BENCHMARKS)
    parser.add_argument("--output-dir", type=Path, default=Path("results/correlation_pipeline/generation"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key", default="local-llamacpp-key")
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--target-model-id", default=DEFAULT_FORWARD_MODEL)
    parser.add_argument("--draft-model-id", default=DEFAULT_MTP_MODEL)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--samples-per-problem",
        type=int,
        default=None,
        help="Override defaults (AIME24/25 and AMC23 use 32; GPQA-D uses 10; others use 1).",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base sampling seed (SPIRAL uses 0).",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.samples_per_problem is not None and args.samples_per_problem < 1:
        raise ValueError("--samples-per-problem must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    summaries = [generate_dataset(dataset, args) for dataset in args.datasets]
    summary_path = args.output_dir / _model_slug(args.model) / "summary.json"
    _write_json_atomic({"status": "complete", "datasets": summaries}, summary_path)
    logger.info("Generation complete: %s", summary_path)


if __name__ == "__main__":
    main()

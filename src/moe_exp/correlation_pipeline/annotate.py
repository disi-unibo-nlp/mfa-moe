"""Apply a frozen GEPA-selected classifier to saved reasoning units."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from moe_exp.correlation_pipeline.defaults import DEFAULT_GENERATION_MODEL
from moe_exp.correlation_pipeline.generate import _model_slug, _write_json_atomic
from moe_exp.correlation_pipeline.spans import (
    SPAN_SCHEMA_VERSION,
    digest,
    reasoning_bounds,
    selected_sentence_indices,
    sentence_spans,
    trace_digest,
    validate_annotation,
)
from moe_exp.gepaLLMAsJudge.metrics import parse_sentence_label
from moe_exp.jsonl import iter_jsonl
from moe_exp.schemas import TraceRecord

logger = logging.getLogger(__name__)


def _annotation_state(trace: TraceRecord, config: dict[str, Any], path: Path):
    """Share checkpoint validation between progress inventory and execution."""
    contract = {
        "schema_version": SPAN_SCHEMA_VERSION,
        "trace_sha256": trace_digest(trace),
        "classifier": config,
        "dataset": trace.dataset,
        "problem_id": trace.problem_id,
        "reasoning_span": list(reasoning_bounds(trace)),
        "sentence_selection": trace.metadata.get("sentence_selection"),
    }
    saved: dict[str, Any] = {}
    if path.is_file():
        try:
            saved = json.loads(path.read_text())
        except (ValueError, OSError):
            pass
    units = sentence_spans(trace)
    selected = selected_sentence_indices(trace, units)
    selected_set = set(selected)
    reusable = saved.get("units", []) if all(saved.get(k) == v for k, v in contract.items()) else []
    # Partial checkpoints may be sparse: requests complete out of source order.
    labeled = {}
    for item in reusable:
        index = item.get("index")
        if not isinstance(index, int) or index not in selected_set:
            continue
        if all(item.get(k) == v for k, v in units[index].items()):
            try:
                labeled[index] = {**units[index], "label": parse_sentence_label(item.get("label"))}
            except ValueError:
                pass
    return contract, units, selected, labeled


def annotate_trace(
    trace: TraceRecord,
    predict: Callable[..., Any],
    config: dict[str, Any],
    path: Path,
    *,
    workers: int = 1,
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("--workers must be positive")
    contract, units, selected, labeled = _annotation_state(trace, config, path)
    question = (
        "\n\n".join(
            message["content"]
            for message in (trace.generation_messages or [])
            if message["role"] == "user"
        )
        or trace.prompt
    )

    def classify(index: int) -> dict[str, Any]:
        # Same question/previous/current/next context; never expose gold labels.
        prediction = predict(
            problem_statement=question,
            previous_sentence=units[index - 1]["text"] if index else "<START OF RESPONSE>",
            sentence=units[index]["text"],
            next_sentence=units[index + 1]["text"]
            if index + 1 < len(units)
            else "<END OF RESPONSE>",
        )
        return {**units[index], "label": parse_sentence_label(prediction)}

    def save(index: int, item: dict[str, Any]) -> None:
        labeled[index] = item
        _write_json_atomic(
            {**contract, "units": [labeled[i] for i in sorted(labeled)], "status": "partial"}, path
        )
        if on_progress is not None:
            on_progress(1)

    remaining = iter(index for index in selected if index not in labeled)
    if workers == 1:
        for index in remaining:
            save(index, classify(index))
    else:
        # Bound both running and queued requests. Only this thread writes files.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}

            def submit_next() -> None:
                index = next(remaining, None)
                if index is not None:
                    pending[pool.submit(classify, index)] = index

            try:
                for _ in range(workers):
                    submit_next()
                while pending:
                    done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                    if on_progress is not None:
                        on_progress(0)
                    for future in done:
                        index = pending.pop(future)
                        save(index, future.result())
                    for _ in done:
                        submit_next()
            finally:
                for future in pending:
                    future.cancel()
    result = {**contract, "units": [labeled[i] for i in selected], "status": "complete"}
    validate_annotation(trace, result)
    _write_json_atomic(result, path)
    return result


def make_predictor(args: argparse.Namespace) -> Callable[..., Any]:
    import dspy

    from moe_exp.gepaLLMAsJudge.run import EpisodeJudge, _make_lm

    program = EpisodeJudge()
    program.load(str(args.judge_program))
    lm_args = argparse.Namespace(
        model=args.judge_model,
        api_base=args.base_url,
        api_key=args.api_key,
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort,
    )
    lm = _make_lm(lm_args, max_tokens=args.max_tokens, temperature=args.temperature)

    def predict(**kwargs: Any) -> Any:
        with dspy.context(lm=lm):
            return program(**kwargs)

    _quiet_request_logs()
    return predict


def _quiet_request_logs() -> None:
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


class TaggingProgress:
    """Periodic plain lines for terminals and redirected job logs."""

    def __init__(self, total: int, reused: int, traces: int):
        self.total, self.reused, self.traces = total, reused, traces
        self.completed = reused
        self.finished_traces = 0
        self.started = time.monotonic()
        self.last_report = self.started
        self.dataset = "starting"
        self.report(force=True)

    def advance(self, count: int) -> None:
        self.completed += count
        self.report()

    def report(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_report < 10:
            return
        self.last_report = now
        remaining = self.total - self.completed
        rate = (self.completed - self.reused) / max(now - self.started, 1e-9)
        eta = "0s" if not remaining else (
            f"{remaining / rate / 60:.1f} min" if rate else "estimating"
        )
        logger.info(
            "Tagging [%s]: %d/%d sentences (%.1f%%) | %d remaining | "
            "%d reused | %.2f sentences/s | ETA %s | traces %d/%d",
            self.dataset, self.completed, self.total,
            100 * self.completed / self.total if self.total else 100,
            remaining, self.reused, rate, eta, self.finished_traces, self.traces,
        )


def annotate_all(
    args: argparse.Namespace, predict: Callable[..., Any] | None = None
) -> dict[str, Any]:
    from moe_exp.correlation_pipeline.benchmarks import DEFAULT_BENCHMARKS

    if getattr(args, "workers", 1) < 1:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    program = json.loads(args.judge_program.read_text())
    if not program.get("classify", {}).get("signature", {}).get("instructions"):
        raise ValueError("--judge-program must be a saved EpisodeJudge program")
    config = {
        "program_sha256": digest(program),
        "model": args.judge_model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
        "context": "question_previous_current_next",
    }
    root = args.generation_dir / _model_slug(args.generation_model)
    datasets = args.datasets or [
        name for name in DEFAULT_BENCHMARKS if (root / name / "traces.jsonl").is_file()
    ]
    if not datasets:
        raise ValueError(f"No completed generation files under {root}")
    # Validate inputs before making any classifier calls.
    for dataset in datasets:
        if not (root / dataset / "traces.jsonl").is_file():
            raise FileNotFoundError(root / dataset / "traces.jsonl")
    logger.info("Counting selected sentences and validating saved tagging checkpoints...")
    total = reused = trace_total = 0
    for dataset in datasets:
        directory = args.output_dir / _model_slug(args.generation_model) / dataset
        for index, row in enumerate(iter_jsonl(root / dataset / "traces.jsonl")):
            if args.limit is not None and index >= args.limit:
                break
            trace = TraceRecord(**row)
            path = directory / "shards" / f"{digest(trace.problem_id)}.json"
            _, _, selected, labeled = _annotation_state(trace, config, path)
            total += len(selected)
            reused += len(labeled)
            trace_total += 1
    progress = None if args.dry_run else TaggingProgress(total, reused, trace_total)
    if not args.dry_run and predict is None:
        predict = make_predictor(args)
    summaries = []
    for dataset in datasets:
        if progress is not None:
            progress.dataset = dataset
            progress.report(force=True)
        directory = args.output_dir / _model_slug(args.generation_model) / dataset
        counts: Counter[str] = Counter()
        traces = units = 0
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / ".annotations.jsonl.tmp"
        handle = temporary.open("w") if not args.dry_run else None
        try:
            for row in iter_jsonl(root / dataset / "traces.jsonl"):
                if args.limit is not None and traces >= args.limit:
                    break
                trace = TraceRecord(**row)
                if args.dry_run:
                    units += len(selected_sentence_indices(trace, sentence_spans(trace)))
                else:
                    assert predict is not None and handle is not None
                    path = directory / "shards" / f"{digest(trace.problem_id)}.json"
                    annotation = annotate_trace(
                        trace, predict, config, path, workers=getattr(args, "workers", 1),
                        on_progress=progress.advance if progress is not None else None,
                    )
                    handle.write(json.dumps(annotation, ensure_ascii=False) + "\n")
                    counts.update(unit["label"] for unit in annotation["units"])
                    units += len(annotation["units"])
                traces += 1
                if progress is not None:
                    progress.finished_traces += 1
                    progress.report()
        finally:
            if handle is not None:
                handle.close()
        summary = {
            "dataset": dataset,
            "traces": traces,
            "units": units,
            "label_counts": dict(counts),
            "limited": args.limit is not None,
        }
        if not args.dry_run:
            temporary.replace(directory / "annotations.jsonl")
            _write_json_atomic({**summary, "classifier": config}, directory / "manifest.json")
        summaries.append(summary)
        logger.info("%s complete: %d traces, %d selected sentences", dataset, traces, units)
        if progress is not None:
            progress.report(force=True)
    result = {
        "status": "dry_run" if args.dry_run else "complete",
        "schema_version": SPAN_SCHEMA_VERSION,
        "classifier": config,
        "datasets": summaries,
    }
    if not args.dry_run:
        _write_json_atomic(
            result, args.output_dir / _model_slug(args.generation_model) / "summary.json"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dir", type=Path, default=Path("results/correlation_pipeline/generation")
    )
    parser.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/correlation_pipeline/annotations-v1")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Default: completed datasets in the six-benchmark default suite",
    )
    parser.add_argument("--judge-program", type=Path, required=True)
    parser.add_argument(
        "--judge-model", required=True, help="Served model used to select the GEPA program"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:41800/v1")
    parser.add_argument("--api-key", default="local-vllm-key")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent sentence requests")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument(
        "--limit", type=int, default=None, help="Trace limit per dataset, retaining whole traces"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Count units without loading/calling the judge"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _quiet_request_logs()
    args = build_parser().parse_args(argv)
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    print(json.dumps(annotate_all(args), indent=2))


if __name__ == "__main__":
    main()

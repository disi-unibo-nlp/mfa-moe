"""Compare request concurrency using saved prompts, without changing experiment traces.

Only the standard library is needed, so this diagnostic also runs on host Python.
Completion-token throughput includes HTTP/queue time; request latencies overlap
when workers > 1 and must never be summed to estimate wall throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moe_exp.correlation_pipeline.client import _post_json


def load_workload(generation_dir: Path, requests: int) -> list[dict[str, Any]]:
    """Round-robin the datasets, retaining the saved messages and sampling seeds."""
    inputs = []
    for path in sorted(generation_dir.glob("*/traces.jsonl")):
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                config = record["metadata"]["generation_config"]
                records.append({
                    "dataset": record["dataset"],
                    "problem_id": record["problem_id"],
                    "messages": record["generation_messages"],
                    "model": config["served_model"],
                    "seed": config["seed"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "top_k": config["top_k"],
                })
                if len(records) == requests:
                    break
        if records:
            inputs.append(records)
    workload = []
    for index in range(requests):
        for records in inputs:
            if index < len(records):
                workload.append(records[index])
                if len(workload) == requests:
                    return workload
    raise ValueError(f"Need {requests} saved traces; found {len(workload)}")


def summarize_trial(
    records: list[dict[str, Any]], wall_seconds: float, *, workers: int, repeat: int
) -> dict[str, Any]:
    tokens = sum(record["completion_tokens"] for record in records)
    return {
        "workers": workers,
        "repeat": repeat,
        "requests": len(records),
        "completion_tokens": tokens,
        "wall_seconds": wall_seconds,
        "completion_tokens_per_second": tokens / wall_seconds,
        "mean_request_seconds": statistics.mean(record["wall_seconds"] for record in records),
        "length_limited_requests": sum(record["finish_reason"] == "length" for record in records),
        "by_dataset": {
            dataset: {
                "requests": sum(record["dataset"] == dataset for record in records),
                "completion_tokens": sum(
                    record["completion_tokens"] for record in records if record["dataset"] == dataset
                ),
            }
            for dataset in sorted({record["dataset"] for record in records})
        },
    }


def request_sample(sample: dict[str, Any], args: argparse.Namespace, max_tokens: int) -> dict[str, Any]:
    payload = {key: sample[key] for key in ("model", "messages", "seed", "temperature", "top_p", "top_k")}
    payload.update(max_tokens=max_tokens, stream=False)
    start = time.perf_counter()
    response = _post_json(
        f"{args.base_url.rstrip('/')}/chat/completions", args.api_key, payload, args.timeout
    )
    elapsed = time.perf_counter() - start
    choice = response["choices"][0]
    tokens = response["usage"]["completion_tokens"]
    if not isinstance(tokens, int) or tokens <= 0:
        raise ValueError(f"Invalid completion token count: {tokens!r}")
    return {
        "dataset": sample["dataset"],
        "problem_id": sample["problem_id"],
        "seed": sample["seed"],
        "wall_seconds": elapsed,
        "completion_tokens": tokens,
        "prompt_tokens": response["usage"].get("prompt_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "server_timings": response.get("timings"),
        "response_sha256": hashlib.sha256(
            json.dumps(choice["message"], sort_keys=True).encode()
        ).hexdigest(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key", default="local-llamacpp-key")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if min([args.requests, args.repeats, args.max_tokens, args.timeout, *args.workers]) < 1:
        parser.error("Request counts, workers, repeats, token limit and timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    workload = load_workload(args.generation_dir, args.requests)
    with (args.output_dir / "workload.json").open("w") as handle:
        json.dump(workload, handle, indent=2)
    # /props records the actual server slots/context; never infer batching from
    # the number of outstanding HTTP requests.
    props_request = urllib.request.Request(
        args.base_url.rstrip("/").removesuffix("/v1") + "/props",
        headers={"Authorization": f"Bearer {args.api_key}"},
    )
    with urllib.request.urlopen(props_request, timeout=30) as response:
        props = json.load(response)
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "generation_dir": str(args.generation_dir),
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "requests_per_trial": args.requests,
        "repeats": args.repeats,
        "server_slots": props.get("total_slots"),
        "server_props": props,
        "workload_sha256": hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest(),
        "limitations": [
            "Timing-only pilot: token-capped outputs are not accuracy evaluations.",
            "Workers measure outstanding requests; server slots determine actual parallel decoding.",
            "Identical saved prompts, seeds and sampling parameters are used in every trial.",
            "No retries, scoring or production trace writes are included in pilot timing.",
        ],
        "trials": [],
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output_dir / "requests.jsonl").open("w", buffering=1) as output:
        for repeat in range(args.repeats):
            # AB/BA reduces warm-cache and order effects between configurations.
            order = args.workers if repeat % 2 == 0 else list(reversed(args.workers))
            for workers in order:
                request_sample(workload[0], args, min(128, args.max_tokens))
                print(f"Starting repeat={repeat + 1}, workers={workers}, requests={len(workload)}", flush=True)
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    records = list(pool.map(lambda sample: request_sample(sample, args, args.max_tokens), workload))
                wall_seconds = time.perf_counter() - start
                for index, record in enumerate(records):
                    output.write(json.dumps({"repeat": repeat + 1, "workers": workers, "index": index, **record}) + "\n")
                trial = summarize_trial(records, wall_seconds, workers=workers, repeat=repeat + 1)
                summary["trials"].append(trial)
                with (args.output_dir / "summary.json").open("w") as handle:
                    json.dump(summary, handle, indent=2)
                print(json.dumps(trial), flush=True)
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["status"] = "complete"
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()

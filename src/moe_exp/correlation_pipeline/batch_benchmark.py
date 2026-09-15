"""Batch/MTP throughput sweep harness.

Runs a fixed sentence slice through the real batched judge path across a matrix of
(batch size x client concurrency x MTP), emitting one scoreboard row per cell and
applying a selection rule that was committed to before any measurement was taken.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from moe_exp.correlation_pipeline.batch_judge import build_batch_payload, map_batch_response, post_batch
from moe_exp.correlation_pipeline.batch_predictor import parse_completion, render_conversation

# Committed before measurement. Do not weaken after seeing the scoreboard.
SELECTION_RULE = (
    "Among cells with failures==0 and http_errors==0, latency_p95_ms <= 2x the batch-1 "
    "baseline p95, and gpu_memory_used_mb <= 90% of 65536 MiB, pick the highest "
    "aggregate_tokens_per_second. Require >10% gain over the batch-1 baseline or keep "
    "single-request mode. Ties within 5% prefer the smaller batch, then lower concurrency."
)

GPU_MEMORY_LIMIT_MB = int(0.90 * 65536)
MIN_GAIN_OVER_BASELINE = 1.10
TIE_BAND = 0.05
MAX_P95_REGRESSION = 2.0


def build_cells(
    *,
    batch_sizes: list[int],
    concurrencies: list[int],
    mtp_modes: list[bool],
) -> list[dict[str, Any]]:
    """Enumerate sweep cells ordered so one server can serve consecutive cells.

    Ordering by (mtp, max_num_seqs) means a server restart is only needed when one of
    those two server-level settings changes.
    """
    cells: list[dict[str, Any]] = []
    for mtp in mtp_modes:
        for max_num_seqs in sorted({16 if b <= 16 else 32 for b in batch_sizes}):
            for batch_size in sorted(batch_sizes):
                if (16 if batch_size <= 16 else 32) != max_num_seqs:
                    continue
                for concurrency in sorted(concurrencies):
                    cells.append(
                        {
                            "batch_size": batch_size,
                            "concurrency": concurrency,
                            "mtp": mtp,
                            "max_num_seqs": max_num_seqs,
                        }
                    )
    return cells


def latency_percentiles(latencies_ms: list[float]) -> tuple[float, float]:
    """Return (p50, p95) using nearest-rank so small samples stay honest."""
    if not latencies_ms:
        raise ValueError("no latencies recorded")
    ordered = sorted(latencies_ms)
    def rank(pct: float) -> float:
        index = max(0, min(len(ordered) - 1, int(round(pct * len(ordered) + 0.5)) - 1))
        return ordered[index]
    return rank(0.50), rank(0.95)


def summarize_cell(
    *,
    cell: dict[str, Any],
    latencies_ms: list[float],
    sentences: int,
    completion_tokens: int,
    elapsed_seconds: float,
    failures: int,
    http_errors: int,
    gpu_memory_used_mb: int,
    gpu_utilization_pct: float,
) -> dict[str, Any]:
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    p50, p95 = latency_percentiles(latencies_ms)
    return {
        **cell,
        "sentences": sentences,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed_seconds,
        "sentences_per_second": sentences / elapsed_seconds,
        "aggregate_tokens_per_second": completion_tokens / elapsed_seconds,
        "batch_generations_per_second": len(latencies_ms) / elapsed_seconds,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "failures": failures,
        "http_errors": http_errors,
        "gpu_memory_used_mb": gpu_memory_used_mb,
        "gpu_utilization_pct": gpu_utilization_pct,
    }


def select_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Apply SELECTION_RULE. Returns None when single-request mode should be kept."""
    baselines = [r for r in rows if r["batch_size"] == 1]
    if not baselines:
        raise ValueError("scoreboard has no batch-1 baseline")
    baseline = max(baselines, key=lambda r: r["aggregate_tokens_per_second"])

    safe = [
        r
        for r in rows
        if r["failures"] == 0
        and r["http_errors"] == 0
        and r["latency_p95_ms"] <= MAX_P95_REGRESSION * baseline["latency_p95_ms"]
        and r["gpu_memory_used_mb"] <= GPU_MEMORY_LIMIT_MB
        and r["batch_size"] != 1
    ]
    qualified = [
        r
        for r in safe
        if r["aggregate_tokens_per_second"]
        >= MIN_GAIN_OVER_BASELINE * baseline["aggregate_tokens_per_second"]
    ]
    if not qualified:
        return None
    best = max(qualified, key=lambda r: r["aggregate_tokens_per_second"])
    band = [
        r
        for r in qualified
        if r["aggregate_tokens_per_second"]
        >= (1.0 - TIE_BAND) * best["aggregate_tokens_per_second"]
    ]
    return min(band, key=lambda r: (r["batch_size"], r["concurrency"]))


def sample_gpu() -> tuple[int, float]:
    """Sample GPU memory (MiB) and utilization (%) summed/averaged over visible GPUs."""
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return 0, 0.0
    memory = 0
    utilizations: list[float] = []
    for line in output.splitlines():
        used, util = (part.strip() for part in line.split(","))
        memory += int(used)
        utilizations.append(float(util))
    return memory, statistics.fmean(utilizations) if utilizations else 0.0


def run_cell(
    cell: dict[str, Any],
    *,
    items: list[dict[str, Any]],
    adapter: Any,
    predict: Any,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Run the fixed slice through one cell and return its scoreboard row plus labels."""
    batch_size = cell["batch_size"]
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    conversations = [
        [render_conversation(adapter, predict, item) for item in batch] for batch in batches
    ]

    latencies_ms: list[float] = []
    failures = 0
    http_errors = 0
    completion_tokens = 0
    labels: dict[int, str] = {}

    def run_batch(batch_index: int) -> None:
        nonlocal failures, http_errors, completion_tokens
        payload = build_batch_payload(
            conversations[batch_index],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        started = time.perf_counter()
        try:
            response = post_batch(payload, base_url=base_url, api_key=api_key)
        except Exception:  # noqa: BLE001 - any transport error is a cell failure
            http_errors += 1
            return
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        usage = response.get("usage") or {}
        completion_tokens += int(usage.get("completion_tokens") or 0)
        try:
            mapped = map_batch_response(response, expected=len(conversations[batch_index]))
        except ValueError:
            failures += 1
            return
        offset = batch_index * batch_size
        for position, completion in mapped.items():
            try:
                labels[offset + position] = parse_completion(adapter, predict, completion)
            except Exception:  # noqa: BLE001 - unparseable label is a correctness failure
                failures += 1

    started = time.perf_counter()
    if cell["concurrency"] == 1:
        for index in range(len(batches)):
            run_batch(index)
    else:
        with ThreadPoolExecutor(max_workers=cell["concurrency"]) as pool:
            list(pool.map(run_batch, range(len(batches))))
    elapsed = time.perf_counter() - started

    memory, utilization = sample_gpu()
    row = summarize_cell(
        cell=cell,
        latencies_ms=latencies_ms or [float("inf")],
        sentences=len(labels),
        completion_tokens=completion_tokens,
        elapsed_seconds=elapsed,
        failures=failures,
        http_errors=http_errors,
        gpu_memory_used_mb=memory,
        gpu_utilization_pct=utilization,
    )
    return {"row": row, "labels": labels}


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

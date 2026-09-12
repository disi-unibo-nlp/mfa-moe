"""Match a serial llama.cpp timing log to saved shards and summarize actual costs.

The strict sequence/token checks deliberately reject mixed, overwritten or
parallel runs instead of silently attributing their timings to the wrong data.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


_TIMESTAMP = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)")
_TASK = re.compile(r"\| task (\d+) \|")
_TIMING = re.compile(r"\|\s+(prompt eval|eval|total) time =\s*([\d.]+) ms /\s*(\d+) tokens")


def read_timings(path: Path) -> list[dict[str, Any]]:
    active: dict[int, dict[str, Any]] = {}
    completed = []
    slots = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            slot_match = re.search(r"n_slots = (\d+)", line)
            if slot_match:
                slots.add(int(slot_match[1]))
            stamp, task_match = _TIMESTAMP.search(line), _TASK.search(line)
            if stamp is None or task_match is None:
                continue
            minutes, seconds, milliseconds, microseconds = map(int, stamp.groups())
            elapsed = minutes * 60 + seconds + milliseconds / 1000 + microseconds / 1e6
            task = active.setdefault(int(task_match[1]), {"task_id": int(task_match[1])})
            if "processing task," in line:
                task["start_seconds"] = elapsed
            timing = _TIMING.search(line)
            if timing:
                kind, ms, tokens = timing.groups()
                key = {"prompt eval": "prompt", "eval": "decode", "total": "total"}[kind]
                task[f"{key}_seconds"] = float(ms) / 1000
                task[f"{key}_tokens"] = int(tokens)
                if key == "total":
                    task["end_seconds"] = elapsed
                    completed.append(task)
    if slots != {1}:
        raise ValueError(f"Need an explicitly single-slot server log; found slots={slots}")
    return completed


def audit(generation_dir: Path, server_log: Path) -> dict[str, Any]:
    timings = read_timings(server_log)
    shards = []
    for path in generation_dir.glob("*/generation_shards/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        metadata = record["metadata"]
        shards.append({
            "dataset": record["dataset"],
            "problem_id": record["problem_id"],
            "completion_tokens": metadata["usage"]["completion_tokens"],
            "finish_reason": metadata["finish_reason"],
            "mtime": path.stat().st_mtime,
        })
    shards.sort(key=lambda shard: shard["mtime"])
    if len(shards) != len(timings) or not shards:
        raise ValueError(f"Cannot align {len(shards)} shards with {len(timings)} completed log requests")
    offsets = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard, timing in zip(shards, timings):
        if shard["completion_tokens"] != timing["decode_tokens"]:
            raise ValueError(f"Token mismatch at {shard['dataset']}/{shard['problem_id']}")
        offsets.append(shard["mtime"] - timing["end_seconds"])
        by_dataset[shard["dataset"]].append({**shard, **timing})
    # Files are saved just after their completion. Large shifts mean timestamps
    # were changed or a different server run is being matched.
    if max(offsets) - min(offsets) > 30:
        raise ValueError("Shard/log timestamp offsets differ by more than 30 seconds")
    datasets = []
    for dataset, rows in by_dataset.items():
        tokens = sum(row["completion_tokens"] for row in rows)
        decode = sum(row["decode_seconds"] for row in rows)
        elapsed = rows[-1]["end_seconds"] - rows[0]["start_seconds"]
        manifest_path = generation_dir / dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        datasets.append({
            "dataset": dataset,
            "status": manifest.get("status", "partial_no_manifest"),
            "traces": len(rows),
            "samples_per_problem": manifest.get("samples_per_problem"),
            "completion_tokens": tokens,
            "mean_completion_tokens": tokens / len(rows),
            "length_limited_traces": sum(row["finish_reason"] == "length" for row in rows),
            "decode_seconds": decode,
            "prompt_seconds": sum(row["prompt_seconds"] for row in rows),
            "observed_wall_seconds": elapsed,
            "decode_tokens_per_second": tokens / decode,
            "wall_tokens_per_second": tokens / elapsed,
        })
    total_tokens = sum(row["completion_tokens"] for row in datasets)
    total_decode = sum(row["decode_seconds"] for row in datasets)
    total_wall = timings[-1]["end_seconds"] - timings[0]["start_seconds"]
    return {
        "generation_dir": str(generation_dir),
        "server_log": str(server_log),
        "alignment": {
            "method": "Serial completion order matched to shard mtime order; every token count verified",
            "matched_requests": len(shards),
            "timestamp_offset_range_seconds": max(offsets) - min(offsets),
            "median_server_start_unix_seconds": statistics.median(offsets),
        },
        "totals": {
            "traces": len(shards),
            "completion_tokens": total_tokens,
            "decode_seconds": total_decode,
            "observed_wall_seconds": total_wall,
            "decode_tokens_per_second": total_tokens / total_decode,
            "wall_tokens_per_second": total_tokens / total_wall,
        },
        "datasets": sorted(datasets, key=lambda row: row["observed_wall_seconds"], reverse=True),
        "limitations": [
            "Wall time starts at the first request and ends at the last completed request.",
            "Model loading, final interrupted requests and forward extraction are excluded.",
            "Partial datasets are included and explicitly marked; their cost is not a full-dataset estimate.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.generation_dir, args.server_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

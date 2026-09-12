from __future__ import annotations

import io
import json
import os
import threading
import time
from collections import Counter

import pytest

from moe_exp.correlation_pipeline import profile_generation as profile
from moe_exp.correlation_pipeline.audit_generation_performance import audit


def test_comparison_preserves_samples_seeds_and_limits_concurrency(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs" / "aime24"
    inputs.mkdir(parents=True)
    records = [
        {
            "dataset": "aime24", "problem_id": f"question__sample_{seed:02d}",
            "generation_messages": [{"role": "user", "content": "A saved question"}],
            "metadata": {"generation_config": {
                "served_model": "test", "seed": seed,
                "temperature": 0.6, "top_p": 0.95, "top_k": 0,
            }},
        }
        for seed in range(32)
    ]
    (inputs / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    monkeypatch.setattr(profile.urllib.request, "urlopen", lambda *a, **kw: io.StringIO('{"total_slots": 1}'))
    active = 0
    maximum = 0
    calls = []
    lock = threading.Lock()

    def request(sample, args, max_tokens):
        nonlocal active, maximum
        if max_tokens == 128:  # Untimed warmup.
            return {}
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append(sample["seed"])
        time.sleep(0.002)
        with lock:
            active -= 1
        return {"dataset": sample["dataset"], "seed": sample["seed"],
                "completion_tokens": max_tokens, "wall_seconds": 1, "finish_reason": "length"}

    monkeypatch.setattr(profile, "request_sample", request)
    output = tmp_path / "profile"
    profile.main(["--generation-dir", str(inputs.parent), "--output-dir", str(output)])
    summary = json.loads((output / "summary.json").read_text())
    saved = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert summary["status"] == "complete"
    assert summary["server_slots"] == 1
    assert [row["workers"] for row in summary["trials"]] == [1, 2, 2, 1]
    assert [row["requests"] for row in summary["trials"]] == [32] * 4
    assert maximum == 2
    assert Counter(calls) == Counter({seed: 4 for seed in range(32)})
    for repeat, workers in [(1, 1), (1, 2), (2, 2), (2, 1)]:
        assert [row["seed"] for row in saved if (row["repeat"], row["workers"]) == (repeat, workers)] == list(range(32))


def test_throughput_uses_wall_time_instead_of_overlapping_latencies():
    records = [
        {"completion_tokens": 100, "wall_seconds": 2, "finish_reason": "stop", "dataset": "test"},
        {"completion_tokens": 100, "wall_seconds": 2, "finish_reason": "length", "dataset": "test"},
    ]
    summary = profile.summarize_trial(records, 2, workers=2, repeat=1)
    assert summary["completion_tokens_per_second"] == 100
    assert summary["length_limited_requests"] == 1


def test_historical_audit_matches_counts_and_marks_partial_datasets(tmp_path):
    inputs = tmp_path / "generation"
    shard_dir = inputs / "partial" / "generation_shards"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "sample.json"
    shard = {"dataset": "partial", "problem_id": "sample", "metadata": {
        "usage": {"completion_tokens": 200}, "finish_reason": "stop",
    }}
    shard_path.write_text(json.dumps(shard))
    os.utime(shard_path, (1003, 1003))
    log = tmp_path / "server.log"
    log.write_text(
        "0.00.000.000 I srv load_model: n_slots = 1\n"
        "0.01.000.000 I slot launch_slot_: id 0 | task 0 | processing task, is_child = 0\n"
        "0.03.000.000 I slot print_timing: id 0 | task 0 | prompt eval time = 100 ms / 10 tokens\n"
        "0.03.000.001 I slot print_timing: id 0 | task 0 | eval time = 1900 ms / 200 tokens\n"
        "0.03.000.002 I slot print_timing: id 0 | task 0 | total time = 2000 ms / 210 tokens\n"
    )
    result = audit(inputs, log)
    assert result["alignment"]["matched_requests"] == 1
    assert result["totals"]["decode_seconds"] == 1.9
    assert result["datasets"][0]["status"] == "partial_no_manifest"
    shard["metadata"]["usage"]["completion_tokens"] = 201
    shard_path.write_text(json.dumps(shard))
    with pytest.raises(ValueError, match="Token mismatch"):
        audit(inputs, log)

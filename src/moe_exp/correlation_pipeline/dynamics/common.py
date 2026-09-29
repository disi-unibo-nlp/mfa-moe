"""Portable contracts for offline analysis and forward reduction."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from . import SCHEMA_VERSION

CLASSES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")
DEFAULT_CONFIG = {"windows": [64, 256], "stride": 64, "lags": [1, 4, 16, 64],
                  "seed": 42, "shuffle_replicates": 16, "min_problems": 10}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({"schema_version": SCHEMA_VERSION, **row},
                                    allow_nan=False) + "\n")
    os.replace(tmp, path)


def identity(trace):
    return [trace.model_id, trace.dataset, trace.problem_id, trace.sample_id]


def keys(trace):
    return {"trace_id": digest(identity(trace)), "model": trace.model_id,
            "dataset": trace.dataset, "problem_id": trace.problem_id,
            "sample_id": trace.sample_id,
            "question_id": digest([trace.dataset, (trace.source_problem_id or trace.problem_id)]), "is_correct": trace.is_correct}


def termination(trace):
    meta = trace.metadata
    finish = meta.get("finish_reason")
    cap = meta.get("generation_config", {}).get("max_tokens")
    count = meta.get("usage", {}).get("completion_tokens")
    if finish == "length" or (count is not None and cap is not None and count >= cap):
        return "generation_cap"
    if "</think>" in trace.cot_text or "<|channel|>final" in trace.cot_text or finish == "stop":
        return "natural_termination"
    return "unknown_termination"


def artifact_path(bundle, stored):
    if not stored:
        return None
    stored = Path(stored)
    local = Path(bundle).parent / "tensors" / stored.name
    return local if local.is_file() else stored if stored.is_file() else None

"""Paths, model specs, constants, and outcome-free attempt-table loading of the path readout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from store import ANALYSIS

ROOT = ANALYSIS
V3R2 = ROOT / "v3_analysis" / "results-r2"
DYN_RESULTS = ROOT / "dynamics-routing" / "results"      # saved sentence tables, read-only
TOKEN_CACHE = ROOT / "depth-paths" / "results" / "s3" / "_tok"   # token ids and top-1 paths per attempt, read-only
STEER = ROOT / "steering-v1"
SPLIT_PATH = STEER / "manifests/split-v1.json"
SPLIT_SEAL = "b14fe2405ce7c3eefdc0be7a4c58787c109d47a2ad48e77d598c1c398ac91731"
FAST = Path("/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3")

MODELS = ("gemma", "glm", "gpt", "nemotron", "qwen330b", "qwen35", "qwen36")
CLASSES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")

ALPHA = 0.5                                   # add-alpha smoothing of the depth bigram (path surprise)

# Columns of the attempts table that carry no outcome information.
OUTCOME_FREE_COLUMNS = ("attempt_id", "question", "dataset", "problem_id", "source_problem_id",
                        "sample_id", "trace_sha256", "completion_tokens", "n_reasoning", "capped",
                        "termination", "tensor_path", "source_location", "cohort", "labelled_units", "labelled_tokens",
                        "sentence_units")
OUTCOME_COLUMNS = ("correct", "adjudication_status", "difficulty", "difficulty_n", "difficulty_source")


def digest(value) -> str:
    """Canonical JSON sha256."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_split() -> dict:
    """The steering question split (seal verified) -> {'dataset|source_problem_id': split}."""
    d = json.loads(SPLIT_PATH.read_text())
    seal = d.pop("sha256")
    if seal != SPLIT_SEAL or digest(d) != seal:
        raise ValueError("question split seal mismatch")
    return {f"{q['dataset']}|{q['source_problem_id']}": q["split"] for q in d["questions"]}


def keep_mask(model: str, questions) -> np.ndarray:
    """qwen36 keeps steering dev + tune questions only; every other model keeps everything."""
    questions = np.asarray(questions)
    if model != "qwen36":
        return np.ones(len(questions), bool)
    split = load_split()
    missing = sorted(set(questions) - set(split))
    if missing:
        raise ValueError(f"questions absent from the question split: {missing[:3]}")
    return np.array([split[q] in ("dev", "tune") for q in questions])


def model_spec(model: str, cohort: str = "A") -> dict:
    """Layers / experts / top-k of the capture (identical for the two cohorts of a model)."""
    ready = json.loads((FAST / "manifests" / model / cohort / "ready.json").read_text())["spec"]
    return dict(num_experts=int(ready["experts"]), num_layers=len(ready["layers"]),
                top_k=int(ready["top_k"]))


def load_attempts(model: str, cohort: str) -> pd.DataFrame:
    """Attempts table of the capture (outcome columns are never read), restricted to the analysed questions.

    Sorted by attempt_id, index reset: row i is attempt index i in every downstream table.
    """
    path = V3R2 / model / cohort / "attempts.parquet"
    names = pq.ParquetFile(path).schema.names
    cols = [c for c in OUTCOME_FREE_COLUMNS if c in names]
    assert not set(cols) & set(OUTCOME_COLUMNS)
    att = pd.read_parquet(path, columns=cols)
    att = att[keep_mask(model, att["question"].to_numpy())]
    return att.sort_values("attempt_id").reset_index(drop=True)

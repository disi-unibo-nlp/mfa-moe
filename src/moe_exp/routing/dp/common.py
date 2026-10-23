"""Paths, model specs, frozen constants, and outcome-free attempt-table loading."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

W = Path(__file__).resolve().parents[1]
ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
V3 = ROOT / "v3_analysis"
V3R2 = V3 / "results-r2"
DYN_RESULTS = ROOT / "dynamics-routing" / "results"      # read-only (sentences.parquet)
STEER = ROOT / "steering-v1"
SPLIT_PATH = STEER / "manifests/split-v1.json"
SPLIT_SEAL = "b14fe2405ce7c3eefdc0be7a4c58787c109d47a2ad48e77d598c1c398ac91731"
FAST = Path("/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3")
RESULTS = W / "results"

MODELS = ("gemma", "glm", "gpt", "nemotron", "qwen330b", "qwen35", "qwen36")
CLASSES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")
EXPLORE, VERIFY = CLASSES.index("Explore"), CLASSES.index("Verify")

# Frozen analysis constants
ALPHA = 0.5                                   # add-alpha smoothing of the depth bigram (F1)
WIN_EDGES = (1024, 2048, 4096, 8192, 16384)   # windows 0-1k, 1-2k, 2-4k, 4-8k, 8-16k (+ rest >= 16k)
N_WIN = len(WIN_EDGES) + 1                    # window index N_WIN-1 is the rest (excluded from variants)
N_WIN_FIXED = len(WIN_EDGES)
P1_ATTEMPTS = 200
P1_TOKENS = 4096
P1_SEED = 0
CLASS_MIN_TOKENS = 32                         # per-class F1: minimum class tokens in an attempt
CLASS_MIN_SENTENCES = 3                       # per-class F1: minimum labelled sentences of the class

# Columns of the v3an attempts table that carry no outcome information.
OUTCOME_FREE_COLUMNS = ("attempt_id", "question", "dataset", "problem_id", "source_problem_id",
                        "sample_id", "trace_sha256", "completion_tokens", "n_reasoning", "capped",
                        "termination", "tensor_path", "source_location", "cohort", "labelled_units", "labelled_tokens",
                        "sentence_units")
OUTCOME_COLUMNS = ("correct", "adjudication_status", "difficulty", "difficulty_n", "difficulty_source")


def digest(value) -> str:
    """Canonical JSON sha256."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def clean(o):
    """JSON-safe copy (NaN -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load_split() -> dict:
    """steering-v1 split-v1 (seal verified) -> {'dataset|source_problem_id': split}."""
    d = json.loads(SPLIT_PATH.read_text())
    seal = d.pop("sha256")
    if seal != SPLIT_SEAL or digest(d) != seal:
        raise ValueError("split-v1 seal mismatch")
    return {f"{q['dataset']}|{q['source_problem_id']}": q["split"] for q in d["questions"]}


def keep_mask(model: str, questions) -> np.ndarray:
    """qwen36 keeps steering dev + tune questions only; every other model keeps everything."""
    questions = np.asarray(questions)
    if model != "qwen36":
        return np.ones(len(questions), bool)
    split = load_split()
    missing = sorted(set(questions) - set(split))
    if missing:
        raise ValueError(f"questions absent from split-v1: {missing[:3]}")
    return np.array([split[q] in ("dev", "tune") for q in questions])


def model_spec(model: str, cohort: str = "A") -> dict:
    """Layers / experts / top-k of the capture (identical for the two cohorts of a model)."""
    ready = json.loads((FAST / "manifests" / model / cohort / "ready.json").read_text())["spec"]
    return dict(num_experts=int(ready["experts"]), num_layers=len(ready["layers"]),
                top_k=int(ready["top_k"]))


def load_attempts(model: str, cohort: str) -> pd.DataFrame:
    """v3an attempts table (outcome columns are never read), restricted to the analysed questions.

    Sorted by attempt_id, index reset: row i is attempt index i in every downstream table.
    """
    path = V3R2 / model / cohort / "attempts.parquet"
    names = pq.ParquetFile(path).schema.names
    cols = [c for c in OUTCOME_FREE_COLUMNS if c in names]
    assert not set(cols) & set(OUTCOME_COLUMNS)
    att = pd.read_parquet(path, columns=cols)
    att = att[keep_mask(model, att["question"].to_numpy())]
    return att.sort_values("attempt_id").reset_index(drop=True)


def load_outcomes(model: str, cohort: str, attempts: pd.DataFrame) -> pd.DataFrame:
    """Correctness and difficulty aligned to `attempts` (called only by the post-freeze analysis)."""
    att = pd.read_parquet(V3R2 / model / cohort / "attempts.parquet",
                          columns=["attempt_id", *OUTCOME_COLUMNS])
    att = att.set_index("attempt_id").reindex(attempts["attempt_id"])
    return att[list(OUTCOME_COLUMNS)].reset_index(drop=True)


def load_static(model: str, cohort: str, attempts: pd.DataFrame) -> pd.DataFrame:
    """v3an saved_router layer-mean summaries (`saved_router|all|<metric>|mean`), attempt-aligned."""
    path = V3R2 / model / cohort / "features.parquet"
    names = pq.ParquetFile(path).schema.names
    cols = [c for c in names if c.startswith("saved_router|all|") and c.endswith("|mean")]
    x = pd.read_parquet(path, columns=cols)
    x = x.reindex(attempts["attempt_id"]).reset_index(drop=True)
    x.columns = [c.split("|")[2] for c in cols]
    return x


def load_sentence_table(model: str, cohort: str) -> pd.DataFrame:
    """Labelled-sentence alignment (dynamics-routing, read-only): reasoning-rank spans and class."""
    cols = ["attempt_id", "sentence_index", "cls", "tok_start", "tok_end", "n_tokens", "n_reasoning",
            "abs_bin"]
    return pd.read_parquet(DYN_RESULTS / model / cohort / "sentences.parquet", columns=cols)


def class_of_rank(sent_rows: pd.DataFrame, n_reasoning: int) -> np.ndarray:
    """Class index per reasoning rank for one attempt (-1 = unlabelled); spans are contiguous."""
    out = np.full(n_reasoning, -1, np.int8)
    for s, e, c in zip(sent_rows["tok_start"].to_numpy(), sent_rows["tok_end"].to_numpy(),
                       sent_rows["cls"].to_numpy()):
        out[int(s):int(e)] = int(c)
    return out

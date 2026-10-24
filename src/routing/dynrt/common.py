"""Paths and constants of the class-routing features (all inputs read-only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from store import ANALYSIS

ROOT = ANALYSIS
V3R2 = ROOT / "v3_analysis" / "results-r2"
STEER = ROOT / "steering-v1"
SPLIT_PATH = STEER / "manifests/split-v1.json"
SPLIT_SEAL = "b14fe2405ce7c3eefdc0be7a4c58787c109d47a2ad48e77d598c1c398ac91731"
RESULTS = ROOT / "dynamics-routing" / "results"      # saved sentence tables, class histograms and pair tables

CLASSES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")
N_REL = 5
KAPPA = 50.0
EPS = 1e-6


def digest(value) -> str:
    """Canonical JSON sha256."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rel_bin(rel) -> np.ndarray:
    """Relative-position quintile (fixed cuts 0.2, 0.4, 0.6, 0.8)."""
    return np.minimum((np.asarray(rel, float) * N_REL).astype(np.int64), N_REL - 1).astype(np.int8)


def load_split() -> dict:
    """The steering question split (seal verified) -> {'dataset|source_problem_id': split}."""
    d = json.loads(SPLIT_PATH.read_text())
    seal = d.pop("sha256")
    if seal != SPLIT_SEAL or digest(d) != seal:
        raise ValueError("question split seal mismatch")
    return {f"{q['dataset']}|{q['source_problem_id']}": q["split"] for q in d["questions"]}


def keep_mask(model: str, questions) -> np.ndarray:
    """qwen36 keeps steering dev + tune questions only; gpt keeps everything."""
    questions = np.asarray(questions)
    if model != "qwen36":
        return np.ones(len(questions), bool)
    split = load_split()
    missing = sorted(set(questions) - set(split))
    if missing:
        raise ValueError(f"questions absent from the question split: {missing[:3]}")
    return np.array([split[q] in ("dev", "tune") for q in questions])


def cohort_dir(model: str, cohort: str, root: Path | None = None) -> Path:
    return (root or RESULTS) / model / cohort

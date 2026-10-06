"""Paths, model specs and shared constants (all inputs read-only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24")
V3 = ROOT / "v3_analysis"
V3R2 = V3 / "results-r2"
CAMPAIGN = ROOT / "campaign-v3"
STEER = ROOT / "steering-v1"
SPLIT_PATH = STEER / "manifests/split-v1.json"
SPLIT_SEAL = "b14fe2405ce7c3eefdc0be7a4c58787c109d47a2ad48e77d598c1c398ac91731"
FAST = Path("/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-v3")
# Read the existing, unchanged outcome-blind A/B feature caches. Clean R3-E
# outputs use a separate versioned directory set by d3c_freeze/d3c_cv.
RESULTS = ROOT / "dynamics-routing/results"

CLASSES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")
EXPLORE = CLASSES.index("Explore")
ABS_EDGES = (1024, 2048, 4096, 8192, 16384)
N_ABS = len(ABS_EDGES) + 1
N_REL = 5
KAPPA = 50.0
EPS = 1e-6
CELL_SHRINK = 20.0
MODELS = ("gpt", "qwen36")


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


def abs_bin(position) -> np.ndarray:
    """Absolute reasoning-token position bin: 0-1k, 1-2k, 2-4k, 4-8k, 8-16k, 16k+ (1k = 1024)."""
    return np.searchsorted(np.asarray(ABS_EDGES), np.asarray(position), side="right").astype(np.int8)


def rel_bin(rel) -> np.ndarray:
    """Relative-position quintile (fixed cuts 0.2, 0.4, 0.6, 0.8)."""
    return np.minimum((np.asarray(rel, float) * N_REL).astype(np.int64), N_REL - 1).astype(np.int8)


def load_split() -> dict:
    """steering-v1 split-v1 (seal verified) -> {'dataset|source_problem_id': split}."""
    d = json.loads(SPLIT_PATH.read_text())
    seal = d.pop("sha256")
    if seal != SPLIT_SEAL or digest(d) != seal:
        raise ValueError("split-v1 seal mismatch")
    return {f"{q['dataset']}|{q['source_problem_id']}": q["split"] for q in d["questions"]}


def keep_mask(model: str, questions) -> np.ndarray:
    """qwen36 keeps steering dev + tune questions only; gpt keeps everything."""
    questions = np.asarray(questions)
    if model != "qwen36":
        return np.ones(len(questions), bool)
    split = load_split()
    missing = sorted(set(questions) - set(split))
    if missing:
        raise ValueError(f"questions absent from split-v1: {missing[:3]}")
    return np.array([split[q] in ("dev", "tune") for q in questions])


def cohort_dir(model: str, cohort: str, root: Path | None = None) -> Path:
    return (root or RESULTS) / model / cohort

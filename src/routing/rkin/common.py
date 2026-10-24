"""Paths, constants and provenance helpers (all inputs read-only; outputs only under rk2/)."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from store import ANALYSIS

RK2 = Path(__file__).resolve().parents[1]
ROOT = ANALYSIS
DYN = ROOT / "dynamics-routing"
DYN_RESULTS = DYN / "results"
V3R2 = ROOT / "v3_analysis" / "results-r2"
DATA = RK2 / "data"
OUT = RK2 / "out"
PREREG = RK2 / "PREREG.md"
PREREG_SEAL = RK2 / "PREREG.sha256"

CLASSES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")
NC = len(CLASSES)
EXPLORE, VERIFY, MONITOR = CLASSES.index("Explore"), CLASSES.index("Verify"), CLASSES.index("Monitor")
ABS_EDGES = (1024, 2048, 4096, 8192, 16384)
N_ABS = len(ABS_EDGES) + 1
GRID = 32                   # window-velocity / acceleration grid step


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def verify_prereg() -> dict:
    """The operational pre-registration must be unchanged since its seal."""
    seal = json.loads(PREREG_SEAL.read_text())
    now = sha256_file(PREREG)
    if now != seal["sha256"]:
        raise RuntimeError("PREREG.md changed after its seal")
    return dict(prereg_sha256=now, sealed_at=seal["sealed_at"])


def code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {p.name: sha256_file(p) for p in sorted(here.glob("*.py"))}


def dep_hashes() -> dict:
    """Hashes of the read-only dynamics-routing / v3an modules RK2 imports."""
    files = [DYN / "dynrt" / n for n in ("common.py", "cv.py", "controls.py", "data.py", "features.py",
                                         "d3a_cv.py", "d3a_eval.py", "d3a_softmax.py", "d3a_text.py",
                                         "d3a_pairs.py")]
    files += [ROOT / "v3_analysis" / "v3an" / n for n in ("data.py", "extract.py", "stats.py")]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def abs_bin(position) -> np.ndarray:
    return np.searchsorted(np.asarray(ABS_EDGES), np.asarray(position), side="right").astype(np.int64)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=1, allow_nan=False))
    tmp.rename(path)

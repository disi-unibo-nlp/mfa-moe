"""D3b shared helpers: JSON-safe conversion, atomic JSON writes, provenance of the code."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .common import W, sha256_file


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


def write_json(path: Path, value) -> None:
    """Write JSON atomically; refuses paths outside the workspace."""
    path = Path(path)
    if not str(path.resolve()).startswith(str(W)):
        raise ValueError("outputs must stay under the dynamics-routing workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(clean(value), indent=1, allow_nan=False))
    os.replace(tmp, path)


def code_hashes() -> dict:
    """sha256 of every d3b_* module (recorded next to each result)."""
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).resolve().parent.glob("d3b_*.py"))}

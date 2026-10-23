"""Sealed campaign receipts: normalised digests, immutable publication and artifact files.

Vendored verbatim from `expert_memory_study.contracts` and `research_campaign_v3_r6.{numeric,contracts,artifacts}`
for the saved-data analyses (bootstrap, nominations, trajectories, selection coverage). The campaign layer's
normalising `digest`/`canonical`/`seal` wrap the raw ones, as they did there; on JSON-native values both agree.
`moe_exp.io` holds the study-side primitives that predate this module; step 6 of the cleanup merges the two.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import uuid

WORK = Path("/leonardo_work/IscrC_MIOSR/lmolfett")
HOME_ROOT = Path("/leonardo/home/userexternal/lmolfett")
FAST_PRIVATE = Path('/leonardo_scratch/fast/IscrC_MIOSR/lmolfett/mfa-moe/campaign-runs')
CLASSES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")
SEED = 20261008


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(value, reasons=None, path=""):
    if isinstance(value, Mapping):
        return {str(k): normalize(v, reasons, path+"/"+str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v, reasons, path+"/"+str(i)) for i, v in enumerate(value)]
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        if reasons is not None: reasons[path] = "missing_input"
        return None
    if isinstance(value, numbers.Real):
        if not math.isfinite(value):
            if reasons is not None: reasons[path] = "non_finite_input"
            return None
        return value.item() if hasattr(value, "item") else value
    if hasattr(value, "tolist"):
        return normalize(value.tolist(), reasons, path)
    return value


def records(rows):
    output = []
    for row in rows:
        reasons = {}
        clean = normalize(row, reasons)
        if reasons:
            clean["unavailable_reasons"] = {**clean.get("unavailable_reasons", {}), **reasons}
        output.append(clean)
    return output


def read_parquet(path):
    import pandas as pd
    return records(pd.read_parquet(path).to_dict("records"))


def finite(value):
    return isinstance(value, numbers.Real) and math.isfinite(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _seal(value):
    payload = {k: v for k, v in value.items() if k != "sha256"}
    return {**payload, "sha256": _digest(payload)}


def verify(value):
    require(value.get("sha256") == _seal(value)["sha256"], "receipt content seal differs")
    return value


def load(path):
    return verify(json.loads(Path(path).read_text()))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_files(files):
    for name, expected in files.items():
        require(file_sha(name) == expected, f"bound source changed: {name}")


def _private_path(path):
    path = Path(path).resolve()
    require(any(path.is_relative_to(root) for root in (WORK, HOME_ROOT)),
            "write outside private HOME/WORK")
    return path


def private_path(path):
    resolved = Path(path).resolve()
    if resolved.is_relative_to(FAST_PRIVATE):
        return resolved
    return _private_path(path)


def digest(value):
    return _digest(normalize(value))


def canonical(value):
    return _canonical(normalize(value))


def seal(value):
    return _seal(normalize(value))


@contextmanager
def atomic_file(path):
    path = private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(file_sha(temporary) == file_sha(path), "conflicting immutable data artifact")
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def npz(path, **arrays):
    import numpy as np
    with atomic_file(path) as stream:
        np.savez_compressed(stream, **arrays)
    return {"path": str(path), "sha256": file_sha(path), "bytes": Path(path).stat().st_size}


def frame(path, rows):
    import pandas as pd
    data = pd.DataFrame(rows)
    with atomic_file(path) as stream:
        if str(path).endswith(".parquet"):
            data.to_parquet(stream, index=False)
        else:
            stream.write(data.to_csv(index=False).encode())
    return {"path": str(path), "sha256": file_sha(path), "rows": len(data)}


def publish(path, value):
    """Idempotent publication accepts only the identical already committed value."""
    path = private_path(path)
    # Normalize tuples and integer dictionary keys before sealing. Readback is
    # JSON, and its canonical ordering must be exactly the committed ordering.
    value = seal(json.loads(json.dumps(normalize(value),ensure_ascii=False,allow_nan=False)))
    if path.exists():
        require(load(path) == value, f"immutable publication conflict: {path}")
        return value
    try:
        with atomic_file(path) as stream:
            stream.write(canonical(value)+b"\n")
        require(load(path)==value,"immutable publication readback differs")
        return value
    except FileExistsError:
        require(load(path) == value, f"concurrent publication conflict: {path}")
        return value

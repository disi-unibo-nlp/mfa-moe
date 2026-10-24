"""Small immutable artifacts and explicit private write boundaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

PRIVATE_ROOTS = (Path("/leonardo_work/IscrC_MIOSR/lmolfett"),
                 Path("/leonardo/home/userexternal/lmolfett"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_value(value):
    if isinstance(value, dict):
        require(all(type(k) in (str,int) for k in value), "unsupported JSON map key")
        require(len({str(k) for k in value})==len(value), "JSON key collision")
        return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_value(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal(value):
    body = json_value({k: v for k, v in value.items() if k != "sha256"})
    return {**body, "sha256": digest(body)}


def verify(value, schema=None):
    require(isinstance(value, dict) and value.get("sha256") ==
            digest({k: v for k, v in value.items() if k != "sha256"}), "artifact seal differs")
    require(schema is None or value.get("schema") == schema, "artifact schema differs")
    return value


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def private_path(path):
    path = Path(path).resolve()
    require(any(path.is_relative_to(root) for root in PRIVATE_ROOTS),
            "write outside approved user HOME/WORK roots")
    return path


def save(path, value, *, replace=False):
    path = private_path(path)
    value = seal(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = load(path)
        if old == value:
            return value
        require(replace, "immutable artifact already exists: " + str(path))
    fd, temporary = tempfile.mkstemp(prefix=".artifact-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return value


def load(path, schema=None):
    return verify(json.loads(Path(path).read_text()), schema)


def check_files(files):
    require(files and all(Path(p).is_file() and file_sha(p) == sha
                         for p, sha in files.items()), "bound source changed or unavailable")


def require_generation_binding(receipt, campaign, model):
    require(receipt.get("schema") == "verification-generation-receipt-v3" and
            receipt.get("model_sha256") == model["sha256"] and
            receipt.get("runtime_sha256") == campaign["runtime_sha256"] and
            receipt.get("executor_sha256") == digest(campaign["code_files"]),
            "generation model/runtime/executor binding differs")


def identity(*parts):
    return digest(["candidate-verification-v3", *parts])

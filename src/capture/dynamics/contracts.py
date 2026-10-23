"""Sealed JSON artifacts used by the continuation branches: a kind, its inputs, config, population and code, hashed together."""
from __future__ import annotations

import json
from pathlib import Path

from capture.dynamics.common import digest, write_json
from provenance import code_provenance

VERSION = 1
KINDS = {"branches", "analysis"}


def seal(kind, payload, *, inputs, config, population, code=None):
    if kind not in KINDS or not population or not inputs:
        raise ValueError("An artifact needs a known kind, inputs and population")
    result = dict(schema_version=VERSION, kind=kind, inputs=inputs, config=config,
                  population=population, code=code or code_provenance(), payload=payload)
    return {**result, "binding": digest(result)}


def validate(artifact, kind=None):
    if artifact.get("schema_version") != VERSION or artifact.get("kind") not in KINDS:
        raise ValueError("Unsupported steering artifact")
    if kind is not None and artifact["kind"] != kind:
        raise ValueError("Wrong artifact kind")
    if artifact.get("binding") != digest({k: v for k, v in artifact.items() if k != "binding"}):
        raise ValueError("Artifact binding mismatch")
    return artifact


def save(path, artifact):
    """Idempotent writes; a changed input/config/code requires a fresh destination."""
    validate(artifact)
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != artifact:
        raise ValueError(f"Refusing incompatible artifact overwrite: {path}")
    write_json(path, artifact)
    if json.loads(path.read_text()) != artifact:
        raise OSError("Artifact readback failed")


def load(path, kind=None):
    return validate(json.loads(Path(path).read_text()), kind)

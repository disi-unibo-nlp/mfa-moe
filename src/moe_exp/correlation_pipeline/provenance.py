"""Reproducibility and integrity metadata without importing the tensor stack."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def code_provenance():
    root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    hasher = hashlib.sha256()
    for path in sorted((root / "src/moe_exp").rglob("*.py")):
        hasher.update(path.relative_to(root).as_posix().encode())
        hasher.update(path.read_bytes())
    return {"git_revision": revision, "source_sha256": hasher.hexdigest()}


def forward_provenance(model, tokenizer, *, revision=None, local_files_only=False):
    config = model.config
    model_revision = getattr(config, "_commit_hash", None)
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    # Local pinned snapshots do not always retain _commit_hash in AutoConfig.
    for obj, field in ((config, "_name_or_path"), (tokenizer, "name_or_path")):
        path = Path(str(getattr(obj, field, "")))
        if path.parent.name == "snapshots":
            if obj is config:
                model_revision = model_revision or path.name
            else:
                tokenizer_revision = tokenizer_revision or path.name
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "tokenizers", "numpy", "scipy", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    fingerprint = None
    if hasattr(tokenizer, "get_vocab"):
        from moe_exp.models.token_replay import tokenizer_fingerprint
        fingerprint = tokenizer_fingerprint(tokenizer)
    return {
        "requested_revision": revision,
        "resolved_model_revision": model_revision,
        "resolved_tokenizer_revision": tokenizer_revision,
        "tokenizer_sha256": fingerprint,
        "local_files_only": local_files_only,
        "quantization": getattr(model, "_replay_quantization", None),
        "code": code_provenance(), "environment": versions,
    }

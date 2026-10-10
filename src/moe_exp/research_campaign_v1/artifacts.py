"""Atomic data artifacts and finite per-cell exports."""
from __future__ import annotations

from contextlib import contextmanager
import os
import uuid
from pathlib import Path

from .contracts import file_sha, private_path, require


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

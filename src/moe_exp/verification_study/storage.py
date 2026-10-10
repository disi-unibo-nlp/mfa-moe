"""Compact per-layer scientific capture; JSON receipts never embed large arrays."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time

from .contracts import digest, file_sha, private_path, require, save, seal


def commit_generation(directory, receipt):
    import numpy as np
    started=time.monotonic()
    directory = private_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    uid = receipt["assignment_uid"]
    require(not (directory / (uid + ".json")).exists(), "generation already committed")
    capture = dict(receipt["capture"])
    arrays = {}
    for layer, values in capture.pop("capture").items():
        for name, value in values.items():
            dtype = np.int16 if name.endswith("ids") else np.float32 if name.endswith("weights") else np.bool_ if name == "active" else np.int32
            arrays[f"L{layer}_{name}"] = np.asarray(value, dtype=dtype)
    path = directory / (uid + ".npz")
    fd, temporary = tempfile.mkstemp(prefix=".capture-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    capture.update(array_path=str(path), array_sha256=file_sha(path), array_bytes=path.stat().st_size,
                   archive_seconds=time.monotonic()-started)
    return save(directory / (uid + ".json"), {**receipt, "capture": capture})


def capture_arrays(receipt):
    import numpy as np
    capture = receipt["capture"]
    require(file_sha(capture["array_path"]) == capture["array_sha256"], "capture changed")
    with np.load(capture["array_path"], allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}

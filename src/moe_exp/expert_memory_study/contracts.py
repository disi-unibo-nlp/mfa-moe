"""Private paths, immutable receipts, source seals and allocation boundaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket

REPO = Path(__file__).resolve().parents[3]
WORK = Path("/leonardo_work/IscrC_MIOSR/lmolfett")
HOME_ROOT = Path("/leonardo/home/userexternal/lmolfett")
STAGING = WORK / "tools/tmp"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seal(value):
    payload = {k: v for k, v in value.items() if k != "sha256"}
    return {**payload, "sha256": digest(payload)}


def verify(value):
    require(value.get("sha256") == seal(value)["sha256"], "receipt content seal differs")
    return value


def load(path):
    return verify(json.loads(Path(path).read_text()))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def private_path(path):
    path = Path(path).resolve()
    require(any(path.is_relative_to(root) for root in (WORK, HOME_ROOT)),
            "write outside private HOME/WORK")
    return path


def save(path, value):
    """Publish atomically with no overwrite, including concurrent writers."""
    path = private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    value = seal(value)
    raw = canonical(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("xb") as f:
        os.chmod(temporary, 0o600)
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    require(load(path) == value, "receipt readback differs")
    return value


def check_files(files):
    for name, expected in files.items():
        require(file_sha(name) == expected, f"bound source changed: {name}")


def allocated(*, gpu=False):
    require(os.environ.get("SLURM_JOB_ID", "").isdigit() and
            os.environ.get("SLURM_STEP_ID") is not None,
            "scientific work requires a Slurm Python step")
    require(os.environ.get("USER") == "lmolfett" and "leonardo" in socket.getfqdn(),
            "wrong cluster/user")
    require(os.environ.get("SLURM_JOB_ACCOUNT", "").lower() == "iscrc_miosr",
            "unexpected allocation account")
    require(os.environ.get("SLURM_JOB_PARTITION") ==
            ("boost_usr_prod" if gpu else "lrd_all_serial"), "wrong allocation partition")
    if gpu:
        require(not socket.gethostname().startswith("login") and
                bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "GPU computation on login node")


def source_binding():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((REPO / "scripts/expert_memory_study").glob("*"))
    paths += [REPO / "src/moe_exp/verification_study/registry.py",
              REPO / "src/moe_exp/correlation_pipeline/model_profiles.py",
              REPO / "src/moe_exp/correlation_pipeline/scoring.py",
              REPO / "src/moe_exp/utils.py"]
    return {str(p): file_sha(p) for p in sorted(paths) if p.is_file()}

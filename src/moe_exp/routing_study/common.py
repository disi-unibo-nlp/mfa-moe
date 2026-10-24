"""Versioned, content-bound contracts shared by the four study commands."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

from moe_exp.moe_identity_guiding.execution import atomic_json, file_hash, run_lock

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"
BENCHMARKS = ("math500", "aime24", "aime25", "amc23", "minerva", "olympiad")
REPETITIONS = {name: (32 if name in ("aime24", "aime25", "amc23") else 1) for name in BENCHMARKS}
CHECKPOINT = dict(model=MODEL, revision=REVISION, quantization="fp8",
                  num_experts=256, top_k=8, layers=list(range(40)), shared_experts=1)
ENGINE = dict(tensor_parallel_size=2, max_num_seqs=8, enforce_eager=True,
              enable_prefix_caching=False, enable_expert_parallel=False,
              enable_eplb=False, speculative_config=None, async_scheduling=False,
              max_model_len=49152, gpu_memory_utilization=.90, generation_config="vllm",
              language_model_only=True, kernel_config={"moe_backend": "marlin"})
SAMPLING = dict(max_tokens=32768, temperature=.6, top_p=.95, top_k=-1)
ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOTS = (Path("/leonardo/home/userexternal/lmolfett"),
                 Path("/leonardo_work/IscrC_MIOSR/lmolfett"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal(value):
    if "sha256" in value:
        raise ValueError("Already sealed")
    return {**value, "sha256": digest(value)}


def verify(value):
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Content hash mismatch")
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path, value, *, immutable=False):
    path = private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}")
        return
    atomic_json(path, value)
    if read_json(path) != value:
        raise OSError("Read-back verification failed")


def private_path(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root) for root in PRIVATE_ROOTS):
        raise ValueError("Outputs must stay in lmolfett's HOME or WORK directory")
    return path


def code_hashes():
    paths = sorted((ROOT / "moe_exp/routing_study").glob("*.py"))
    paths += sorted((ROOT / "moe_exp/routing_study").glob("*.json"))
    paths += [ROOT / "moe_exp" / name for name in (
        "moe_identity_guiding/execution.py", "moe_identity_guiding/calibration.py",
        "moe_identity_guiding/routing.py", "moe_margin_guiding/calibration.py",
        "moe_margin_guiding/routing.py", "correlation_pipeline/scoring.py", "utils.py")]
    result = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    for name in ("pyproject.toml", "scripts/run_routing_study.py", "scripts/verify_routing_study.py",
                 "scripts/routing_study_env.sh", "sbatch/routing_study_smoke.sbatch"):
        result[name] = file_hash(ROOT.parent/name)
    return result


def require_compute():
    import getpass
    import socket
    if (getpass.getuser() != "lmolfett" or not socket.gethostname().startswith("lrdn")
            or not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID")):
        raise RuntimeError("Model work requires an authorized Slurm srun step on LEONARDO")


def bounded_input(paths, limit=64 * 1024**2):
    if sum(Path(p).stat().st_size for p in paths) > limit:
        require_compute()


def problem_key(row):
    key = row.get("source_problem_id") or row.get("problem_id")
    if not key or row["dataset"] not in BENCHMARKS:
        raise ValueError("Need a supported benchmark and problem identity")
    return json.dumps([row["dataset"], str(key)], separators=(",", ":"))


def attempt_key(row):
    return digest([row["dataset"], row["problem_id"], row.get("sample_id", 0)])


def checkpoint_ok(value):
    if value != CHECKPOINT:
        raise ValueError("Require the frozen Qwen3.6 FP8 checkpoint and native expert configuration")

"""Compatible short IPC Slurm transport; scientific r1 executors remain unchanged.

Cards carry a separate transport seal. The original card can be reconstructed
exactly, and its scientific executor, deployment, UIDs and prices remain intact.
Only this allocated process's cache environment and child submission environment
are changed. No scheduler operation is performed by this module.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import runpy
import socket
import sys
import tempfile
import time
import traceback

from moe_exp.verification_study.contracts import (
    check_files, digest, file_sha, load, private_path, require, save, seal, verify)

SCRIPT = Path(__file__).resolve()
STAGING = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
HF_ROOT = Path("/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/cache/hf")
SHELLS = {
    "extension": SCRIPT.with_name("verification_study_v3_extension_r3_gpu.sbatch"),
    "reader": SCRIPT.with_name("verification_study_v3_reader_r3.sbatch"),
    "native": SCRIPT.with_name("verification_study_v3_native_r3_gpu.sbatch"),
}
RUNTIME_CODES = {"extension": "e", "reader": "r", "native": "n", "controller": "c"}
IPC_PATH_MAX_BYTES = 107
IPC_UUID = "00000000-0000-0000-0000-000000000000"
WORKERS = {
    "extension": SCRIPT.with_name("verification_study_v3_extension_r1.py"),
    "reader": SCRIPT.with_name("verification_study_v3_reader_r1.py"),
    "native": SCRIPT.with_name("verification_study_v3_task.py"),
}
TRANSPORT_FIELDS = {"transport_executor", "transport_kind", "transport_parent_card_sha256"}
CACHE_VARIABLES = (
    "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
    "TRITON_CACHE_DIR", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH",
    "HF_HOME", "HF_HUB_CACHE", "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "VLLM_RPC_BASE_PATH",
)
SUBMISSION_PREFIXES = ("SLURM_", "SRUN_", "SBATCH_", "PMI_", "PMIX_")


def source_files():
    """New compatible closure for a controller program seal."""
    return {str(path): file_sha(path) for path in (SCRIPT, *SHELLS.values())}


def clean_submission_environment(mapping=None):
    """Return a child-only environment without inherited scheduler directives.

The current allocated process retains its Slurm identity. Callers pass this
copy to sbatch, including --test-only; no environment values are logged here.
"""
    source = os.environ if mapping is None else mapping
    return {key: value for key, value in source.items()
            if not key.startswith(SUBMISSION_PREFIXES)}


def binding(campaign, kind):
    verify(campaign)
    require(kind in SHELLS, "unknown transport kind")
    paths = (SCRIPT, SHELLS[kind], WORKERS[kind])
    return seal({"schema": "verification-transport-executor-v3", "kind": kind,
        "operational_version": "transport-r3", "ipc_path_max_bytes": IPC_PATH_MAX_BYTES,
        "helper": str(SCRIPT), "shell": str(SHELLS[kind]), "worker": str(WORKERS[kind]),
        "campaign_sha256": campaign["sha256"],
        "scientific_source_sha256": digest(campaign["code_files"]),
        "runtime_sha256": campaign["runtime_sha256"],
        "code_files": {str(path): file_sha(path) for path in paths},
        "compatibility": "cache paths and Slurm process transport only; unchanged scientific worker and deployment"})


def _validate_parent(campaign_path, campaign, card, kind):
    verify(card, "verification-job-card-v3")
    require(kind in SHELLS, "unknown transport kind")
    require(card["campaign_sha256"] == campaign["sha256"] and
            Path(card["campaign_path"]).resolve() == Path(campaign_path).resolve() and
            card["code_files"] == campaign["code_files"], "transport scientific source binding differs")
    check_files(campaign["code_files"])
    resources = card["resources"]
    require(resources["gpus"] == 2 and resources["cpus"] == 16 and
            resources["memory_gb"] == 120 and resources["partition"] == "boost_usr_prod",
            "transport requires original qualified 2 GPU / 16 CPU / 120 GB shape")
    if kind == "extension":
        require("j1_executor" in card and "extension_files" in card and
                card["extension"]["kind"] in {"j1-audit-slice", "grade-j1-slice"},
                "transport is not a GPU J1 extension")
        check_files(card["extension_files"])
        check_files(card["j1_executor"]["code_files"])
        require(card["j1_executor"]["campaign_sha256"] == campaign["sha256"] and
                card["j1_executor"]["runtime_sha256"] == campaign["runtime_sha256"] and
                card["j1_executor"]["reader_contract_sha256"] == campaign["reader_sha256"],
                "transport J1 deployment differs")
    elif kind == "reader":
        require("reader_executor" in card and "extension" not in card,
                "transport is not a semantic reader card")
        check_files(card["reader_executor"]["code_files"])
        require(card["reader_executor"]["campaign_sha256"] == campaign["sha256"] and
                card["reader_executor"]["runtime_sha256"] == campaign["runtime_sha256"] and
                card["reader_executor"]["reader_contract_sha256"] == campaign["reader_sha256"],
                "transport reader deployment differs")
    else:
        require("extension" not in card and "reader_executor" not in card and
                card["stage"] in {"run", "calibrate"}, "transport is not a native scientific card")


def wrap_card(campaign_path, parent_card, kind):
    """Create a new submission identity, retaining the old scientific card.

Existing wrapped cards are verified and returned unchanged. Original scientific
receipts deliberately retain the old reader/J1 executor seals, with a separate
source-bound transport invocation proof joined by actual job ID.
"""
    campaign = load(campaign_path)
    if "transport_executor" in parent_card:
        verify_card(campaign_path, parent_card, kind)
        return parent_card
    require(not (TRANSPORT_FIELDS & set(parent_card)), "partial transport card")
    _validate_parent(campaign_path, campaign, parent_card, kind)
    return seal({**parent_card, "transport_executor": binding(campaign, kind),
                 "transport_kind": kind, "transport_parent_card_sha256": parent_card["sha256"]})


def verify_card(campaign_path, card, kind):
    verify(card, "verification-job-card-v3")
    campaign = load(campaign_path)
    require(card.get("transport_kind") == kind and TRANSPORT_FIELDS <= set(card),
            "transport kind or parent binding missing")
    parent = seal({key: value for key, value in card.items() if key not in TRANSPORT_FIELDS})
    require(parent["sha256"] == card["transport_parent_card_sha256"],
            "transport changed the original scientific card")
    _validate_parent(campaign_path, campaign, parent, kind)
    check_files(card["transport_executor"]["code_files"])
    require(card["transport_executor"] == binding(campaign, kind), "transport source binding changed")
    return campaign, parent


def environment_proof(mapping=None):
    source = os.environ if mapping is None else mapping
    return {key: source.get(key) for key in CACHE_VARIABLES}


def require_step(card):
    require(getpass.getuser() == "lmolfett" and socket.gethostname().split(".", 1)[0].startswith("lrdn"),
            "transport requires lmolfett on a LEONARDO compute node")
    require(re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")) and
            os.environ.get("SLURM_STEP_ID") not in {None, "", "batch", "extern"},
            "transport requires an allocated scientific Slurm step")
    require(os.environ.get("SLURM_CLUSTER_NAME") == "leonardo" and
            os.environ.get("SLURM_JOB_ACCOUNT") == "iscrc_miosr" and
            os.environ.get("SLURM_JOB_PARTITION") == card["resources"]["partition"] and
            os.environ.get("SLURM_CPUS_PER_TASK") == "16", "transport allocated resource differs")


def runtime_path(kind, job_id, *, staging=STAGING):
    """Short, disjoint per-job namespace, including the full uint64 ID range."""
    require(kind in RUNTIME_CODES and re.fullmatch(r"[0-9]{1,20}", str(job_id)),
            "unsafe transport runtime identity")
    number = int(job_id)
    require(0 < number <= (1 << 64) - 1 and str(number) == str(job_id),
            "unsafe transport runtime identity")
    # Decimal stays readable for actual Slurm IDs. A marked hex token keeps
    # unusually large uint64 IDs disjoint and within the Unix socket limit.
    token = str(job_id) if len(str(job_id)) <= 18 else "h" + format(number, "x")
    requested = Path(staging) / "t3" / (RUNTIME_CODES[kind] + token)
    require(not requested.is_symlink(), "transport runtime is a symlink")
    base = private_path(requested)
    require(base.is_relative_to(STAGING), "transport runtime must be in private WORK staging")
    ipc_path_bound(base / "r")
    return base


def pin_runtime(kind, job_id, *, staging=STAGING):
    """Reassert caches after Slurm TaskProlog, before heavy Python imports."""
    base = runtime_path(kind, job_id, staging=staging)
    return pin_runtime_environment(base)


def ipc_path_bound(rpc_path):
    """vLLM network_utils appends '/' + str(uuid4()); Linux reserves NUL."""
    path = str(Path(rpc_path) / IPC_UUID)
    length = len(os.fsencode(path))
    require(length <= IPC_PATH_MAX_BYTES,
            f"vLLM IPC socket path exceeds {IPC_PATH_MAX_BYTES} bytes: {length}")
    return {"rpc_path": str(rpc_path), "uuid_suffix": IPC_UUID,
            "socket_path_bytes_max": length, "socket_path_bytes_limit": IPC_PATH_MAX_BYTES}


def pin_runtime_environment(runtime_path):
    """Pin a controller's or GPU worker's private WORK caches in this process."""
    require(not Path(runtime_path).is_symlink(), "transport runtime is a symlink")
    base = private_path(runtime_path)
    require(base.is_relative_to(Path("/leonardo_work/IscrC_MIOSR/lmolfett")),
            "transport runtime must be in private user WORK")
    # Assert the exact suffix before mkdir, environment changes, or imports.
    ipc_path_bound(base / "r")
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(base.stat().st_uid == os.getuid() and not base.stat().st_mode & 0o077,
            "transport runtime owner or private permissions differ")
    directories = {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "xdg",
        "XDG_CONFIG_HOME": "config", "TRITON_CACHE_DIR": "triton", "TORCH_HOME": "torch",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "CUDA_CACHE_PATH": "cuda",
        "VLLM_CACHE_ROOT": "vllm-cache", "VLLM_CONFIG_ROOT": "vllm-config", "VLLM_RPC_BASE_PATH": "r"}
    for variable, name in directories.items():
        path = base / name
        require(not path.is_symlink(), "transport cache is a symlink")
        path.mkdir(mode=0o700, exist_ok=True)
        require(path.stat().st_uid == os.getuid() and not path.stat().st_mode & 0o077,
                "transport cache owner or private permissions differ")
        os.environ[variable] = str(path)
    require(HF_ROOT.is_dir() and (HF_ROOT / "hub").is_dir(), "existing offline HF cache unavailable")
    os.environ["HF_HOME"] = str(HF_ROOT)
    os.environ["HF_HUB_CACHE"] = str(HF_ROOT / "hub")
    # Python may already have cached a node/site default. Pin this process too.
    tempfile.tempdir = os.environ["TMPDIR"]
    return base


def worker_arguments(card_path, kind, deadline):
    require(kind in WORKERS and deadline is not None and deadline > time.time(), "finite transport deadline missing")
    args = [str(WORKERS[kind])]
    if kind == "native":
        args.append(str(card_path))
    else:
        args.extend(("execute", "--job-card", str(card_path)))
    return [*args, "--deadline-epoch", str(deadline)]


def execute(card_path, kind, deadline):
    """Delegate to the unchanged real worker under a sealed transport card."""
    card_path = Path(card_path).resolve()
    card = load(card_path, "verification-job-card-v3")
    campaign, parent = verify_card(Path(card["campaign_path"]), card, kind)
    require_step(card)
    incoming = environment_proof()
    job_id = os.environ["SLURM_JOB_ID"]
    slot = private_path(Path(campaign["result_root"]) / "transport" / "invocations" /
                        (job_id + "-" + str(time.time_ns())))
    identity = {"job_id": job_id, "job_card_path": str(card_path), "job_card_sha256": card["sha256"],
        "parent_job_card_sha256": parent["sha256"], "campaign_sha256": campaign["sha256"],
        "transport_executor": card["transport_executor"], "kind": kind}
    save(slot / "ENTERING_STEP.json", {"schema": "verification-transport-step-environment-v3",
        **identity, "cache_environment": incoming, "hostname": socket.gethostname(), "user": getpass.getuser(),
        "step_id": os.environ["SLURM_STEP_ID"], "epoch": time.time()})
    previous_argv = sys.argv
    try:
        base = pin_runtime(kind, job_id)
        start = save(slot / "START.json", {"schema": "verification-transport-invocation-v3", **identity,
            "entering_step_proof_sha256": load(slot / "ENTERING_STEP.json")["sha256"],
            "runtime_path": str(base), "cache_environment": environment_proof(),
            "ipc_path_bound": ipc_path_bound(os.environ["VLLM_RPC_BASE_PATH"]),
            "python_tempdir": tempfile.gettempdir(), "started_epoch": time.time()})
        sys.argv = worker_arguments(card_path, kind, deadline)
        try:
            runpy.run_path(str(WORKERS[kind]), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in {None, 0}:
                raise
        return save(slot / "RESULT.json", {"schema": "verification-transport-result-v3", **identity,
            "status": "WORKER_RETURNED", "start_sha256": start["sha256"], "ended_epoch": time.time(),
            "scientific_success": "REQUIRES_ORIGINAL_RECEIPTS_AND_ACCOUNTING"})
    except BaseException as exc:
        save(slot / "ERROR.json", {"schema": "verification-transport-error-v3", **identity,
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "cache_environment": environment_proof(), "ended_epoch": time.time()})
        raise
    finally:
        sys.argv = previous_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute",))
    parser.add_argument("--job-card", type=Path, required=True)
    parser.add_argument("--kind", choices=tuple(SHELLS), required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    result = execute(args.job_card, args.kind, args.deadline_epoch)
    print(json.dumps({key: result[key] for key in ("schema", "sha256", "status", "job_id")}, indent=2))


if __name__ == "__main__":
    main()

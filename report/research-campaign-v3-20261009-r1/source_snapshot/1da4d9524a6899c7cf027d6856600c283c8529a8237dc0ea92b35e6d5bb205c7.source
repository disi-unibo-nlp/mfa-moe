"""Immutable execution identities and atomic, recoverable campaign storage."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

from moe_exp.expert_memory_study.contracts import (
    REPO, STAGING, allocated, canonical, check_files, digest, file_sha, load,
    private_path, require, save, seal, verify,
)
from .numeric import normalize
from moe_exp.expert_memory_study.contracts import digest as _digest, canonical as _canonical, seal as _seal


def digest(value):
    return _digest(normalize(value))


def canonical(value):
    return _canonical(normalize(value))


def seal(value):
    return _seal(normalize(value))

MODELS = ("gpt", "nemotron", "qwen35", "qwen36", "gemma", "glm", "qwen330b")
CLASSES = ("Read", "Analyze", "Plan", "Implement", "Explore", "Verify", "Monitor")
BENCHMARKS = ("aime24", "aime25", "amc23", "math500", "minerva", "olympiad")
CONFIGURATIONS = ("MMM", "LMM", "HMM", "MLM", "MHM", "MML", "MMH", "LLL", "HHH")
METHODS = ("state_informed", "reap", "all_reasoning", "matched_label", "random")
REMOVALS = (.125, .25, .5)
SEED = 20261008
CEILING = 42050.
MARGIN = 1.25
STORAGE_RESERVE = 10 * 1024**3
REPLICATE = "run-000"
STAGES = {
    "resolve": (), "derive": ("resolve","mapping"), "native":("derive",),"supplementary":("derive","mapping"),"coordination":("derive","mapping"),
    "mapping": ("resolve",), "analysis_finalize":("coordination","supplementary"),
    "measure": ("resolve", "derive","native","analysis_finalize"),
    "freeze": ("derive", "measure"), "qualify": ("freeze",),
    "price": ("qualify",), "admit": ("price","reader_price","analysis_finalize"), "execute": ("admit",),
    "reader_price":("price",),"contexts":("execute",),
    "annotate": ("contexts",), "score": ("execute",),"snapshot":("resolve",),
    "aggregate": ("annotate", "score"), "closeout": ("aggregate",),
}
TERMINAL = frozenset(("complete", "reused", "insufficient_support", "execution_failed",
                     "unresolved_submission", "resource_limited", "upstream_unavailable"))
EXECUTION_FIELDS = ("model", "profile_sha256", "source", "seed", "replicate",
                    "mask", "action", "cap", "penalty_history_sha256")


def execution_id(row):
    """Windows, plotting, threshold labels and arm names are analysis identities."""
    require(all(k in row for k in EXECUTION_FIELDS), "incomplete execution identity")
    from .masks import resolve
    return digest({k: resolve(row[k]) if k=="mask" else row[k] for k in EXECUTION_FIELDS})


def source_binding():
    from importlib.metadata import distribution
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((REPO / "scripts/research_campaign_v3").glob("*"))
    paths += list((REPO / "tests/research_campaign_v3").glob("*.py"))
    # Imported scientific/loader code stays frozen and is bound, never edited.
    paths += list((REPO / "src/moe_exp/expert_memory_study").glob("*.py"))
    paths += list((REPO / "src/moe_exp/verification_study").glob("*.py"))
    paths += [REPO / "src/moe_exp/correlation_pipeline" / n for n in (
        "model_profiles.py", "scoring.py", "spans.py", "batch_predictor.py", "batch_judge.py", "features.py", "schema.py")]
    for directory in ("models", "correlation_pipeline/dynamics", "gepaLLMAsJudge"):
        paths += list((REPO / "src/moe_exp" / directory).rglob("*.py"))
    paths += [REPO/"src/moe_exp"/n for n in ("schemas.py","utils.py")]
    paths += [REPO/"scripts/experimental_resume/routing_local_logits_v2.py"]
    for package in ('numpy','pandas','pyarrow','torch','vllm','triton','matplotlib'):
        dist=distribution(package)
        paths += [Path(dist.locate_file(p)) for p in dist.files or () if str(p).endswith('.dist-info/METADATA')]
    return {str(p): file_sha(p) for p in sorted(set(paths)) if p.is_file()}


@contextmanager
def locked(root, name="campaign"):
    root = private_path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / (digest(name) + ".lock")).open("a") as handle:
        os.chmod(handle.name, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


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
        from .artifacts import atomic_file
        with atomic_file(path) as stream:
            stream.write(canonical(value)+b"\n")
        require(load(path)==value,"immutable publication readback differs")
        return value
    except FileExistsError:
        require(load(path) == value, f"concurrent publication conflict: {path}")
        return value


class Units:
    """A started scientific unit is never redrawn after a lost response."""
    def __init__(self, root):
        self.root = private_path(root)

    def state(self, uid):
        require(len(uid) == 64 and set(uid) <= set("0123456789abcdef"), "invalid unit ID")
        result = self.root / f"{uid}.result.json"
        if result.exists():
            return load(result)
        from .resume import inherited_state
        inherited = inherited_state(self.root, uid)
        if inherited is not None:
            return inherited
        starts = sorted(self.root.glob(f"{uid}.start-*.json"))
        return {"status": "started", "starts": [load(p) for p in starts]} if starts else None

    def begin(self, row, job_id):
        uid = row["uid"]
        with locked(self.root, uid):
            state = self.state(uid)
            if state is not None:
                return False
            publish(self.root / f"{uid}.start-0.json", {
                "uid": uid, "row_sha256": digest(row), "job_id": str(job_id),
                "started_epoch": time.time(), "status": "started"})
            return True

    def commit(self, row, value):
        with locked(self.root, row["uid"]):
            require(self.state(row["uid"]) is not None, "unstarted unit commit")
            return publish(self.root / f"{row['uid']}.result.json", {
                "uid": row["uid"], "assignment": row, **value})


def protocol():
    return seal({
        "schema": "research-campaign-v3", "coordination_subpanels": {"linear_alpha": .025, "log_odds_alpha": .025, "shared_resamples": 10000, "undefined_draws_retained": True}, "models": MODELS, "classes": CLASSES,
        "benchmarks": BENCHMARKS, "configurations": CONFIGURATIONS,
        "methods": METHODS, "removals": REMOVALS, "allocations": ("equal", "unequal"),
        "seed": SEED, "replicate": REPLICATE, "bootstrap_resamples": 10000,
        "pruning_families_per_benchmark": 100, "causal_families_per_benchmark": 30,
        "fold_universe": 1486, "folds": 5, "weighting": "benchmark then family",
        "actual_pruning_roster":{"questions":420,"families":400},
        "actual_causal_roster":{"questions":131,"families":129},
        "unavailable_prefixes_replaced":False,"full_A_ownership_expands_enrollment":False,
        "component_reuse":"recursive source/population/profile/runtime/measurement bindings",
        "temporary_cache_lifetime":"one model active; retire after durable dependent evidence",
        "reap_quantity":"native pre-reduction contributions; separate unchanged-selection unit-weight invocation",
        "coordination_gaps": (1, 2, 4, 8), "inferential_pairs": 16,
        "extent": (.25, .5, .75), "strength": (.25, .5, .75),
        "pulse_quantiles": (.25, .5, .75), "window_multipliers": (4, 8, 16),
        "calibration_positions_per_attempt": 64, "ceiling_local_hours": CEILING,
        "reservation_margin": MARGIN, "storage_reserve_bytes": STORAGE_RESERVE,
        "milestone_utc": "2026-10-15T21:59:59Z", "deadline_is_compute_cutoff": False,
        "allowed_infrastructure_replacements": 1, "stages": STAGES,
        "corpus": "card-v3", "uncertainty": "whole-family conditional on frozen selections",
        "new_training": False, "online_policy": False, "exact_cross_run_replay": False,
        "authorization": "User: implement and launch the supplied plan; Implement the plan",
    })

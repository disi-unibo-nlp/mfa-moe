"""Versioned, content-bound artifacts shared by discovery and intervention.

Question identity deliberately excludes model, attempt, cohort and outcome. A
population is always explicit; B weights describe only its eligible population.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

from .common import digest, write_json
from ..provenance import code_provenance, file_sha256

VERSION = 1
KINDS = {"inventory", "features", "folds", "candidates", "branches", "analysis", "sizing"}


def question_id(dataset, source_problem_id):
    if not dataset or not source_problem_id:
        raise ValueError("Dataset and source question identity are required")
    return digest([dataset, source_problem_id])


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


def file_binding(path):
    path = Path(path).resolve()
    before = path.stat()
    sha = file_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed during hashing: {path}")
    return dict(path=str(path), sha256=sha, bytes=before.st_size)


def deduplicate(rows):
    """Keep all capture receipts and membership probabilities, one attempt per budget."""
    result = {}
    invariant = ("question_id", "trace_sha256", "completion_tokens", "model_revision",
                 "precision", "backend", "layers", "num_experts", "top_k", "is_correct",
                 "termination", "outcome_version")
    for row in rows:
        key = (row["model"], row["attempt_id"], row["budget_population"])
        if key not in result:
            result[key] = {**row, "memberships": [], "captures": []}
        previous = result[key]
        for field in invariant:
            if previous.get(field) != row.get(field):
                raise ValueError(f"Conflicting duplicate {key}: {field}")
        for field in ("memberships", "captures"):
            for value in row[field]:
                if value not in previous[field]:
                    if field == "memberships" and any(v["cohort"] == value["cohort"]
                                                     for v in previous[field]):
                        raise ValueError("Conflicting sampling membership")
                    previous[field].append(value)
    return [result[k] for k in sorted(result)]


def sampling_weight(row, cohort):
    member = next(m for m in row["memberships"] if m["cohort"] == cohort)
    field = "arm_inclusion_probability" if cohort == "B" else "inclusion_probability"
    p = member.get(field)
    if not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 < p <= 1:
        raise ValueError(f"Missing/invalid {cohort} sampling probability")
    if cohort == "B" and member.get("inclusion_probability") is not None:
        raise ValueError("B has no unconditional population inclusion probability")
    return 1 / p


def freeze_folds(rows, inventory_binding, *, seed=42, reserve_fraction=.2):
    """Global question assignment, including shared questions across all models.

    Stable hash ordering is reproducible without depending on row ordering. The
    reserve is exploratory: earlier aggregate analysis already inspected A.
    """
    if not 0 < reserve_fraction < 1:
        raise ValueError("Reserve fraction must lie strictly between zero and one")
    membership = defaultdict(set)
    for row in rows:
        membership[row["question_id"]].update(m["cohort"] for m in row["memberships"])
    a_only = [q for q, m in membership.items() if "A" in m and "B" not in m]
    order = lambda q: digest([seed, q])
    reserved = set(sorted(a_only, key=order)[:math.ceil(reserve_fraction * len(a_only))])
    development = sorted(set(membership) - reserved, key=order)
    outer = {q: i % 4 for i, q in enumerate(development)}
    inner = {}
    for f in range(4):
        eligible = sorted((q for q in development if outer[q] != f),
                          key=lambda q: digest([seed, f, q]))
        inner[str(f)] = {q: i % 3 for i, q in enumerate(eligible)}
    return seal("folds", dict(reserved_questions=sorted(reserved, key=order),
        outer=outer, inner=inner, b_question_union=sorted(q for q, m in membership.items() if "B" in m),
        reserve_status="exploratory_previously_inspected"), inputs={"inventory": inventory_binding},
        config=dict(seed=seed, outer_folds=4, inner_folds=3, reserve_fraction=reserve_fraction),
        population="global_question_union_original_budget")


def require_development(rows, folds):
    validate(folds, "folds")
    assigned = folds["payload"]["outer"]
    if any(r["question_id"] not in assigned for r in rows):
        raise ValueError("Reserved or unassigned question entered fitting")


def validate_feature(row):
    """Timing is an exclusive completion-token boundary, never a relative position."""
    endpoint = row["decision_token"]
    if type(endpoint) is not int or endpoint <= 0:
        raise ValueError("Invalid decision token")
    if row["budget_population"] != "original":
        raise ValueError("Prospective fixed-budget analysis requires original attempts")
    if row.get("feature_end_token", endpoint) > endpoint:
        raise ValueError("Future feature at a past decision")
    for w in row.get("routing_windows", []):
        if not 0 <= w["start_token"] < w["end_token"] <= endpoint:
            raise ValueError("Routing window extends past the decision")
    if row.get("destination_start_token") is not None and endpoint > row["destination_start_token"]:
        raise ValueError("Transition feature includes destination text")
    if row.get("state") and row.get("state_provenance") != "cross_fitted_prefix_only":
        raise ValueError("Retrospective state labels cannot enter deployable features")
    return row

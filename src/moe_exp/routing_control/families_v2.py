"""Bind discovery to the existing independent family reservation."""
from .study_v1 import require
from .study_v2 import seal, verify


def family_boundary(discovery_freeze, independent_freeze):
    verify(discovery_freeze)
    verify(independent_freeze)
    return validate_boundary(seal({"schema": "routing-family-boundary-v2",
        "discovery": discovery_freeze["new_parent_pools"]["parent_pools"]["discovery"],
        "validation": [r["family"] for r in independent_freeze["validation"]],
        "final": [r["family"] for r in independent_freeze["final"]],
        "discovery_freeze_sha256": discovery_freeze["sha256"],
        "independent_freeze_sha256": independent_freeze["sha256"],
        "source_revision": independent_freeze["source_revision"]}))


def validate_boundary(value):
    verify(value)
    pools = [value[k] for k in ("discovery", "validation", "final")]
    require(value.get("schema") == "routing-family-boundary-v2" and pools[0] and len(pools[1]) == 32 and
            len(pools[2]) == 96 and all(len(p) == len(set(p)) for p in pools) and
            not set(pools[0]) & set(pools[1] + pools[2]) and not set(pools[1]) & set(pools[2]) and
            value.get("source_revision") and value.get("discovery_freeze_sha256") and value.get("independent_freeze_sha256"),
            "family partition differs or independent family leakage")
    return value


def require_discovery(families, boundary):
    validate_boundary(boundary)
    require(set(families) <= set(boundary["discovery"]), "reserved/foreign family in discovery analysis")

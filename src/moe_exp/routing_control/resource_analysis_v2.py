"""Exact-contract measured stage pricing, including storage and bounded recovery."""
from __future__ import annotations

import math

from .study_v1 import require
from .study_v2 import CAP, _execution, seal, verify

ROLES = ("generation", "controller", "replay", "annotation", "grading", "capture", "runtime", "recovery")


def price_stage(manifest, workloads, telemetry):
    verify(manifest)
    _execution(manifest["execution"])
    require(set(workloads) == set(ROLES), "all resource roles must be explicitly itemized")
    require(workloads["generation"].get("output_tokens") == manifest["assigned"] * CAP,
            "price must cover every assigned 16k continuation")
    require(workloads["recovery"].get("maximum_assignment_retries") is not None and
            type(workloads["recovery"]["maximum_assignment_retries"]) is int and
            workloads["recovery"]["maximum_assignment_retries"] >= 0, "bounded recovery allowance missing")
    rows, missing = [], []
    for role in ROLES:
        work = workloads[role]
        fields = ("input_tokens", "output_tokens", "loads", "calls")
        require(all(type(work.get(k)) is int and work[k] >= 0 for k in fields), "invalid measured-role workload")
        if not any(work[k] for k in fields):
            require(work.get("no_work_reason"), "zero-work role needs an explicit reason")
            rows.append({"role": role, "status": "NO_WORK_FROZEN", "seconds": 0., "GPU_hours": 0.,
                         "billing_core_hours": 0., "workload": work})
            continue
        measured = telemetry.get(role)
        if measured is None:
            missing.append(role)
            continue
        verify(measured)
        keys = ("prefill_tps", "decode_tps", "load_seconds", "shutdown_seconds", "seconds_per_call", "gpus", "cpus", "billing_cores")
        require(measured.get("status") == "MEASURED" and measured.get("execution_contract_sha256") == manifest["execution"]["sha256"] and
                measured.get("measurement_artifact_sha256") and
                all(isinstance(measured.get(k), (int, float)) and not isinstance(measured[k], bool) and
                    math.isfinite(measured[k]) and measured[k] >= 0 for k in keys), "pilot contract or measured telemetry differs")
        require(type(measured["cpus"]) is int and measured["cpus"] >= 1 and
                measured["billing_cores"] >= measured["cpus"], "measured CPU/billing allocation required")
        require(not work["input_tokens"] or measured["prefill_tps"] > 0, "missing prefill measurement")
        require(not work["output_tokens"] or measured["decode_tps"] > 0, "missing decode measurement")
        if role in ("generation", "replay", "recovery"):
            require(measured.get("continuation_cap") == CAP, "short-horizon pilot cannot price 16k generation")
        seconds = ((work["input_tokens"] / measured["prefill_tps"] if work["input_tokens"] else 0) +
                   (work["output_tokens"] / measured["decode_tps"] if work["output_tokens"] else 0) +
                   work["loads"] * (measured["load_seconds"] + measured["shutdown_seconds"]) +
                   work["calls"] * measured["seconds_per_call"])
        rows.append({"role": role, "status": "PRICED", "seconds": seconds, "GPU_hours": seconds * measured["gpus"] / 3600,
            "billing_core_hours": seconds * measured["billing_cores"] / 3600,
            "workload": work, "measurement_sha256": measured["sha256"],
            "cost_ledger": "policy" if role in ("generation", "controller") else "research"})
    retries = workloads["recovery"]["maximum_assignment_retries"]
    require(workloads["recovery"]["output_tokens"] >= retries * CAP,
            "recovery output allowance cannot cover its assignment retries")
    # Both TP ranks retain native and executed arrays, including failed whole
    # assignments. Dose, token cursors and original/frozen inputs are additional.
    storage = (manifest["assigned"] + retries) * CAP * (2 * 2 * 40 * 8 * (2 + 4) + 40 * 2 * 4 + 4)
    prefix_bytes = workloads["capture"].get("prefix_storage_bytes", 0)
    require(type(prefix_bytes) is int and prefix_bytes >= 0, 'invalid original/frozen prefix storage allowance')
    storage += prefix_bytes
    require(type(workloads["capture"].get("metadata_storage_bytes")) is int and
            workloads["capture"]["metadata_storage_bytes"] >= 0, "capture metadata storage allowance required")
    storage += workloads["capture"]["metadata_storage_bytes"]
    return seal({"schema": "routing-study-complete-price-v2", "stage": manifest["stage"],
        "manifest_sha256": manifest["sha256"], "execution_contract_sha256": manifest["execution"]["sha256"],
        "workers": rows, "unavailable_roles": missing, "capture_storage_bytes": storage,
        "maximum_GPU_hours": None if missing else sum(r["GPU_hours"] for r in rows),
        "maximum_billing_core_hours": None if missing else sum(r["billing_core_hours"] for r in rows),
        "complete_seconds": None if missing else sum(r["seconds"] for r in rows),
        "controller_calls_bound": workloads["controller"]["calls"],
        "policy_output_token_bound": workloads["generation"]["output_tokens"] + workloads["controller"]["output_tokens"],
        "status": "UNAVAILABLE_MEASURED_PRICE" if missing else "PRICED"})

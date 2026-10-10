"""Measured role costs, finite reservations and complete-allocation feasibility."""
from __future__ import annotations

import math

from .contracts import require, seal
from .protocol import ARMS, MODELS


def final_plan(campaign, roster, freeze, prices, availability, *, completed_uids=()):
    """Price every necessary canonical draw before releasing any final card."""
    require(roster["stage"] == "final" and roster["campaign_sha256"] == campaign["sha256"] and
            roster["frozen_masks_sha256"] == freeze["sha256"] and
            set(prices) == {(m, a) for m in MODELS for a in ARMS}, "incomplete final sources/prices")
    identities = {(r["family"], r["model"], r["arm"]) for r in roster["records"]}
    require(len(identities) == len(roster["records"]) == roster["families"] * 28 and
            len({r["family"] for r in roster["records"]}) == roster["families"], "final paired allocation is incomplete")
    require(availability.get("complete") is True and availability.get("source_files") and
            availability.get("reviewed_epoch") and availability.get("parallel_gpu_jobs", 0) > 0,
            "complete observed availability required")
    completed = set(completed_uids)
    require(completed <= {r["uid"] for r in roster["records"]}, "foreign completed assignment")
    shards, counts, expected_bytes = [], {}, 0
    for m in MODELS:
        for a in ARMS:
            key, price = (m, a), prices[m, a]
            require(price["campaign_sha256"] == campaign["sha256"] and price["model"] == m and
                    price["arm"] == a and price["context"] == 40960,
                    "foreign role/context price")
            pending = [r for r in roster["records"] if r["model"] == m and r["arm"] == a and
                       r["alias_of_arm"] == a and r["uid"] not in completed]
            counts[key] = len(pending)
            if not pending:
                continue
            fit = math.floor((28800 / 1.25 - price["load_seconds"]) / price["worst_cap_assignment_seconds"])
            require(fit > 0, "one cap-bound assignment does not fit eight-hour shard")
            # Include conservative cap-sized text/token payload and compact metadata;
            # never infer disk capacity from a short retained completion alone.
            expected_bytes += len(pending) * max(1024**2, price.get("assignment_artifact_bytes") or 0)
            for offset in range(0, len(pending), fit):
                rows = pending[offset:offset + fit]
                shards.append({"model": m, "arm": a, "assignment_uids": [r["uid"] for r in rows],
                               **shard_price(price, len(rows))})
    assessment = feasible(prices, counts, remaining_core_hours=availability["remaining_core_hours"],
                          available_storage_bytes=availability["available_storage_bytes"],
                          expected_storage_bytes=expected_bytes,
                          remaining_seconds=availability["remaining_seconds"],
                          parallel_gpu_jobs=availability["parallel_gpu_jobs"])
    exact_core_hours = sum(s["billing_core_hours_upper"] for s in shards)
    reasons = list(assessment["reasons"])
    if exact_core_hours > availability["remaining_core_hours"] and "ENTITLEMENT_INFEASIBLE" not in reasons:
        reasons.append("ENTITLEMENT_INFEASIBLE")
    assessment = {**assessment, "billing_core_hours_upper": exact_core_hours,
                  "reasons": reasons, "status": "INFEASIBLE" if reasons else "NEEDS_SCHEDULER_CONFIRMATION"}
    return seal({**assessment, "campaign_sha256": campaign["sha256"],
                 "assignments_sha256": roster["sha256"], "frozen_masks_sha256": freeze["sha256"],
                 "availability_sha256": availability["sha256"],
                 "prices": {f"{m}/{a}": p["sha256"] for (m, a), p in prices.items()},
                 "shards": shards, "expected_storage_bytes_upper": expected_bytes,
                 "remaining_canonical_draws": sum(counts.values()), "completed_uids": sorted(completed),
                 "N": roster["families"], "source_files": availability["source_files"]})


def approve_final_plan(plan, review):
    require(plan["status"] == "NEEDS_SCHEDULER_CONFIRMATION" and
            review.get("approved_plan_sha256") == plan["sha256"] and review.get("user_quote") and
            review.get("complete_allocation_authorized") is True and
            review.get("scheduler_capacity_confirmed") is True and review.get("deadline_confirmed") is True and
            review.get("reserved_core_hours", -1) >= plan["billing_core_hours_upper"] and
            review.get("reserved_storage_bytes", -1) >= plan["expected_storage_bytes_upper"],
            "complete measured allocation requires exact user authorization and scheduler/deadline confirmation")
    return seal({**plan, "status": "FEASIBLE_APPROVED", "dispatch_authorized": True,
                 "unapproved_plan_sha256": plan["sha256"], "review_sha256": review["sha256"]})


def price_generation(result, *, model, arm, context):
    require(result["execution_complete"] is True and result["generated_tokens"] > 0 and
            result["context"] == context and result["output_cap"] == 32768 and
            result["closeout_seconds"] >= 0, "invalid retained role/context measurement")
    n = result["generated_tokens"]
    require(n > 1 and result["decode_seconds"] > 0, "decode role unmeasured")
    return seal({"schema": "expert-memory-role-price-v1", "model": model, "arm": arm,
                 "context": context, "result_sha256": result["sha256"],
                 "assignment_uid": result["assignment"]["uid"],
                 "load_seconds": result["load_seconds"], "prefill_seconds": result["prefill_seconds"],
                 "decode_seconds_per_token": result["decode_seconds"] / (n - 1),
                 "closeout_seconds": result["closeout_seconds"],
                 "worst_cap_assignment_seconds": result["prefill_seconds"] +
                 result["decode_seconds"] / (n - 1) * 32767 + result["closeout_seconds"],
                 "measurement_tokens": n, "retained_in_pilot": True,
                 "assignment_artifact_bytes": result.get("artifact_bytes"),
                 "scope": "one original largest-input pilot assignment, conservative extrapolation to output cap"})


def shard_price(price, assignments, *, limit_seconds=28800):
    require(assignments > 0 and limit_seconds <= 28800, "invalid finite shard")
    seconds = math.ceil(1.25 * (price["load_seconds"] + assignments * price["worst_cap_assignment_seconds"]))
    require(seconds <= limit_seconds, "priced shard exceeds finite walltime; use fewer assignments")
    return {"assignments": assignments, "reservation_seconds": seconds,
            "billing_core_hours_upper": seconds * 16 / 3600, "gpu_hours_upper": seconds * 2 / 3600,
            "margin": .25, "price_sha256": price["sha256"]}


def feasible(prices, counts, *, remaining_core_hours, available_storage_bytes, expected_storage_bytes,
             remaining_seconds, parallel_gpu_jobs):
    require(set(prices) == {(m, a) for m in MODELS for a in ARMS}, "all-seven/all-arm prices required")
    seconds, allocation = {}, 0.
    for key, price in prices.items():
        count = counts[key]
        per = price["worst_cap_assignment_seconds"]
        fit = math.floor((28800 / 1.25 - price["load_seconds"]) / per)
        require(fit > 0, "one cap-bound assignment does not fit eight-hour shard")
        loads = math.ceil(count / fit) if count else 0
        seconds[key] = 1.25 * (loads * price["load_seconds"] + count * per)
        allocation += seconds[key] * 16 / 3600
    makespan_lower = sum(seconds.values()) / max(1, parallel_gpu_jobs)
    reasons = []
    if allocation > remaining_core_hours:
        reasons.append("ENTITLEMENT_INFEASIBLE")
    if expected_storage_bytes > available_storage_bytes:
        reasons.append("STORAGE_INFEASIBLE")
    if makespan_lower > remaining_seconds:
        reasons.append("DEADLINE_INFEASIBLE")
    return seal({"schema": "expert-memory-feasibility-v1", "status": "INFEASIBLE" if reasons else "NEEDS_SCHEDULER_CONFIRMATION",
                 "reasons": reasons, "billing_core_hours_upper": allocation,
                 "deadline_seconds_lower": makespan_lower,
                 "dispatch_authorized": False, "precision_sample_may_not_shrink": True})

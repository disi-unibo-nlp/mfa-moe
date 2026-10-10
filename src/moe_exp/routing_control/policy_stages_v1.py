"""Prospective routing-only policy stages, qualified prefix triggers, and prices.

These are execution contracts, not a qualification result. A recognizer and
GPU operator/capture implementation must each have a frozen passing audit
before a manifest can be dispatched. No future labels or answer grades enter
the online controller interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .prefix import PrefixInput
from .study_v1 import TRANSITIONS, digest, require


@dataclass(frozen=True)
class RoutingAction:
    action_id: str
    transition: str
    layer: int
    expert: int
    operator: str
    strength: float
    delay: int = 0
    pulse_tokens: int = 256

    def __post_init__(self):
        require(self.transition in TRANSITIONS and self.action_id, "unknown action or transition")
        require(type(self.layer) is int and 0 <= self.layer < 40 and
                type(self.expert) is int and 0 <= self.expert < 256, "invalid layer-specific expert")
        require(self.operator in ("bias", "force", "reweight") and math.isfinite(self.strength), "invalid operator")
        require(self.operator != "force" or self.strength == 0, "forced inclusion has no logit strength")
        require(self.operator != "reweight" or self.strength > 0, "policy amplification requires a positive log-weight change")
        require(self.operator != "bias" or self.strength != 0, "zero bias is native, not an action")
        require(self.delay in (0, 64) and self.pulse_tokens == 256, "unfrozen timing or pulse length")

    def worker_action(self):
        return {"name": self.action_id, "transition": self.transition,
                "experts": [[self.layer, [self.expert]]], "kind": self.operator,
                "sign": -1 if self.strength < 0 else 1, "magnitude": abs(self.strength)}


class PrefixPolicyController:
    """One request's append-only state; a delayed no-fire stays in its ledger.

    The recognizer sees only the problem, already emitted IDs and their decoded
    text. Its audit must certify substantive acts and complete/checked status.
    Text supplied by a caller, offline labels, answers and future IDs are not
    read. Controller token/input/runtime expenditure is required on every call.
    """

    def __init__(self, actions, recognizer, recognition_audit, *, ordered=False):
        require(1 <= len(actions) <= 2 and len({a.action_id for a in actions}) == len(actions),
                "policy permits at most two distinct actions")
        require(recognition_audit.get("status") == "PASS" and recognition_audit.get("prefix_only") is True and
                recognition_audit.get("recognizes_substantive_acts") is True and recognition_audit.get("sha256"),
                "prefix recognizer lacks a passing semantic audit")
        require(all(a.transition in recognition_audit["supported_transitions"] for a in actions),
                "unsupported prefix transition")
        require(not ordered or len(actions) == 2 and actions[0].transition != actions[1].transition,
                "ordered composition needs two distinct substantive actions")
        self.actions, self.recognizer, self.audit = tuple(actions), recognizer, recognition_audit
        self.ordered, self.previous, self.closed = ordered, None, False
        self.opportunities, self.decisions, self.calls = {}, {}, []

    def observe(self, record, decode):
        prefix = PrefixInput.from_record({"problem": record["problem"],
            "emitted_token_ids": record["emitted_token_ids"], "emitted_text": decode(record["emitted_token_ids"])})
        if self.previous is not None:
            require(prefix.problem == self.previous.problem and
                    prefix.emitted_token_ids[:len(self.previous.emitted_token_ids)] == self.previous.emitted_token_ids and
                    prefix.emitted_text.startswith(self.previous.emitted_text), "online prefix changed its history")
            if prefix.emitted_token_ids == self.previous.emitted_token_ids:
                require(prefix.emitted_text == self.previous.emitted_text, "decoder changed an identical prefix")
                return []
        self.previous = prefix
        position = len(prefix.emitted_token_ids)
        self.closed = self.closed or "</think>" in prefix.emitted_text
        if self.closed:
            for action in self.actions:
                if action.action_id in self.opportunities and action.action_id not in self.decisions:
                    self.decisions[action.action_id] = {"status": "NO_FIRE_CLOSED", "position": position}
            return []
        response = self.recognizer(prefix)
        for key in ("controller_tokens", "controller_input_tokens", "controller_seconds"):
            require(key in response and isinstance(response[key], (int, float)) and
                    math.isfinite(response[key]) and response[key] >= 0, "missing online controller cost")
        require(type(response["controller_tokens"]) is int and type(response["controller_input_tokens"]) is int,
                "token accounting must be integer")
        require(response.get("prefix_sha256") == digest(asdict(prefix)), "recognizer answered a different prefix")
        require(type(response.get("boundary")) is bool and isinstance(response.get("substantive_acts"), list),
                "unqualified semantic recognition response")
        self.calls.append({"position": position, **{key: response[key] for key in
            ("controller_tokens", "controller_input_tokens", "controller_seconds", "prefix_sha256")}})
        fired = []
        for index, action in enumerate(self.actions):
            if action.action_id in self.decisions:
                continue
            if self.ordered and any(a.action_id not in self.decisions or self.decisions[a.action_id]["status"] != "FIRED"
                                    for a in self.actions[:index]):
                continue
            opportunity = (response["boundary"] and
                ((action.transition == "candidate_to_verify" and response.get("candidate_status") == "complete_unchecked") or
                 (action.transition == "approach_to_commit" and response.get("approach_identified") is True) or
                 (action.transition == "checked_candidate_to_close" and response.get("candidate_status") == "complete_checked")))
            if opportunity and action.action_id not in self.opportunities:
                self.opportunities[action.action_id] = {"position": position, "prefix_sha256": response["prefix_sha256"]}
            if action.action_id not in self.opportunities:
                continue
            start = self.opportunities[action.action_id]["position"]
            if action.transition in response["substantive_acts"]:
                self.decisions[action.action_id] = {"status": "NO_FIRE_DESIRED_ACT_OCCURRED", "position": position}
            elif position >= start + action.delay:
                # The current emitted boundary predicts the next token. For a
                # delayed policy, callers must observe the exact 64-token edge.
                require(position == start + action.delay, "executor skipped the frozen delayed boundary")
                decision = {"status": "FIRED", "position": position, "action": action.worker_action(),
                            "end": position + action.pulse_tokens}
                self.decisions[action.action_id] = decision
                fired.append(decision)
        return fired

    def routing_rows(self, positions, native_active):
        positions, native_active = np.asarray(positions), np.asarray(native_active, dtype=bool)
        require(positions.ndim == 1 and positions.dtype.kind in "iu" and native_active.shape == positions.shape,
                "routing rows require one isolated request")
        names = np.full(len(positions), "native", dtype=object)
        for action in self.actions:
            decision = self.decisions.get(action.action_id, {})
            if decision.get("status") != "FIRED":
                continue
            mask = native_active & (positions >= decision["position"]) & (positions < decision["end"])
            require(np.all(names[mask] == "native"), "composition pulses overlap; freeze a qualified composition rule")
            names[mask] = action.action_id
        return names

    def ledger(self):
        return {"opportunities": self.opportunities, "decisions": self.decisions, "controller_calls": self.calls,
            "controller_tokens": sum(c["controller_tokens"] for c in self.calls),
            "controller_input_tokens": sum(c["controller_input_tokens"] for c in self.calls),
            "controller_seconds": sum(c["controller_seconds"] for c in self.calls),
            "qualification_sha256": self.audit["sha256"], "top_k": 8, "shared_expert": "unchanged"}

    def finish(self, *, hit_cap=False, error=False):
        """Seal assignment-level non-fires without dropping their policy costs."""
        for action in self.actions:
            if action.action_id not in self.decisions:
                status = ("NO_FIRE_EXECUTION_ERROR" if error else "NO_FIRE_CAP" if hit_cap and
                    action.action_id in self.opportunities else "NO_ELIGIBLE_OPPORTUNITY")
                self.decisions[action.action_id] = {"status": status,
                    "position": len(self.previous.emitted_token_ids) if self.previous else 0}
        return self.ledger()


def matched_control(action, expert, strength, calibration):
    require(type(expert) is int and expert != action.expert and 0 <= expert < 256, "invalid random control")
    require(calibration.get("pool_rule") == "existing_native_exposure_matching" and
            calibration.get("population") == "discovery" and calibration.get("sha256"),
            "control lacks discovery exposure calibration")
    require(action.operator != "bias" or strength * action.strength > 0, "control bias sign differs")
    random = RoutingAction(action.action_id + "_random", action.transition, action.layer, expert,
                           action.operator, strength, action.delay)
    return {"action": asdict(random), "calibration_sha256": calibration["sha256"],
            "target_displacement": calibration.get("target_displacement"),
            "random_displacement": calibration.get("random_displacement"),
            "matched": calibration.get("matched") is True,
            "unmatched_retained": calibration.get("matched") is not True}


def freeze_independent_assignments(families, stage, policies=(), *, selection=None, random_policy=None, incumbent=None):
    require(families.get("status") == "FROZEN" and len(families["validation"]) == 32 and len(families["final"]) == 96,
            "independent family reservation is unavailable")
    validation = {r["family"] for r in families["validation"]}
    final = {r["family"] for r in families["final"]}
    require(len(validation) == 32 and len(final) == 96 and not validation & final,
            "independent family pools overlap")
    require(stage in ("native_validation", "validation", "final"), "unknown evaluation stage")
    if stage == "native_validation":
        arms, selected_rows = [("native", None)], families["validation"]
    elif stage == "validation":
        require(len(policies) <= 2 and len({p["policy_id"] for p in policies}) == len(policies),
                "validation permits at most two policies")
        require(all(p.get("qualification_status") == "PASS" and p.get("selection_population") == "discovery"
                    for p in policies), "unqualified validation policy or outcome leakage")
        arms = [("native", None)] + [(p["policy_id"], p["sha256"]) for p in policies]
        selected_rows = families["validation"]
    else:
        require(selection and selection.get("status") == "SELECTED_ON_VALIDATION" and
                selection.get("validation_families_sha256") == digest(families["validation"]) and selection.get("sha256"),
                "no frozen validation winner to advance")
        require(random_policy and incumbent and random_policy.get("operator_timing_matched_to") == selection["selected_policy"] and
                random_policy.get("calibration_population") == "discovery" and random_policy.get("sha256") and
                incumbent.get("frozen_before_validation") is True and incumbent.get("sha256"),
                "final comparators are not independently frozen and matched")
        arms = [("native", None), ("selected", selection["selected_policy_sha256"]),
                ("matched_random", random_policy["sha256"]), ("incumbent", incumbent["sha256"])]
        selected_rows = families["final"]
    rows = []
    for family in selected_rows:
        for seed in (0, 1):
            # Freeze execution order without outcome-dependent sorting.
            for arm, policy_sha in sorted(arms, key=lambda a: digest([family["family"], seed, a[0], "execution-order-v1"])):
                body = {"stage": stage, "family": family["family"], "question_id": family["question_id"],
                    "seed": seed, "arm": arm, "policy_sha256": policy_sha, "cap": 16384,
                    "top_k": 8, "shared_expert": "unchanged"}
                rows.append({**body, "uid": digest(body)})
    return {"schema": "routing-study-independent-assignments-v1", "stage": stage, "rows": rows,
            "family_freeze_sha256": families.get("sha256"), "source_revision": families["source_revision"],
            "maximum_assignments": len(rows), "test_outcomes_used_for_selection": False}


def price_workers(workloads, telemetry):
    """Price every role with its own measured load, prefill and decode rate.

    Controller generation is policy expenditure. Offline labels and grading
    are research expenditure. Every role includes shutdown, fixed overhead,
    and explicitly bounded extra loads/calls before authorization.
    """
    required = {"generation", "controller", "labels", "grading"}
    require(set(workloads) == required, "price must itemize generation, controller, labels and grading")
    rows, unavailable = [], []
    for role in sorted(required):
        work = workloads[role]
        require(all(type(work.get(k)) is int and work[k] >= 0 for k in ("input_tokens", "output_tokens", "loads", "recovery_loads")),
                "invalid role workload")
        if not work["input_tokens"] and not work["output_tokens"] and not work["loads"]:
            rows.append({"role": role, "seconds": 0, "GPU_hours": 0, "status": "NO_WORK_FROZEN"})
            continue
        measured = telemetry.get(role, {})
        keys = ("prefill_tps", "decode_tps", "load_seconds", "shutdown_seconds", "overhead_seconds", "call_recovery_factor", "gpus")
        if not measured.get("measurement_sha256") or not all(k in measured for k in keys):
            unavailable.append(role)
            continue
        require(all(isinstance(measured[k], (int, float)) and math.isfinite(measured[k]) and measured[k] >= 0 for k in keys),
                "nonfinite measured throughput")
        require(measured["prefill_tps"] > 0 and measured["decode_tps"] > 0 and measured["call_recovery_factor"] >= 1 and
                measured["gpus"] >= 1 and work["loads"] >= 1, "unqualified measured role price")
        prefill = work["input_tokens"] / measured["prefill_tps"]
        decode = work["output_tokens"] / measured["decode_tps"]
        seconds = (prefill + decode) * measured["call_recovery_factor"] + (
            work["loads"] + work["recovery_loads"]) * (measured["load_seconds"] + measured["shutdown_seconds"]) + measured["overhead_seconds"]
        rows.append({"role": role, "workload": work, "telemetry": measured, "prefill_seconds": prefill,
            "decode_seconds": decode, "seconds": seconds, "GPU_hours": seconds * measured["gpus"] / 3600,
            "cost_ledger": "policy" if role in ("generation", "controller") else "research", "status": "PRICED"})
    return {"schema": "routing-study-complete-worker-price-v1", "workers": rows,
            "unavailable_roles": unavailable, "GPU_hours": None if unavailable else sum(r["GPU_hours"] for r in rows),
            "status": "UNAVAILABLE_MEASURED_THROUGHPUT" if unavailable else "PRICED_REQUIRES_CONCRETE_AUTHORIZATION"}

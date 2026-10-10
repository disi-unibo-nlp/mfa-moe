"""Additive 16k execution, enrollment, qualification and resource contracts.

Historical schemas are unchanged. A design artifact is not a dispatch receipt:
all worker, recognizer, pricing and resource gates must independently pass.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math

from .study_v1 import REFERENCE_TARGETS, TRANSITIONS, digest, require

SCHEMA = "routing-study-local-context-v2"
CAP = 16_384
EARLY_WINDOW = 1_024
CHECKS = ("prefix_future_invariance", "prediction_row_alignment", "closure_row_included",
          "post_closure_native", "inactive_path_fidelity", "capture_accuracy_40_layers",
          "batch_isolation", "pause_resume", "sampler_continuity", "bounded_recovery")
RECOGNIZES = ("intermediate", "complete_unchecked", "complete_checked", "identified_approach",
              "substantive_verification", "substantive_commitment")


def seal(body):
    return {**body, "sha256": digest(body)}


def verify(value):
    require(isinstance(value, dict) and value.get("sha256") == digest(
        {k: v for k, v in value.items() if k != "sha256"}), "artifact seal differs")
    return value


@dataclass(frozen=True)
class Action:
    action_id: str
    transition: str
    layer: int
    experts: tuple[int, ...]
    operator: str
    strength: float
    delay: int = 0
    pulse_tokens: int = 256

    def __post_init__(self):
        require(self.action_id and self.transition in TRANSITIONS and type(self.layer) is int and
                0 <= self.layer < 40 and type(self.experts) is tuple and 1 <= len(self.experts) <= 2 and
                len(set(self.experts)) == len(self.experts) and
                all(type(e) is int and 0 <= e < 256 for e in self.experts), "invalid sparse action")
        require(self.operator in ("bias", "force", "reweight") and
                isinstance(self.strength, (int, float)) and not isinstance(self.strength, bool) and
                math.isfinite(self.strength) and
                (self.strength == 0 if self.operator == "force" else self.strength != 0 if
                 self.operator == "bias" else self.strength > 0), "invalid routing operator")
        require(type(self.delay) is int and self.delay in (0, 64) and self.pulse_tokens == 256,
                "unfrozen pulse or timing")

    def worker_action(self):
        return {"name": self.action_id, "transition": self.transition, "experts": [[self.layer, list(self.experts)]],
                "kind": self.operator, "sign": -1 if self.strength < 0 else 1, "magnitude": abs(self.strength)}


def protocol():
    return seal({"schema": SCHEMA, "model": "Qwen3.6-35B-A3B-FP8", "continuation_cap": CAP,
        "early_endpoint_window": EARLY_WINDOW, "seeds": [0, 1], "top_k": 8, "shared_expert": "unchanged",
        "reasoning_only": True, "max_new_singletons": 2, "max_prefixes_per_family_transition": 1,
        "max_families_per_transition": 24, "max_pulses_per_transition": 1, "max_pulses_per_request": 2,
        "pulse_tokens": 256, "timing_delays": [0, 64], "simultaneous_multilayer": False,
        "first_context_decline": "NO_FIRE_FOR_TRANSITION", "context_components": 8, "context_K": [1, 2, 4],
        "capture": {"layers": list(range(40)), "executed_ids": True, "selected_weights": True,
                    "row_semantics": "one prediction row per emitted token including closure-producing row"},
        "policy_cost": "generated_tokens + controller_generated_tokens",
        "confidence": "UNAVAILABLE_UNLESS_NATIVE_SEMANTICS_QUALIFY", "required_execution_checks": list(CHECKS),
        "exploratory_advancement": "verified dose; positive semantic point effects versus native and matched random; three positive family folds; simultaneous intervals reported",
        "selection": "maximum validation accuracy with non-increasing token cost; cost, intervened positions, policy ID ties",
        "accuracy_loss_margin": None, "historical_caps_reusable_as_16k": False,
        "dispatch": "STAGE_SPECIFIC_MEASURED_PRICE_AND_RESOURCE_AUTHORIZATION_REQUIRED"})


def qualify_recognizer(audits, *, family_boundary, supported_transitions, method="cheap", veto_price=None):
    """Prefix-only apparent precision gate with held-out uncertainty/coverage.

    Full future-visible reader audits cannot qualify this interface. Missing
    symbolic audits fail the gate, rather than dropping symbolic candidates.
    """
    require(method in ("cheap", "native_semantic_veto") and supported_transitions and
            set(supported_transitions) <= set(TRANSITIONS), "unsupported recognizer")
    from .families_v2 import require_discovery
    require(all(verify(a) and a.get("prefix_only") is True and a.get("population") == "discovery" for a in audits),
            "recognizer readers saw future text or independent outcomes")
    indexed = {a["decision"]: a for a in audits}
    require(len(indexed) == len(audits) and set(indexed) == set(RECOGNIZES), "incomplete semantic decision audits")
    reasons = []
    for decision, audit in indexed.items():
        families = audit.get("eligible_families", [])
        require_discovery(families + audit.get("held_out_families", []), family_boundary)
        n, yes = audit.get("apparent_positive_count"), audit.get("confirmed_positive_count")
        require(type(n) is int and type(yes) is int and 0 <= yes <= n and
                len(set(families)) == len(families), "invalid apparent precision accounting")
        require(audit.get("held_out_families") and not set(families) & set(audit["held_out_families"]) and
                isinstance(audit.get("held_out_precision_ci95"), list) and len(audit["held_out_precision_ci95"]) == 2 and
                all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in audit["held_out_precision_ci95"]) and
                isinstance(audit.get("held_out_coverage"), (int, float)) and 0 <= audit["held_out_coverage"] <= 1,
                "held-out uncertainty and coverage required")
        if len(families) < 10 or n == 0 or yes / n < .70:
            reasons.append(decision + ":PRECISION_OR_FAMILY_SUPPORT")
        if audit.get("symbolic_candidates_audited") is not True:
            reasons.append(decision + ":SYMBOLIC_AUDIT_MISSING")
    if method == "native_semantic_veto":
        require(veto_price and verify(veto_price) and veto_price.get("status") == "PRICED" and
                veto_price.get("bounded_controller_calls") and veto_price.get("full_controller_cost") is True,
                "native semantic veto must be bounded and fully priced")
    return seal({"schema": "routing-prefix-recognizer-qualification-v2", "status": "FAIL" if reasons else "PASS",
        "family_boundary_sha256": family_boundary["sha256"],
        "method": method, "prefix_only": True, "recognizes_substantive_acts": not reasons,
        "supported_transitions": list(supported_transitions), "decision_audit_sha256": [a["sha256"] for a in audits],
        "veto_price_sha256": veto_price["sha256"] if veto_price else None,
        "missing_or_ambiguous": "ABSTAIN", "failure_reasons": reasons})


def execution_contract(*, checkpoint_revision, engine_profile, sampler, resume, recognizer, worker,
                       family_boundary, context_model=None, opportunity_gate=None):
    from .families_v2 import validate_boundary
    validate_boundary(family_boundary)
    for artifact in (engine_profile, sampler, resume, recognizer, worker):
        verify(artifact)
    require(checkpoint_revision and sampler.get("max_tokens") == CAP and sampler.get("seeds") == [0, 1],
            "checkpoint or new scientific sampler cap differs")
    require(resume.get("status") == "PASS" and resume.get("engine_rng_state") is True and
            resume.get("seed_only_restart") is False, "unqualified sampler/resume continuity")
    require(recognizer.get("status") == "PASS" and recognizer.get("prefix_only") is True,
            "prefix recognizer is unavailable")
    require(recognizer.get("family_boundary_sha256") == family_boundary["sha256"], "recognizer family partition differs")
    require(worker.get("status") == "PASS" and all(worker.get("checks", {}).get(k) is True for k in CHECKS) and
            worker.get("code_files") and worker.get("cross_run_variability_audited") is True,
            "v2 execution-local worker qualification is unavailable")
    if context_model is not None:
        verify(context_model)
        require(context_model.get("population") == "discovery" and
                context_model.get("status") in ("MULTIPLE_CONTEXTS_SUPPORTED", "FALLBACK_K1") and
                context_model.get("family_boundary_sha256") == family_boundary["sha256"], "unqualified context model")
    if opportunity_gate is not None:
        verify(opportunity_gate)
        require(opportunity_gate.get("population") == "discovery" and opportunity_gate.get("status") == "PASS",
                "unqualified opportunity gate")
    return seal({"schema": "routing-study-execution-contract-v2", "protocol_sha256": protocol()["sha256"],
        "checkpoint_revision": checkpoint_revision, "engine_profile_sha256": engine_profile["sha256"],
        "family_boundary": family_boundary,
        "sampler_sha256": sampler["sha256"], "resume_sha256": resume["sha256"],
        "prefix_recognizer_sha256": recognizer["sha256"], "worker_qualification_sha256": worker["sha256"],
        "context_model_sha256": context_model["sha256"] if context_model else None,
        "opportunity_gate_sha256": opportunity_gate["sha256"] if opportunity_gate else None,
        "continuation_cap": CAP, "early_endpoint_window": EARLY_WINDOW,
        "capture_layers": list(range(40)), "capture_executed_ids": True, "capture_selected_weights": True,
        "top_k": 8, "shared_expert": "unchanged", "nonfires_retained": True, "code_files": worker["code_files"]})


def earliest_prefixes(prefixes, transition):
    grouped = defaultdict(list)
    for p in prefixes:
        if p["transition"] != transition or p.get("eligible") is not True:
            continue
        require(p.get("population") == "discovery" and p.get("prefix_recognition_qualified") is True and
                p.get("prefix_ids_sha256") and p.get("uid") and p.get("family") and
                type(p.get("position")) is int and p["position"] >= 0, "unqualified prefix identity")
        grouped[p["family"]].append(p)
    selected = [min(values, key=lambda p: (p["position"], p["uid"])) for values in grouped.values()]
    return sorted(selected, key=lambda p: digest([SCHEMA, p["family"]]))[:24]


def _execution(execution):
    verify(execution)
    from .families_v2 import validate_boundary
    validate_boundary(execution["family_boundary"])
    require(execution.get("schema") == "routing-study-execution-contract-v2" and
            execution.get("protocol_sha256") == protocol()["sha256"] and
            execution.get("continuation_cap") == CAP and execution.get("early_endpoint_window") == EARLY_WINDOW and
            execution.get("capture_layers") == list(range(40)) and execution.get("capture_selected_weights") is True,
            "wrong execution contract or historical horizon")
    require(execution.get("capture_executed_ids") is True and execution.get("top_k") == 8 and
            execution.get("shared_expert") == "unchanged" and execution.get("code_files"),
            "execution geometry or source binding differs")


def freeze_stage(stage, targets, prefixes, controls, execution, *, evidence=(), gate=None):
    _execution(execution)
    from .families_v2 import require_discovery
    require_discovery([p["family"] for p in prefixes], execution["family_boundary"])
    require(stage in ("target", "operator", "timing", "context") and 1 <= len(targets) <= 2 and
            len({(t["layer"], t["expert"]) for t in targets}) == len(targets), "invalid singleton stage")
    supports = {(r["layer"], r["expert"]): verify(r) for r in evidence}
    rows = []
    if stage == "context":
        verify(gate)
        require(gate.get("status") == "PASS" and gate.get("population") == "discovery" and
                gate["sha256"] == execution["opportunity_gate_sha256"] and execution["context_model_sha256"],
                "context gate not bound to execution")
    for target in targets:
        identity = target["layer"], target["expert"]
        require(target.get("prefix_recognition_qualified") is True and target["transition"] in TRANSITIONS,
                "target lacks qualified prefix support")
        if stage == "target":
            require(identity not in REFERENCE_TARGETS and min(target["positive_families"], target["comparison_families"]) >= 10 and
                    target["consistent_folds"] >= 3, "unsupported new singleton")
            options = [("positive", "bias", 1., 0), ("negative", "bias", -1., 0)]
        else:
            support = supports.get(identity, {})
            require(support.get("individually_supported") is True and support.get("population") == "discovery",
                    "individual exploratory support required")
            if stage == "operator":
                options = [("bias", "bias", 1., 0), ("force", "force", 0., 0), ("reweight", "reweight", 1., 0)]
            else:
                op, strength = support["selected_operator"], support["selected_strength"]
                require(op == "force" or strength > 0, "negative actions are diagnostic, not automatic policies")
                options = ([("first", op, strength, 0), ("delayed64", op, strength, 64)] if stage == "timing" else
                           [("state_only", op, strength, support["selected_delay"]),
                            ("routing_context", op, strength, support["selected_delay"]),
                            ("random_skip", op, strength, support["selected_delay"])])
        selected = earliest_prefixes(prefixes, target["transition"])
        require(selected, "no qualified prefix opportunities")
        for prefix in selected:
            for seed in (0, 1):
                base = {"stage": stage, "target": list(identity), "prefix_uid": prefix["uid"],
                    "prefix_ids_sha256": prefix["prefix_ids_sha256"], "family": prefix["family"],
                    "transition": target["transition"], "seed": seed, "continuation_cap": CAP,
                    "early_endpoint_window": EARLY_WINDOW, "execution_contract_sha256": execution["sha256"],
                    "nonfires_retained": True, "opportunity_gate_sha256": gate["sha256"] if gate else None}
                native = {**base, "arm": "native", "action": None, "control": None}
                rows.append({**native, "uid": digest([SCHEMA, native])})
                for arm, operator, strength, delay in options:
                    action = Action(f"L{identity[0]}E{identity[1]}_{arm}", target["transition"], identity[0],
                                    (identity[1],), operator, strength, delay)
                    cell = {**base, "arm": arm, "action": asdict(action), "control": None}
                    rows.append({**cell, "uid": digest([SCHEMA, cell])})
                    if stage == "context":
                        continue
                    control = controls.get((*identity, prefix["family"], seed, operator, -1 if strength < 0 else 1))
                    verify(control)
                    require(control.get("population") == "discovery" and control.get("pool_rule") == "existing_native_exposure_matching" and
                            type(control.get("expert")) is int and control["expert"] != identity[1], "unqualified random control")
                    random_strength = strength if stage == "target" else control["strength"]
                    require(operator != "bias" or random_strength * strength > 0, "random control sign differs")
                    random = Action(action.action_id + "_random", action.transition, action.layer,
                                    (control["expert"],), operator, random_strength, delay)
                    cell = {**base, "arm": "random_" + arm, "action": asdict(random), "control": control}
                    rows.append({**cell, "uid": digest([SCHEMA, cell])})
    require(len({r["uid"] for r in rows}) == len(rows), "duplicate assignment")
    return seal({"schema": "routing-study-experiment-manifest-v2", "stage": stage, "execution": execution,
        "targets": targets, "assignments": rows, "assigned": len(rows), "seeds": [0, 1],
        "continuation_cap": CAP, "early_endpoint_window": EARLY_WINDOW,
        "dispatch_status": "REQUIRES_MEASURED_PRICE_AND_STAGE_RESOURCE_AUTHORIZATION"})


def advance_action(result):
    """Exploratory effects may be uncertain; intervals must still be reported."""
    verify(result)
    require(result.get("population") == "discovery" and result.get("continuation_cap") == CAP and
            result.get("early_endpoint_window") == EARLY_WINDOW and
            result.get("simultaneous_ci95") is not None, "new-stage measurement or intervals missing")
    return (result.get("dose_verified") is True and result.get("random_dose_matched") is True and
            result.get("semantic_vs_native", 0) > 0 and result.get("semantic_vs_random", 0) > 0 and
            sum(x is not None and x > 0 for x in result.get("family_fold_effects", [])) >= 3)


def select_operator(results):
    eligible = [r for r in results if advance_action(r) and (r["operator"] == "force" or r.get("strength", 0) > 0)]
    for r in eligible:
        require(r["operator"] in ("bias", "force", "reweight") and r["token_cost"] >= 0 and
                r["intervened_positions"] >= 0, "invalid operator tie-break measurements")
    return min(eligible, key=lambda r: (-min(r["semantic_vs_native"], r["semantic_vs_random"]),
        r["token_cost"], r["intervened_positions"], ("bias", "force", "reweight").index(r["operator"]))) if eligible else None


def incumbent_actions():
    return (Action("incumbent_verify", "candidate_to_verify", 28, (9, 189), "force", 0),
            Action("incumbent_commit", "approach_to_commit", 24, (159,), "bias", 1))


def dispatch_gate(manifest, price, authorization, live):
    """Validate a concrete dispatch; does not submit or infer authorization."""
    for value in (manifest, price, authorization, live):
        verify(value)
    _execution(manifest["execution"])
    from .study_v1 import file_sha
    require(all(file_sha(path) == sha for path, sha in manifest["execution"]["code_files"].items()),
            "qualified worker source changed")
    require(price.get("status") == "PRICED" and price.get("manifest_sha256") == manifest["sha256"] and
            price.get("execution_contract_sha256") == manifest["execution"]["sha256"], "complete matching stage price missing")
    require(authorization.get("approved") is True and authorization.get("stage") == manifest["stage"] and
            authorization.get("manifest_sha256") == manifest["sha256"] and authorization.get("price_sha256") == price["sha256"] and
            authorization.get("maximum_GPU_hours", -1) >= price["maximum_GPU_hours"] and
            authorization.get("maximum_billing_core_hours", -1) >= price["maximum_billing_core_hours"] and
            authorization.get("maximum_storage_bytes", -1) >= price["capture_storage_bytes"], "stage resource authorization missing")
    require(live.get("user") == "lmolfett" and "leonardo" in live.get("hostname", "") and
            live.get("account") == "iscrc_miosr" and live.get("partition") and live.get("qos") and
            live.get("verified_utc") and live.get("eligible") is True, "live cluster entitlement missing")
    return seal({"schema": "routing-study-authorized-dispatch-v2", "manifest_sha256": manifest["sha256"],
                 "price_sha256": price["sha256"], "authorization_sha256": authorization["sha256"],
                 "live_preflight_sha256": live["sha256"], "status": "AUTHORIZED"})

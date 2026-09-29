"""Development-prefix sham matching and separate alternative-route likelihood checks."""
from __future__ import annotations

import math
from moe_exp.correlation_pipeline.dynamics.common import digest


def matched_sham(group, prevalence, *, layer, seed=42):
    """Match each target's marginal prevalence without reusing target identities."""
    if len(set(group)) != len(group) or any(e not in prevalence for e in group):
        raise ValueError("Invalid target group or missing development prevalence")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in prevalence.values()):
        raise ValueError("Prevalence must be per-token selection probability")
    available = set(prevalence) - set(group)
    if len(available) < len(group):
        raise ValueError("Insufficient same-layer sham experts")
    chosen = []
    for target in sorted(group, key=lambda e: (-prevalence[e], e)):
        best = min(available, key=lambda e: (abs(prevalence[e] - prevalence[target]), digest([seed, layer, e])))
        chosen.append(best)
        available.remove(best)
    return dict(layer=layer, experts=chosen, target_experts=list(group),
        target_prevalence=sum(prevalence[e] for e in group),
        sham_prevalence=sum(prevalence[e] for e in chosen),
        absolute_prevalence_error=abs(sum(prevalence[e] for e in group) - sum(prevalence[e] for e in chosen)),
        dose_matching="requires_independent_realized_change_calibration")


def likelihood_panel(prefix, verified_tokens, routes, evaluate, *, verification_binding):
    """Evaluate teacher-forced verified tokens at equal expert-execution counts.

    The callback must report per-token log probabilities and actual executions.
    No free-generation score or downstream accuracy claim is inferred.
    """
    if not verification_binding or not verified_tokens or "baseline" not in routes:
        raise ValueError("Verified continuation and baseline are required")
    outputs = {}
    for name, route in routes.items():
        value = evaluate(prefix, verified_tokens, route)
        if value.get("target_token_ids") != verified_tokens or len(value.get("log_probabilities", [])) != len(verified_tokens):
            raise ValueError("Alternative routes were not scored on identical continuation tokens")
        if any(not math.isfinite(v) or v > 1e-6 for v in value["log_probabilities"]):
            raise ValueError("Invalid token log probabilities")
        if type(value.get("expert_executions")) is not int or value["expert_executions"] <= 0:
            raise ValueError("Missing measured expert executions")
        outputs[name] = value
    baseline = outputs["baseline"]
    if any(v["expert_executions"] != baseline["expert_executions"] for v in outputs.values()):
        raise ValueError("Alternative-route panel is not equal compute")
    return dict(schema_version=1, claim="teacher_forced_likelihood_only", verification_binding=verification_binding,
        prefix_binding=digest(prefix), target_binding=digest(verified_tokens),
        routes={name: dict(**value, mean_log_probability_change=sum(
            a - b for a, b in zip(value["log_probabilities"], baseline["log_probabilities"])) / len(verified_tokens))
            for name, value in outputs.items()})

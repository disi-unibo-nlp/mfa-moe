"""Frozen selection/action/timing identities; unavailable conditions remain explicit."""
from __future__ import annotations
import math
from .common import CHECKPOINT, code_hashes, seal, verify, checkpoint_ok

ACTIONS = ("native", "identity", "steer_promote", "steer_suppress", "rice", "temperature", "margin")
GATES = ("always", "persistent", "unstable", "random_persistent", "random_unstable",
         "position_persistent", "position_unstable")


def freeze(action, strength=0., *, targets=(), parameters=None, calibration=None,
           gate="always", token_scope="generated_reasoning", unavailable=None, code=None):
    value = dict(schema=1, action=action, strength=float(strength),
                 selector={"native": "native", "identity": "accuracy_lift",
                           "steer_promote": "outcome_rate_difference",
                           "steer_suppress": "outcome_rate_difference",
                           "rice": "signed_marker_npmi", "temperature": "native",
                           "margin": "calibrated_boundary_gap"}[action],
                 targets=[list(t) for t in targets], parameters=parameters or {},
                 gate=gate, token_scope=token_scope, checkpoint=CHECKPOINT,
                 calibration=calibration, code=code if code is not None else code_hashes(),
                 status="unavailable" if unavailable else "frozen", unavailable_reason=unavailable)
    value = seal(value)
    validate(value, executable=False)
    return value


def validate(policy, *, executable=True, current_code=False):
    verify(policy)
    checkpoint_ok(policy["checkpoint"])
    if (policy["action"] not in ACTIONS or policy["gate"] not in GATES
            or policy["token_scope"] not in ("generated_reasoning", "historical_all_tokens")
            or not math.isfinite(policy["strength"])):
        raise ValueError("Unknown action/gate/scope or nonfinite strength")
    targets = [tuple(t) for t in policy["targets"]]
    if len(set(targets)) != len(targets) or any(
            len(t) != 2 or type(t[0]) is not int or type(t[1]) is not int
            or t[0] not in CHECKPOINT["layers"] or not 0 <= t[1] < 256 for t in targets):
        raise ValueError("Targets must be unique layer-qualified routed experts")
    if executable and policy["status"] != "frozen":
        raise ValueError("Unavailable condition: " + str(policy["unavailable_reason"]))
    if policy["status"] not in ("unavailable", "frozen"):
        raise ValueError("Invalid status")
    if policy["status"] == "frozen":
        action, strength, params = policy["action"], policy["strength"], policy["parameters"]
        if action.startswith("steer_") or action == "rice":
            if len(targets) != 2:
                raise ValueError("Exactly two global targets required")
        if action in ("rice", "temperature") and strength <= 0:
            raise ValueError("Weight multipliers/temperatures must be positive")
        if action == "identity":
            for layer, info in params.get("layers", {}).items():
                scores = info["scores"]
                if (int(layer) not in CHECKPOINT["layers"] or len(scores) != 256
                        or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores)
                        or sum(s > 0 for s in scores) > 8):
                    raise ValueError("Invalid per-layer identity scores")
            if strength and not any(any(v["scores"]) for v in params.get("layers", {}).values()):
                raise ValueError("No supported identity targets")
        if action == "margin":
            if not 0 <= strength <= 1 or not params.get("layers"):
                raise ValueError("Margin needs calibrated layer ranges")
            for layer, info in params["layers"].items():
                if int(layer) not in CHECKPOINT["layers"] or not 0 <= info["lower"] <= info["upper"] <= 1/8:
                    raise ValueError("Invalid margin interval")
        if action.startswith("steer_") and params.get("epsilon") != .01:
            raise ValueError("SteerMoE epsilon is frozen at .01")
        if action == "rice" and type(params.get("renormalize")) is not bool:
            raise ValueError("RICE must name normalization")
        if action not in ("native", "temperature") and not (action == "identity" and strength == 0):
            if not policy["calibration"]:
                raise ValueError("Calibration identity required")
        if policy["gate"] != "always" and not params.get("timing"):
            raise ValueError("Adaptive thresholds and control rates must be frozen")
    if current_code and policy["code"] != code_hashes():
        raise ValueError("Policy code changed; freeze a new study version")
    return policy


def static_grid(calibration, *, code=None):
    binding = {"sha256": calibration["sha256"], "split_sha256": calibration["split_sha256"],
               "attempts": calibration["attempts"], "checkpoint": CHECKPOINT}
    result = [freeze("native", code=code), freeze("identity", 0, code=code)]
    identity = calibration["identity"]
    for alpha in (-1, -.5, .5, 1):
        result.append(freeze("identity", alpha, parameters=identity.get("policy"),
                             calibration=binding, unavailable=identity.get("unavailable"), code=code))
    for action, key in (("steer_promote", "promote"), ("steer_suppress", "suppress")):
        value = calibration["steer"][key]
        result.append(freeze(action, 1, targets=value.get("targets", []), parameters={"epsilon": .01},
                             calibration=binding, unavailable=value.get("unavailable"), code=code))
    rice = calibration["rice"]
    for beta, normalize in ((.5, True), (2, True), (4, True), (2, False)):
        result.append(freeze("rice", beta, targets=rice.get("targets", []),
                             parameters={"renormalize": normalize}, calibration=binding,
                             unavailable=rice.get("unavailable"), code=code))
    for tau in (.5, 2):
        result.append(freeze("temperature", tau, code=code))
    for strength in (.5, 1):
        margin = calibration["margin"]
        result.append(freeze("margin", strength, parameters=margin.get("policy"),
                             calibration=binding, unavailable=margin.get("unavailable"), code=code))
    return result


def adaptive_variants(policy, timing):
    validate(policy)
    binding = policy["calibration"]
    if binding is None:
        binding = timing["calibration"]
    result = []
    for gate in GATES[1:]:
        result.append(freeze(policy["action"], policy["strength"], targets=policy["targets"],
            parameters={**policy["parameters"], "timing": timing}, calibration=binding,
            gate=gate, code=policy["code"]))
    return result

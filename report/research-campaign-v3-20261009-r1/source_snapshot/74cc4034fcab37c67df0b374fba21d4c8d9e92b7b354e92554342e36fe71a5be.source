"""Source-bound singleton actions and operator-matched random control."""
from __future__ import annotations

from .contracts import require, verify
from .nomination import random_expert


def action_for(policy, assignment):
    verify(policy, "verification-policy-v3")
    require(policy["status"] == "PASS" and policy["model_sha256"] == assignment["model_sha256"] and
            assignment["policy_sha256"] == policy["sha256"], "policy/assignment binding differs")
    expert = policy["target"]
    bias = policy["target_bias"]
    if assignment["arm"] == "matched_random":
        expert = random_expert([int(e) for e in policy["random_biases"]], family=assignment["family"],
                               model=assignment["model"], target=policy["target"])
        bias = policy["random_biases"][str(expert)]
    if assignment["arm"] == "native":
        bias = 0.
    require(assignment["arm"] in ("native", "target", "matched_random"), "unknown policy arm")
    return {"layer": policy["layer"], "expert": expert, "bias": bias}

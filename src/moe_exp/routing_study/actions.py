"""Tensor actions at the native selection/dispatch boundary (lazy torch import)."""
from __future__ import annotations


def logits_action(logits, policy, layer):
    import torch
    action, alpha, params = policy["action"], policy["strength"], policy["parameters"]
    if action == "identity":
        scores = params.get("layers", {}).get(str(layer), {}).get("scores")
        if not alpha or scores is None:
            return logits
        from moe_exp.moe_identity_guiding.routing import IdentityBias
        return IdentityBias(scores, alpha, 8, diagnostics="minimal")(None, (), logits)
    if action == "margin":
        target = params["layers"].get(str(layer))
        if target is None or not alpha:
            return logits
        from moe_exp.moe_margin_guiding.routing import MarginGuide
        return MarginGuide(target, alpha, 8, 256, diagnostics="minimal")(None, (), logits)
    if action in ("steer_promote", "steer_suppress"):
        targets = [e for l, e in policy["targets"] if l == layer]
        if not targets:
            return logits
        scores = logits.float().log_softmax(-1)
        high = action == "steer_promote"
        extreme = (scores.max(-1).values if high else scores.min(-1).values)
        scores[:, targets] = (extreme + (.01 if high else -.01))[:, None]
        candidate = scores.to(logits.dtype)
        # Preserve strict ordering when epsilon rounds away in BF16/FP16.
        native_extreme = extreme.to(logits.dtype)
        rounded = candidate[:, targets[0]]
        lost = rounded <= native_extreme if high else rounded >= native_extreme
        adjacent = torch.nextafter(native_extreme, torch.full_like(native_extreme,
                                   float("inf") if high else -float("inf")))
        value = torch.where(lost, adjacent, rounded)
        if not torch.isfinite(value).all():
            raise ValueError("Finite-precision SteerMoE adjustment overflow")
        candidate[:, targets] = value[:, None]
        return candidate
    return logits


def weights_action(weights, ids, policy, layer):
    """Returns only new weights. IDs are never reconstructed or selected here."""
    import torch
    action, strength = policy["action"], policy["strength"]
    if action == "temperature":
        # Stable exponentiation; true zeros remain zero.
        logw = torch.where(weights > 0, weights.float().log(),
                           torch.full_like(weights.float(), -float("inf")))
        changed = (logw / strength).softmax(-1).to(weights.dtype)
    elif action == "rice":
        targets = [e for l, e in policy["targets"] if l == layer]
        if not targets:
            return weights
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for expert in targets:
            mask |= ids == expert
        changed = weights.float() * torch.where(mask, strength, 1.)
        if policy["parameters"]["renormalize"]:
            changed /= changed.sum(-1, keepdim=True)
        changed = torch.where(mask.any(-1, keepdim=True), changed.to(weights.dtype), weights)
    else:
        return weights
    if not torch.isfinite(changed).all() or (changed < 0).any() or (changed.sum(-1) <= 0).any():
        raise ValueError("Invalid executed weights")
    return changed

"""Hard expert inclusion/exclusion and reference-style router-score override."""
from .common import require


def hard_select(logits, native_weights, native_ids, targets, direction, *, scoring='softmax',
                renormalize=True, correction_bias=None, scale=1., expert_scale=None, active=None):
    import torch
    require(direction in ('activate', 'deactivate'), 'invalid intervention direction')
    E, k = logits.shape[-1], native_ids.shape[-1]
    targets = sorted(set(targets))
    require(targets and len(targets) < k and min(targets) >= 0 and max(targets) < E,
            'invalid intervention expert count or identity')
    require(E-len(targets) >= k, 'deactivation cannot preserve native k')
    require(native_weights.shape == native_ids.shape and len(logits) == len(native_ids), 'routing shape differs')
    q = logits.float().softmax(-1) if scoring == 'softmax' else logits.float().sigmoid() if scoring == 'sigmoid' else None
    require(q is not None, 'unsupported native affinity')
    priority = q if correction_bias is None else q + correction_bias.float()
    priority = priority.clone()
    priority[:, targets] = -float('inf')
    if direction == 'activate':
        rest = priority.topk(k-len(targets), dim=-1, sorted=True).indices
        fixed = torch.tensor(targets, device=logits.device).expand(len(logits), -1)
        ids = torch.cat((fixed, rest), dim=-1)
    else:
        ids = priority.topk(k, dim=-1, sorted=True).indices
    changed = (ids.sort(-1).values != native_ids.long().sort(-1).values).any(-1)
    if active is not None:
        changed &= torch.as_tensor(active, dtype=torch.bool, device=logits.device)
    if not bool(changed.any()):
        return native_weights, native_ids, changed
    weights = q.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    weights = weights * scale
    if expert_scale is not None:
        weights *= expert_scale[ids].to(weights.dtype)
    return (torch.where(changed[:, None], weights.to(native_weights.dtype), native_weights),
            torch.where(changed[:, None], ids.to(native_ids.dtype), native_ids), changed)


def score_override(logits, targets, direction, *, epsilon=0.01, active=None, scoring='softmax',
                   correction_bias=None):
    """Return logits with the reference-style score override for active rows.

    This follows the operator used by the public SteerMoE implementation: scores
    are evaluated in log-softmax space, positive target experts are assigned the
    row maximum plus ``epsilon`` and negative target experts the row minimum
    minus ``epsilon``.  The native selector must still be called on the returned
    logits so its top-k, correction bias, normalization and scaling semantics
    are preserved.
    """
    import torch
    require(direction in ('activate', 'deactivate'), 'invalid intervention direction')
    require(scoring == 'softmax', 'score override requires a softmax router')
    require(logits.ndim == 2 and targets, 'invalid score-override shape or target set')
    targets = sorted(set(targets))
    require(all(type(e) is int and 0 <= e < logits.shape[-1] for e in targets),
            'invalid score-override expert identity')
    if active is None:
        active = torch.ones(len(logits), dtype=torch.bool, device=logits.device)
    else:
        active = torch.as_tensor(active, dtype=torch.bool, device=logits.device)
    require(active.shape == (len(logits),), 'score-override active mask differs')
    changed = torch.zeros(len(logits), dtype=torch.bool, device=logits.device)
    if not bool(active.any()):
        return logits, changed
    out = logits.clone()
    rows = torch.where(active)[0]
    target_index = torch.tensor(targets, dtype=torch.long, device=logits.device)
    selected = logits[rows].float().log_softmax(-1)
    if direction == 'activate':
        value = selected.max(dim=-1, keepdim=True).values + epsilon
    else:
        value = selected.min(dim=-1, keepdim=True).values - epsilon
    # Correction bias belongs to the native selector; the override is defined
    # on log-softmax scores before that selector applies its own policy.
    selected[:, target_index] = value
    out[rows] = selected.to(logits.dtype)
    changed[rows] = True
    return out, changed


def score_override_select(logits, native_weights, native_ids, targets, direction, *, epsilon=0.01,
                          active=None, scoring='softmax', renormalize=True, correction_bias=None,
                          scale=1., expert_scale=None):
    """Reference-style score override followed by the native top-k rule.

    The native selector is not re-entered because vLLM's fused selector may
    consume a cached routing shape.  Selection is performed from the overridden
    scores with the native k, correction bias, weighting, normalization and
    scaling semantics explicitly reproduced here.
    """
    import torch
    require(direction in ('activate', 'deactivate'), 'invalid intervention direction')
    require(scoring == 'softmax', 'score override requires a softmax router')
    require(logits.ndim == 2 and targets, 'invalid score-override shape or target set')
    targets = sorted(set(targets))
    require(all(type(e) is int and 0 <= e < logits.shape[-1] for e in targets),
            'invalid score-override expert identity')
    E, k = logits.shape[-1], native_ids.shape[-1]
    require(len(targets) < k and E-len(targets) >= k, 'score override cannot preserve native k')
    if active is None:
        active = torch.ones(len(logits), dtype=torch.bool, device=logits.device)
    active = torch.as_tensor(active, dtype=torch.bool, device=logits.device)
    require(active.shape == (len(logits),), 'score-override active mask differs')
    modified = logits.float().log_softmax(-1)
    target_index = torch.tensor(targets, dtype=torch.long, device=logits.device)
    q_native = modified.softmax(-1)
    priority = q_native if correction_bias is None else q_native + correction_bias.float()
    priority = priority.clone()
    target_mask = torch.zeros(E, dtype=torch.bool, device=logits.device)
    target_mask[target_index] = True
    if direction == 'activate':
        selected = priority.masked_fill(target_mask[None, :], -float('inf')).topk(k-len(targets), dim=-1).indices
        ids = torch.cat((target_index.expand(len(logits), -1), selected), dim=-1)
    else:
        ids = priority.masked_fill(target_mask[None, :], -float('inf')).topk(k, dim=-1).indices
    # Target scores are the extrema in the same score space used by the
    # reference operator; selected non-target scores remain native.
    scores = modified.clone()
    for row in torch.where(active)[0].tolist():
        scores[row, target_index] = (scores[row].max()+epsilon if direction == 'activate'
                                     else scores[row].min()-epsilon)
    q = scores.softmax(-1)
    weights = q.gather(1, ids)
    if renormalize:
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    weights = weights*scale
    if expert_scale is not None:
        weights *= expert_scale[ids].to(weights.dtype)
    changed = (ids.sort(-1).values != native_ids.long().sort(-1).values).any(-1) | \
              (weights.to(native_weights.dtype) != native_weights).any(-1)
    changed &= active
    out_ids = torch.where(active[:, None], ids.to(native_ids.dtype), native_ids)
    out_weights = torch.where(active[:, None], weights.to(native_weights.dtype), native_weights)
    if not bool(changed.any()):
        return native_weights, native_ids, changed
    return out_weights, out_ids, changed


def policy(router, parent):
    name = type(router).__name__
    require(getattr(router, 'eplb_state', None) is None, 'EPLB is outside the qualified profile')
    require(getattr(router, 'num_expert_group', 1) in (None, 1) and
            getattr(router, 'topk_group', 1) in (None, 1) and
            getattr(router, 'num_fused_shared_experts', 0) == 0 and
            getattr(router, '_hash_indices_table', None) is None, 'unsupported grouped/shared/hash routing')
    if name == 'CustomRoutingRouter':
        require(hasattr(parent, 'per_expert_scale'), 'unknown custom routing semantics')
        return {'scoring': 'softmax', 'renormalize': True, 'expert_scale': parent.per_expert_scale}
    require(name in ('FusedTopKRouter', 'FusedTopKBiasRouter', 'GroupedTopKRouter'),
            'unsupported router: '+name)
    return {'scoring': router.scoring_func, 'renormalize': router.renormalize,
            'correction_bias': getattr(router, 'e_score_correction_bias', None),
            'scale': getattr(router, 'routed_scaling_factor', 1.)}

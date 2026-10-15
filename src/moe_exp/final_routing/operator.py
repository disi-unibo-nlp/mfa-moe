"""Hard expert inclusion/exclusion, preserving native affinities and no-ops."""
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

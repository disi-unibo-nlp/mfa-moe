"""Selection scores of the routers and the margin between the last selected and the best unselected expert.

Only verified selector arithmetic is reconstructed; the experts the model really dispatched stay the reference.
"""
import torch


def selection_scores(logits, bias=None, *, scoring='softmax', fused_grouped=False):
    if scoring == 'sigmoid':
        # CUDA grouped_topk applies its tanh sigmoid in float, casts to input dtype,
        # then adds correction bias. Compute on the selector's device, not CPU.
        scores = ((torch.tanh(logits.float() * .5) * .5) + .5).to(logits.dtype) if fused_grouped else logits.sigmoid()
    elif scoring == 'softmax':
        scores = logits.softmax(-1) if bias is not None else logits
    else:
        raise ValueError('Unverified scoring function')
    return scores if bias is None else scores + bias


def boundary_gaps(scores, ids):
    x = scores.float()
    other = x.clone()
    other.scatter_(-1, ids.long(), float('-inf'))
    return x.gather(-1, ids.long()).min(-1).values - other.max(-1).values


def require_native_boundary(scores, ids):
    """The experts the model dispatched must be the top-k of the scores (up to 1e-6). Returns the gaps, at least 0."""
    gaps = boundary_gaps(scores, ids)
    if not torch.isfinite(gaps).all() or (gaps < -1e-6).any():
        raise ValueError('Native selected sets disagree with scores')
    return gaps.clamp_min(0)

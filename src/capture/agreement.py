"""Agreement checks between two captures of the same trace.

glm: the vLLM capture must agree with a Hugging Face reference (compare_backend_captures). gpt: the KV-chunked
replay must agree with the single-pass replay (compare_native_captures). A failed HF/vLLM check raises
DifferentialFailed with the full report, so a failure can be diagnosed from the saved numbers.
"""
import math

import numpy as np
import torch

from capture.native_selection import boundary_gaps, require_native_boundary


def _ranks(values):
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def _spearman(x, y):
    rx, ry = _ranks(np.asarray(x, dtype=float)), _ranks(np.asarray(y, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def window_dynamics(ids, experts, window=32):
    """Per-window routing dynamics from selected expert IDs [layers, tokens, k].

    Averaged over layers: entropy of the window's expert usage, Jensen-Shannon divergence
    from the previous window's usage, and mean Jaccard overlap of consecutive tokens' sets.
    Windows of ~sentence length stand in for sentences (no text alignment needed)."""
    ids = np.asarray(ids)
    layers, tokens, _ = ids.shape
    count = tokens // window
    usage = np.zeros((layers, count, experts))
    overlap = np.zeros((layers, count))
    for w in range(count):
        block = ids[:, w * window:(w + 1) * window]
        for layer in range(layers):
            usage[layer, w] = np.bincount(block[layer].ravel(), minlength=experts)[:experts]
            sets = [set(row) for row in block[layer].tolist()]
            overlap[layer, w] = np.mean([len(a & b) / len(a | b) for a, b in zip(sets, sets[1:])])
    p = usage / usage.sum(-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.where(p > 0, np.log(p), 0.0)
        entropy = -(p * logp).sum(-1)
        m = (p[:, 1:] + p[:, :-1]) / 2
        logm = np.where(m > 0, np.log(m), 0.0)
        kl_a = (p[:, 1:] * (np.where(p[:, 1:] > 0, np.log(p[:, 1:]), 0.0) - logm)).sum(-1)
        kl_b = (p[:, :-1] * (np.where(p[:, :-1] > 0, np.log(p[:, :-1]), 0.0) - logm)).sum(-1)
    return dict(entropy=entropy.mean(0), jsd=((kl_a + kl_b) / 2).mean(0), overlap=overlap.mean(0))


def dynamics_agreement(reference_ids, native_ids, experts, window=32):
    """Spearman agreement of window dynamics computed from the two captures."""
    a, b = window_dynamics(reference_ids, experts, window), window_dynamics(native_ids, experts, window)
    return dict(window=window, windows=int(len(a['entropy'])),
                spearman={name: _spearman(a[name], b[name]) for name in a})


def discontinuities(errors, factor=3.0, floor=5e-3):
    """Layers whose error jumps above `factor` x the median of their +-2 neighbours."""
    out = []
    for i, value in enumerate(errors):
        neighbours = [errors[j] for j in range(max(0, i - 2), min(len(errors), i + 3)) if j != i]
        if neighbours and value > floor and value > factor * float(np.median(neighbours)):
            out.append(i)
    return out


class DifferentialFailed(ValueError):
    """A failed HF/vLLM differential; `report` holds every computed statistic."""

    def __init__(self, report):
        super().__init__('; '.join(report['failures']))
        self.report = report


def compare_hf_vllm_capture(hf, native, *, total_input_tokens, layers,
                            hf_weight_scale=1.0, vllm_weight_scale=1.0,
                            near_tie_eps=1e-3,
                            selection_relative_l2_bound=5e-3, weight_relative_l2_bound=5e-3,
                            first_layer_relative_l2_bound=5e-3, global_relative_l2_bound=0.25,
                            min_dynamics_spearman=0.9, dynamics_min_tokens=2048,
                            first_layer_weight_p99_bound=0.05):
    """Short-trace HF/vLLM differential gated on the selection-score noise itself.

    bfloat16 HF and the Triton/Marlin MoE kernels round the selector differently, so
    near-tie experts can flip. If every selection score moves by at most delta at a
    (layer, token), two experts can only swap order when their reference gap is below
    2*delta; so the gate is: selection scores agree to selection_relative_l2_bound
    (the same 0.5% as router logits), every flip is explained by the observed
    per-position score shift, and per-layer effective weights agree on the tokens
    whose sets agree. Flip counts, near-tie (gap < near_tie_eps) statistics and the
    score-shift distribution are reported, not gated, so a failure is diagnosable.
    Router logits, hidden states and effective weights are compared per layer: bf16 drift
    between two kernel implementations accumulates with depth (GLM-4.7-Flash, 32k tokens:
    hidden 0.4% at the first routed layer rising to ~9%), whereas a semantic error (layer
    mapping, missing bias, wrong scaling or activation) already shows at the first routed
    layer. So the first routed layer must agree within first_layer_relative_l2_bound for
    router, hidden and weights; deeper layers are reported, with only a catastrophic cap of
    global_relative_l2_bound on router and hidden. The HF/vLLM expert-set disagreement
    (flip_fraction) is an implementation-sensitivity figure, reported not gated; a layer whose
    error jumps above 3x its neighbours' median fails (a mapping bug spikes, drift does not).
    On traces of >= dynamics_min_tokens the routing dynamics used downstream (window usage
    entropy, adjacent-window JSD, consecutive-token overlap) must agree across the two
    captures with Spearman >= min_dynamics_spearman. Every check is evaluated and reported before any failure is
    raised (DifferentialFailed carries the full report). Full-trace native dispatch and the
    -1e-6 selection-boundary checks remain independent gates.
    """
    if not 0 < total_input_tokens <= 32768:
        raise ValueError('HF/vLLM differential requires at most 32768 input tokens')
    if any(not math.isfinite(x) or x <= 0 for x in (hf_weight_scale, vllm_weight_scale)):
        raise ValueError('Invalid audited routed weight scale')
    failures = []
    for data in (hf, native):
        if data.get('audit', {}).get('native_dispatch') is not True:
            raise ValueError('HF/vLLM differential requires verified native dispatch')
        require_native_boundary(data['selection'], data['ids'])
    ids = hf['ids']
    if ids.ndim != 3 or ids.shape[0] != len(layers) or ids.shape != native['ids'].shape:
        raise ValueError('HF/vLLM layer/token/expert shape mismatch')
    for name, data in (('hf', hf), ('vllm', native)):
        if data['ids'].dtype not in (torch.int32, torch.int64):
            raise ValueError('HF/vLLM native expert IDs must be integers')
        sorted_ids = data['ids'].sort(-1).values
        if bool((torch.diff(sorted_ids, dim=-1) == 0).any()):
            raise ValueError('Duplicate expert in ' + name + ' native dispatch')
    gaps = boundary_gaps(hf['selection'], ids)
    if not bool(torch.isfinite(gaps).all()):
        raise ValueError('HF/vLLM reference selection gaps are not finite')
    reference_scores, native_scores = hf['selection'].float(), native['selection'].float()
    if reference_scores.shape != native_scores.shape or reference_scores.shape[:2] != ids.shape[:2] \
            or not bool(torch.isfinite(native_scores).all()):
        raise ValueError('Invalid HF/vLLM selection score tensors')
    selection_error = float(torch.linalg.vector_norm(reference_scores - native_scores)
                            / torch.linalg.vector_norm(reference_scores).clamp_min(1e-12))
    if not math.isfinite(selection_error) or selection_error > selection_relative_l2_bound:
        failures.append(f'HF/vLLM selection scores relative-L2 error: {selection_error}')
    score_shift = (reference_scores - native_scores).abs().amax(-1)
    reference_sets = ids.sort(-1).values
    native_sets = native['ids'].sort(-1).values
    agree = (reference_sets == native_sets).all(-1)
    near = gaps < near_tie_eps
    flips = ~agree
    explained = gaps <= 2 * score_shift + 1e-6
    unexplained_total = int((flips & ~explained).sum())
    if unexplained_total:
        failures.append('HF/vLLM expert sets flipped beyond the observed score noise: ' + str(unexplained_total))
    near_tie_flips = flips & near
    flips_total = int(flips.sum())
    near_total = int(near.sum())
    near_flips_total = int(near_tie_flips.sum())
    non_near_total = int((flips & ~near).sum())
    near_tie_flip_rate = (near_flips_total / near_total) if near_total else 0.0
    quantiles = torch.quantile(score_shift.flatten()[:1 << 24].double(),
                               torch.tensor([.5, .99, .999], dtype=torch.float64)).tolist()
    a = hf['weights'].float().gather(-1, hf['ids'].argsort(-1)) * hf_weight_scale
    b = native['weights'].float().gather(-1, native['ids'].argsort(-1)) * vllm_weight_scale
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError('HF/vLLM native effective weights are not finite')
    tensor_errors = {}
    for name in ('router', 'hidden'):
        x, y = hf[name].float(), native[name].float()
        if x.shape != y.shape or x.shape[:2] != ids.shape[:2] or not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
            raise ValueError(f'Invalid HF/vLLM {name} tensor')
        per = [float(torch.linalg.vector_norm(x[i] - y[i]) / torch.linalg.vector_norm(x[i]).clamp_min(1e-12))
               for i in range(len(layers))]
        overall = float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(x).clamp_min(1e-12))
        tensor_errors[name] = dict(global_relative_l2=overall, per_layer_relative_l2=per)
        jumps = discontinuities(per)
        if jumps:
            failures.append(f'HF/vLLM {name} error discontinuity at layers {[layers[i] for i in jumps]}')
        if name == 'hidden':
            tensor_errors[name]['norm_ratio_per_layer'] = [
                float(torch.linalg.vector_norm(y[i]) / torch.linalg.vector_norm(x[i]).clamp_min(1e-12))
                for i in range(len(layers))]
        if not math.isfinite(per[0]) or per[0] > first_layer_relative_l2_bound:
            failures.append(f'HF/vLLM differential error at first routed layer {layers[0]}: {name}={per[0]}')
        if not math.isfinite(overall) or overall > global_relative_l2_bound:
            failures.append(f'HF/vLLM differential error: {name}={overall}')
    per_layer = []
    for index, layer in enumerate(layers):
        token_mask = agree[index]
        if bool(token_mask.any()):
            left, right = a[index][token_mask], b[index][token_mask]
            weight_error = float(torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(left).clamp_min(1e-12))
            if not math.isfinite(weight_error) or (index == 0 and weight_error > weight_relative_l2_bound):
                failures.append(f'HF/vLLM layer {layer} effective weights relative-L2 error: {weight_error}')
            max_abs_error = float((left - right).abs().max())
        else:
            weight_error, max_abs_error = None, None
        layer_flips = int(flips[index].sum())
        layer_near = int(near[index].sum())
        layer_near_flips = int(near_tie_flips[index].sum())
        per_layer.append(dict(layer=layer, tokens=int(ids.shape[1]), agreeing_tokens=int(token_mask.sum()),
                              flips=layer_flips, near_ties=layer_near, near_tie_flips=layer_near_flips,
                              near_tie_flip_rate=(layer_near_flips / layer_near) if layer_near else 0.0,
                              weight_relative_l2=weight_error, weight_max_abs_error=max_abs_error,
                              score_shift_max=float(score_shift[index].max()),
                              router_relative_l2=tensor_errors['router']['per_layer_relative_l2'][index],
                              hidden_relative_l2=tensor_errors['hidden']['per_layer_relative_l2'][index]))
    result = {'status': 'passed',
              'native_selected_sets_identical': bool(agree.all()),
              'total_input_tokens': total_input_tokens, 'layers': list(layers),
              'completion_tokens': ids.shape[1], 'hf_weight_scale': hf_weight_scale,
              'vllm_weight_scale': vllm_weight_scale,
              'near_tie_eps': near_tie_eps, 'selection_relative_l2_bound': selection_relative_l2_bound,
              'weight_relative_l2_bound': weight_relative_l2_bound, 'selection_relative_l2': selection_error,
              'score_shift_quantiles': dict(zip(('p50', 'p99', 'p999'), quantiles)),
              'score_shift_max': float(score_shift.max()), 'flip_fraction': flips_total / flips.numel(),
              'flips': flips_total, 'unexplained_flips': unexplained_total,
              'near_ties': near_total, 'near_tie_flips': near_flips_total,
              'near_tie_flip_rate': near_tie_flip_rate, 'non_near_tie_flips': non_near_total,
              'first_layer_relative_l2_bound': first_layer_relative_l2_bound,
              'global_relative_l2_bound': global_relative_l2_bound,
              'per_layer': per_layer}
    for name in ('router', 'hidden'):
        x, y = hf[name].float(), native[name].float()
        result[name + '_relative_l2'] = float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(x).clamp_min(1e-12))
    if bool(agree.any()):
        if bool(agree[0].any()):
            # elementwise tail at the first routed layer: bf16 noise gives a p99 of ~2%, a bug
            # affecting a subset of experts (indexing, scaling) pushes it far higher
            x, y = a[0][agree[0]], b[0][agree[0]]
            relative = ((x - y).abs() / x.abs().clamp_min(1e-6)).flatten()
            tail = dict(p99=float(relative.quantile(.99)), max=float(relative.max()),
                        fraction_over_1pct=float((~torch.isclose(x, y, rtol=.01, atol=.001)).float().mean()))
            result['first_layer_weight_elementwise'] = tail
            if not tail['p99'] <= first_layer_weight_p99_bound:
                failures.append(f"HF/vLLM first-layer effective weights elementwise p99 relative error {tail['p99']}")
        result['weights_max_abs_error'] = float((a[agree] - b[agree]).abs().max())
    else:
        result['weights_max_abs_error'] = None
    result['tensor_errors'] = tensor_errors
    if ids.shape[1] >= dynamics_min_tokens:
        agreement = dynamics_agreement(ids.cpu(), native['ids'].cpu(), experts=int(hf['selection'].shape[-1]))
        result['dynamics_agreement'] = agreement
        for metric, rho in agreement['spearman'].items():
            if not rho >= min_dynamics_spearman:
                failures.append(f'HF/vLLM routing dynamics disagree: {metric} Spearman={rho}')
    else:
        result['dynamics_agreement'] = dict(status='not_evaluated', tokens=int(ids.shape[1]),
                                            min_tokens=dynamics_min_tokens)
    if failures:
        result.update(status='failed', failures=failures)
        raise DifferentialFailed(result)
    return result


def compare_backend_captures(reference, candidate, *, total_input_tokens, layers):
    """HF reference against the vLLM capture of the same trace; HF weights already include their scale.

    The vLLM audit must carry the observed per-layer runner output scale.
    A homogeneous GLM scale is expected, and unequal layer scales are rejected.
    """
    scales = candidate.get('audit', {}).get('routed_output_scaling')
    if not isinstance(scales, dict) or set(scales) != {str(layer) for layer in layers}:
        raise ValueError('Missing audited vLLM routed output scales')
    if len(set(scales.values())) != 1:
        raise ValueError('Differential requires homogeneous audited routed output scales')
    for data in (reference, candidate):
        if data.get('audit', {}).get('layers') != list(layers):
            raise ValueError('Differential absolute layer topology differs')
    return compare_hf_vllm_capture(reference, candidate, total_input_tokens=total_input_tokens,
                                  layers=layers, hf_weight_scale=1.0,
                                  vllm_weight_scale=float(next(iter(scales.values()))))


def compare_native_captures(reference, candidate, *, relative_l2_limit=.005):
    """Fail closed on expert-set disagreements; numerical tolerance is explicit."""
    if reference['ids'].shape != candidate['ids'].shape or not torch.equal(
            reference['ids'].sort(-1).values, candidate['ids'].sort(-1).values):
        raise ValueError('Chunked replay changed native selected expert sets')
    results = {}
    for field in ('router', 'hidden'):
        a, b = reference[field].float(), candidate[field].float()
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Invalid differential shape/value: ' + field)
        error = float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a).clamp_min(1e-12))
        if not math.isfinite(error) or error > relative_l2_limit:
            raise ValueError('Chunked differential exceeded tolerance: ' + field)
        results[field + '_relative_l2'] = error
    a = reference['weights'].gather(-1, reference['ids'].argsort(-1)).float()
    b = candidate['weights'].gather(-1, candidate['ids'].argsort(-1)).float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.allclose(a, b, rtol=.01, atol=.001):
        raise ValueError('Chunked replay changed native executed weights')
    return dict(status='passed', selected_sets_identical=True, **results)

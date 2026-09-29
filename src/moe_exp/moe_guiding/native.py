"""Native top-k policies, explicit token timing, and realized-dose telemetry.

The selector is the installed backend's own callable, including tie breaking,
selection biases and normalization. No-op returns its tensors unchanged. This
module's CPU tests do not qualify a GPU backend; qualification is a separate gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Callable
import time

import torch


@dataclass(frozen=True)
class NativePolicy:
    layer: int
    experts: tuple[int, ...]
    action: str = "noop"
    dose: float = 0.
    start_token: int = 2048
    pulse_tokens: int | None = 256

    def __post_init__(self):
        if self.action not in ("noop", "selection_bias", "selected_reweight"):
            raise ValueError("Unknown native routing action")
        if type(self.layer) is not int or self.layer < 0 or not self.experts:
            raise ValueError("One native layer and nonempty expert group are required")
        if any(type(e) is not int or e < 0 for e in self.experts) or len(set(self.experts)) != len(self.experts):
            raise ValueError("Expert IDs must be distinct nonnegative integers")
        if not math.isfinite(self.dose) or self.start_token < 0:
            raise ValueError("Invalid dose or start")
        if self.pulse_tokens is not None and self.pulse_tokens < 1:
            raise ValueError("Pulse duration must be positive")
        if self.action == "noop" and self.dose != 0:
            raise ValueError("No-op dose must be zero")
        object.__setattr__(self, "experts", tuple(self.experts))

    def to_dict(self):
        result = asdict(self)
        result["experts"] = list(self.experts)
        return result


@dataclass(frozen=True)
class TokenContext:
    request_id: str
    # Zero-based next output token index. A prefix of length P branches at P.
    output_token: int
    reasoning: bool
    valid: bool = True


def _validate_route(weights, ids, tokens, top_k, experts):
    if weights.shape != (tokens, top_k) or ids.shape != weights.shape:
        raise ValueError("Native selector changed token count or native k")
    if ids.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("Native expert IDs must be integers")
    if (ids < 0).any() or (ids >= experts).any() or (ids.sort(-1).values.diff(dim=-1) == 0).any():
        raise ValueError("Invalid/duplicate native expert IDs")
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights.sum(-1) <= 0).any():
        raise ValueError("Invalid native dispatched weights")


def manipulate(logits, select: Callable, policy: NativePolicy, *, top_k, active=None):
    """Select returns (native weights, native IDs); membership is compared as a set."""
    if logits.ndim != 2 or not 1 <= top_k <= logits.shape[-1] or max(policy.experts) >= logits.shape[-1]:
        raise ValueError("Policy does not match the native router shape")
    n, e = logits.shape
    baseline_w, baseline_ids = select(logits)
    _validate_route(baseline_w, baseline_ids, n, top_k, e)
    mask = torch.ones(n, dtype=torch.bool, device=logits.device) if active is None else active
    if mask.shape != (n,) or mask.dtype != torch.bool:
        raise ValueError("Token attribution mask mismatch")
    w, ids = baseline_w, baseline_ids
    if policy.action != "noop" and policy.dose != 0 and mask.any():
        if policy.action == "selection_bias":
            changed = logits.clone()
            group = torch.zeros(e, device=logits.device, dtype=logits.dtype)
            group[list(policy.experts)] = policy.dose
            changed = changed + mask[:, None] * group[None, :]
            proposed_w, proposed_ids = select(changed)
            _validate_route(proposed_w, proposed_ids, n, top_k, e)
            # Do not let another selector evaluation change inactive/tied rows.
            w = torch.where(mask[:, None], proposed_w, baseline_w)
            ids = torch.where(mask[:, None], proposed_ids, baseline_ids)
        else:
            in_group = torch.zeros(e, dtype=torch.bool, device=logits.device)
            in_group[list(policy.experts)] = True
            selected_group = in_group[ids.long()]
            changed_rows = mask & selected_group.any(-1) & ~selected_group.all(-1)
            shift = selected_group.double() * policy.dose
            shift -= shift.max(-1, keepdim=True).values
            proposed = baseline_w.double() * shift.exp()
            proposed = proposed / proposed.sum(-1, keepdim=True) * baseline_w.double().sum(-1, keepdim=True)
            w = torch.where(changed_rows[:, None], proposed.to(baseline_w.dtype), baseline_w)
    _validate_route(w, ids, n, top_k, e)
    baseline_dense = torch.zeros((n, e), dtype=torch.float64, device=w.device)
    changed_dense = torch.zeros_like(baseline_dense)
    baseline_dense.scatter_add_(1, baseline_ids.long(), baseline_w.double())
    changed_dense.scatter_add_(1, ids.long(), w.double())
    telemetry = dict(membership_changed=(baseline_ids.sort(-1).values != ids.sort(-1).values).any(-1),
        weight_l1=(changed_dense - baseline_dense).abs().sum(-1),
        baseline_weight_scale=baseline_w.sum(-1), weight_scale=w.sum(-1),
        baseline_ids=baseline_ids, selected_ids=ids, baseline_weights=baseline_w, selected_weights=w)
    return w, ids, telemetry


class TimedNativeRouter:
    """Per-request timing; caller must supply actual token rows, including padding.

    No global token counter is used. Every call is auditable via the supplied sink.
    Synchronization overhead is intentional in a mechanism pilot and must be timed.
    """
    def __init__(self, policy, *, top_k, sink):
        self.policy, self.top_k, self.sink = policy, top_k, sink
        self.last = {}
        self.rows_seen = 0
        self.controller_seconds = 0.

    def __call__(self, logits, select, contexts):
        started = time.perf_counter()
        if len(contexts) != len(logits):
            raise ValueError("Missing request/token attribution")
        active, reasons = [], []
        p = self.policy
        for c in contexts:
            if not c.request_id:
                raise ValueError("Missing request identity")
            if c.valid:
                if c.output_token <= self.last.get(c.request_id, -10**12):
                    raise ValueError("Duplicate/out-of-order request token at native layer")
                self.last[c.request_id] = c.output_token
            reason = ("padding" if not c.valid else "not_reasoning" if not c.reasoning else
                      "before_pulse" if c.output_token < p.start_token else
                      "after_pulse" if p.pulse_tokens is not None and c.output_token >= p.start_token + p.pulse_tokens else
                      "noop" if p.action == "noop" or p.dose == 0 else "applied")
            active.append(reason == "applied")
            reasons.append(reason)
        w, ids, changes = manipulate(logits, select, p, top_k=self.top_k,
                                     active=torch.tensor(active, device=logits.device))
        values = {k: v.detach().cpu().tolist() for k, v in changes.items()}
        for index, context in enumerate(contexts):
            self.sink(dict(schema_version=1, request_id=context.request_id,
                output_token=context.output_token, layer=p.layer, reasoning=context.reasoning,
                reason=reasons[index], action=p.action, dose=p.dose,
                **{k: v[index] for k, v in values.items()}))
        self.rows_seen += sum(c.valid for c in contexts)
        self.controller_seconds += time.perf_counter() - started
        return w, ids


def calibrate_dose(logits, select, policy, *, top_k, doses, target=.05, metric="membership_changed"):
    """Measured dose grid, independently called for positive, negative and sham groups.

    Calibration prefixes must be development-only. Discrete set changes can make
    a target unattainable; report the error instead of asserting a matched dose.
    """
    if metric not in ("membership_changed", "weight_l1") or target < 0:
        raise ValueError("Unknown dose metric")
    rows = []
    for dose in doses:
        _, _, telemetry = manipulate(logits, select, replace(policy, dose=dose), top_k=top_k)
        rows.append(dict(dose=dose, membership_changed=float(telemetry["membership_changed"].float().mean()),
                         weight_l1=float(telemetry["weight_l1"].mean())))
    if not rows:
        raise ValueError("Empty calibration grid")
    best = min(rows, key=lambda r: (abs(r[metric] - target), abs(r["dose"])))
    return dict(target=target, metric=metric, selected=best,
                absolute_error=abs(best[metric] - target), grid=rows, policy=policy.to_dict())


def qualify_noop(native, instrumented, *, input_binding, backend, atol=0., rtol=0.):
    """Evidence for one backend/precision/revision, including outputs and attribution."""
    required = ("ids", "weights", "output", "attribution")
    if any(k not in native or k not in instrumented for k in required):
        raise ValueError("Incomplete no-op parity evidence")
    ids_equal = torch.equal(native["ids"], instrumented["ids"])
    weights_equal = torch.allclose(native["weights"], instrumented["weights"], atol=atol, rtol=rtol)
    outputs_equal = torch.allclose(native["output"], instrumented["output"], atol=atol, rtol=rtol)
    attribution_equal = native["attribution"] == instrumented["attribution"]
    return dict(schema_version=1, status="passed" if all((ids_equal, weights_equal, outputs_equal, attribution_equal)) else "failed",
        input_binding=input_binding, backend=backend, selected_ids_equal=ids_equal,
        dispatched_weights_equal=weights_equal, output_equal=outputs_equal,
        attribution_equal=attribution_equal, atol=atol, rtol=rtol)

from __future__ import annotations

import torch

from moe_exp.moe_guiding.config import RoutingConfig


def _route(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    config: RoutingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gating_output.ndim != 2 or topk != 2 or gating_output.shape[-1] < 3:
        raise ValueError(
            "Expected [num_tokens, num_experts] logits, top-2, and at least 3 experts."
        )

    # The margin uses the full softmax, before any selected-expert normalization.
    probabilities = torch.softmax(gating_output.float(), dim=-1)
    top_p, top_ids = probabilities.topk(2, dim=-1, sorted=True)
    margin = 2.0 * top_p.sum(dim=-1) - 1.0
    triggered = margin > config.threshold
    if config.condition == "baseline":
        triggered = torch.zeros_like(triggered)

    # Exclude top-1 when breaking minimum ties so the two IDs are always distinct.
    least_logits = gating_output.float().scatter(-1, top_ids[:, :1], float("inf"))
    least_ids = least_logits.argmin(dim=-1)
    second_ids = torch.where(triggered, least_ids, top_ids[:, 1])
    ids = torch.stack((top_ids[:, 0], second_ids), dim=-1)
    weights = probabilities.gather(-1, ids) if config.condition == "selected" else top_p
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.contiguous(), ids.to(torch.int32).contiguous(), triggered


def margin_route(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM callback: margin > 0.4, retaining selected experts' own probabilities."""
    weights, ids, _ = _route(gating_output, topk, renormalize, RoutingConfig())
    return weights, ids


class MarginRouter:
    """Configurable callback with bounded, device-local counters for eager runs.

    Counters describe token-layer evaluations, including any padding supplied by
    the engine. They are synchronized only when snapshot() is called after a run.
    """

    def __init__(self, config: RoutingConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.tokens = 0
        self.interventions: torch.Tensor | None = None

    def __call__(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, ids, triggered = _route(gating_output, topk, renormalize, self.config)
        self.calls += 1
        self.tokens += gating_output.shape[0]
        count = triggered.sum().detach()
        if self.interventions is None:
            self.interventions = count
        else:
            self.interventions.add_(count)
        return weights, ids

    def snapshot(self) -> dict:
        count = 0 if self.interventions is None else int(self.interventions.item())
        return {
            "calls": self.calls,
            "token_evaluations": self.tokens,
            "interventions": count,
            "intervention_rate": count / self.tokens if self.tokens else None,
        }

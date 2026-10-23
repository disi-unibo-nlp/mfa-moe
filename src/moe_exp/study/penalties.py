"""Restore cumulative completion history through vLLM's original penalty operator.

Vendored verbatim from `research_campaign_v3_r6.penalties`; the study engine loads it by the string
`moe_exp.study.penalties:PrefixPenalties`.
"""
from __future__ import annotations

from moe_exp.receipts import require

KEY = "research_prefix_penalties"


def validate(params):
    value = (params.extra_args or {}).get(KEY)
    if value is None:
        return None
    require(set(value) == {"prefix", "presence", "frequency", "repetition"} and
            all(type(t) is int and t >= 0 for t in value["prefix"]), "invalid penalty history")
    require(params.presence_penalty == params.frequency_penalty == 0 and params.repetition_penalty == 1,
            "prefix history would be penalized twice")
    return value


def processor_class():
    from vllm.v1.sample.logits_processor import LogitsProcessor, MoveDirectionality
    import torch
    from vllm.v1.sample.ops.penalties import apply_all_penalties

    class PrefixPenalties(LogitsProcessor):
        @classmethod
        def validate_params(cls, params):
            validate(params)

        def __init__(self, vllm_config, device, is_pin_memory):
            self.rows = {}

        def is_argmax_invariant(self):
            return False

        def update_state(self, update):
            if update is None:
                return
            for i in update.removed:
                self.rows.pop(i, None)
            for i, params, prompt, output in update.added:
                self.rows.pop(i, None)
                value = validate(params)
                if value is not None:
                    self.rows[i] = (value, prompt, output)
            for left, right, direction in update.moved:
                a, b = self.rows.pop(left, None), self.rows.pop(right, None)
                if a is not None:
                    self.rows[right] = a
                if direction == MoveDirectionality.SWAP and b is not None:
                    self.rows[left] = b

        def apply(self, logits):
            for i, (value, prompt, output) in self.rows.items():
                device = logits.device
                history = value["prefix"] + list(output)
                logits[i:i+1] = apply_all_penalties(
                    logits[i:i+1], torch.tensor([prompt], device=device),
                    torch.tensor([value["presence"]], device=device),
                    torch.tensor([value["frequency"]], device=device),
                    torch.tensor([value["repetition"]], device=device), [history])
            return logits
    return PrefixPenalties


def __getattr__(name):
    if name == "PrefixPenalties":
        return processor_class()
    raise AttributeError(name)

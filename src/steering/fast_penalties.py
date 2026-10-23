"""Incremental prefix-history penalties for continuous batching.

Same contract and operator order as `research_campaign_v3_r6.penalties.PrefixPenalties`
(repetition over prompt|history, then frequency, then presence over history =
saved prefix + generated tokens), but per-row device state is built once and
updated with new tokens only, instead of rebuilding prompt/history tensors on
every decode step.
"""
from __future__ import annotations

from steering.penalties import validate


def processor_class():
    import torch
    from vllm.v1.sample.logits_processor import LogitsProcessor, MoveDirectionality

    class FastPrefixPenalties(LogitsProcessor):
        @classmethod
        def validate_params(cls, params):
            validate(params)

        def __init__(self, vllm_config, device, is_pin_memory):
            self.device = device
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
                neutral = value is None or (value['presence'] == 0 and value['frequency'] == 0 and
                                            value['repetition'] == 1)
                if not neutral:
                    self.rows[i] = {'value': value, 'prompt': prompt, 'output': output, 'seen': 0,
                                    'counts': None, 'prompt_mask': None}
            for left, right, direction in update.moved:
                a, b = self.rows.pop(left, None), self.rows.pop(right, None)
                if a is not None:
                    self.rows[right] = a
                if direction == MoveDirectionality.SWAP and b is not None:
                    self.rows[left] = b

        def _sync(self, row, vocab):
            if row['counts'] is None:
                counts = torch.zeros(vocab + 1, dtype=torch.long, device=self.device)
                history = torch.tensor(row['value']['prefix'], dtype=torch.long, device=self.device)
                if len(history):
                    counts.scatter_add_(0, history, torch.ones_like(history))
                row['counts'] = counts
                if row['value']['repetition'] != 1:
                    mask = torch.zeros(vocab + 1, dtype=torch.long, device=self.device)
                    prompt = torch.tensor(list(row['prompt']), dtype=torch.long, device=self.device)
                    mask.scatter_add_(0, prompt, torch.ones_like(prompt))
                    row['prompt_mask'] = mask[:vocab] > 0
            output = row['output']
            if len(output) > row['seen']:
                new = torch.tensor(output[row['seen']:], dtype=torch.long, device=self.device)
                new.masked_fill_(new < 0, vocab)
                row['counts'].scatter_add_(0, new, torch.ones_like(new))
                row['seen'] = len(output)
            return row['counts'][:vocab]

        def apply(self, logits):
            vocab = logits.shape[-1]
            for i, row in self.rows.items():
                counts = self._sync(row, vocab)[None]
                mask = counts > 0
                value = row['value']
                part = logits[i:i+1]
                if value['repetition'] != 1:
                    from vllm._custom_ops import apply_repetition_penalties
                    apply_repetition_penalties(part, row['prompt_mask'][None], mask,
                                               torch.tensor([value['repetition']], device=logits.device,
                                                            dtype=logits.dtype))
                part -= torch.tensor([value['frequency']], device=logits.device).unsqueeze(1) * counts
                part -= torch.tensor([value['presence']], device=logits.device).unsqueeze(1) * mask
                logits[i:i+1] = part
            return logits
    return FastPrefixPenalties


def __getattr__(name):
    if name == 'FastPrefixPenalties':
        return processor_class()
    raise AttributeError(name)

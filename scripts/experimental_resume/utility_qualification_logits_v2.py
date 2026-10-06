"""Deterministic closure/recompute fixture; exact no-op on utility requests."""
from __future__ import annotations

KEY = 'utility_engineering_token_sequence'


def token_sequence(params):
    sequence = (params.extra_args or {}).get(KEY)
    if sequence is not None and (not isinstance(sequence, list) or not sequence or
            len(sequence) > 1024 or any(type(i) is not int or i < 0 for i in sequence)):
        raise ValueError('engineering token fixture must contain at most 1024 exact IDs')
    return sequence


def processor_class():
    from vllm.v1.sample.logits_processor import LogitsProcessor, MoveDirectionality

    class UtilityQualificationTokens(LogitsProcessor):
        @classmethod
        def validate_params(cls, sampling_params):
            token_sequence(sampling_params)

        def __init__(self, vllm_config, device, is_pin_memory):
            self.rows = {}

        def is_argmax_invariant(self):
            return False

        def update_state(self, batch_update):
            if batch_update is None:
                return
            for index in batch_update.removed:
                self.rows.pop(index, None)
            for index, params, prompt, output in batch_update.added:
                self.rows.pop(index, None)
                sequence = token_sequence(params)
                if sequence is not None:
                    self.rows[index] = (sequence, output)
            for left, right, direction in batch_update.moved:
                a, b = self.rows.pop(left, None), self.rows.pop(right, None)
                if a is not None:
                    self.rows[right] = a
                if direction == MoveDirectionality.SWAP and b is not None:
                    self.rows[left] = b

        def apply(self, logits):
            for index, (sequence, output) in self.rows.items():
                position = len(output)
                if position < len(sequence):
                    token = sequence[position]
                    if token >= logits.shape[1]:
                        raise ValueError('engineering token outside vocabulary')
                    logits[index].fill_(float('-inf'))
                    logits[index, token] = 0
            return logits

    return UtilityQualificationTokens


def __getattr__(name):
    if name == 'UtilityQualificationTokens':
        return processor_class()
    raise AttributeError(name)

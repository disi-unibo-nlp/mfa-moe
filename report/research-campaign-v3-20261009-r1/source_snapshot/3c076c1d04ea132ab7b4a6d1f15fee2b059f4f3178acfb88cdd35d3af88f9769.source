"""Compact 16k engineering fixture; exact no-op without its explicit key."""
from __future__ import annotations

KEY = 'routing_engineering_repeat'


def validate(params):
    value = (params.extra_args or {}).get(KEY)
    if value is not None:
        if (not isinstance(value, dict) or set(value) != {'token_id', 'cap', 'overrides'} or
                type(value['token_id']) is not int or value['token_id'] < 0 or
                type(value['cap']) is not int or not 1 <= value['cap'] <= 16384 or
                not isinstance(value['overrides'], dict) or any(
                    not str(k).isdigit() or not 0 <= int(k) < value['cap'] or type(v) is not int or v < 0
                    for k, v in value['overrides'].items())):
            raise ValueError('invalid compact engineering token fixture')
    return value


def processor_class():
    from vllm.v1.sample.logits_processor import LogitsProcessor, MoveDirectionality

    class EngineeringRepeat(LogitsProcessor):
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
            for index in update.removed:
                self.rows.pop(index, None)
            for index, params, prompt, output in update.added:
                self.rows.pop(index, None)
                value = validate(params)
                if value is not None:
                    self.rows[index] = (value, output)
            for left, right, direction in update.moved:
                a, b = self.rows.pop(left, None), self.rows.pop(right, None)
                if a is not None:
                    self.rows[right] = a
                if direction == MoveDirectionality.SWAP and b is not None:
                    self.rows[left] = b

        def apply(self, logits):
            for index, (value, output) in self.rows.items():
                position = len(output)
                if position < value['cap']:
                    token = value['overrides'].get(str(position), value['token_id'])
                    if token >= logits.shape[1]:
                        raise ValueError('engineering token outside vocabulary')
                    logits[index].fill_(float('-inf'))
                    logits[index, token] = 0
            return logits

    return EngineeringRepeat


def __getattr__(name):
    if name == 'EngineeringRepeat':
        return processor_class()
    raise AttributeError(name)

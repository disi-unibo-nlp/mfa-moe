"""Normalize Hugging Face chat-template results without changing reader text."""
from collections.abc import Mapping


def normalize_prompt_ids(value):
    if isinstance(value, Mapping):
        if 'input_ids' not in value:
            raise ValueError('reader chat template returned no input_ids')
        value = value['input_ids']
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('reader prompt must be one nonempty token sequence')
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError('reader prompt contains invalid IDs or multiple sequences')
    return list(value)


def reader_prompt_ids(tokenizer, messages):
    return normalize_prompt_ids(tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, enable_thinking=True, reasoning_effort='low', return_dict=False))

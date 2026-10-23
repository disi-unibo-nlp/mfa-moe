"""Teacher-forced forward pass over a saved prompt and completion: router signals of the generated tokens."""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, showing all your work clearly. "
    "State your final answer at the end."
)

_CHAT_TEMPLATE_FALLBACK = (
    "System: {system}\n\nUser: {user}\n\nAssistant:"
)


def _format_prompt(
    tokenizer: PreTrainedTokenizerBase,
    problem: str,
    system_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
) -> str:
    """Apply the tokenizer's chat template; fall back to plain text."""
    sys_msg = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    chat_messages = messages or [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": problem},
    ]
    try:
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError):
        if messages:
            return "\n\n".join(
                f"{message['role'].title()}: {message['content']}"
                for message in chat_messages
            ) + "\n\nAssistant:"
        return _CHAT_TEMPLATE_FALLBACK.format(
            system=sys_msg,
            user=problem,
        )


def _find_prompt_length(
    tokenizer: PreTrainedTokenizerBase,
    formatted_prompt: str,
    full_text: str,
) -> int:
    """Find the generated-token boundary in the jointly tokenized text.

    Fast-tokenizer character offsets make boundary-merged tokens explicit: a
    token that straddles the prompt/continuation join belongs to the generated
    slice because it contains continuation characters. Slow tokenizers fall
    back to the longest common token prefix.
    """
    try:
        encoding = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoding["offset_mapping"][0].tolist()
        boundary = len(formatted_prompt)
        for token_index, (start, end) in enumerate(offsets):
            if end > start and end > boundary:
                return token_index
        return len(offsets)
    except (KeyError, NotImplementedError, TypeError, ValueError):
        pass

    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"][0]

    # Fast path: if the prompt tokens are an exact prefix of full tokens, use that length
    prompt_len = len(prompt_ids)
    if prompt_len <= len(full_ids) and torch.equal(full_ids[:prompt_len], prompt_ids):
        return prompt_len

    common_prefix = 0
    for prompt_id, full_id in zip(prompt_ids.tolist(), full_ids.tolist(), strict=False):
        if prompt_id != full_id:
            break
        common_prefix += 1
    return common_prefix


def extract_logs_single_pass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem: str,
    cot_text: str,
    extract_hidden_states: bool = False,
    system_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    layer_indices: list[int] | None = None,
    token_replay: dict | None = None,
    routing_details: dict | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward pass with the full prompt + CoT to extract model logs.

    With token_replay the saved prompt and completion token IDs are fed as they are. Without it, the prompt
    is rebuilt from system_prompt/messages, which must be the SAME as at generation time
    (TraceRecord.system_prompt; None = the default SYSTEM_PROMPT), so the extracted routing is conditioned
    on the generation-time context.

    Returns:
        router_logits: (num_layers, seq_len, num_experts)
        If extract_hidden_states is True, also returns:
        hidden_states: (num_layers, seq_len, hidden_size)  [Only for the generated part]
        NOTE: hidden_states are decoder-block inputs, matching Hugging Face's
        hidden_states[:-1], not the post-attention representations at the router.
        The signals are copied to CPU as the selected layers execute (router_adapters.stream_router_signals).
    """
    first_device = next(model.parameters()).device

    if token_replay is not None:
        from models.token_replay import validate_token_replay
        prompt_ids, completion_ids = validate_token_replay(token_replay, tokenizer, cot_text)
        ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        prompt_len = len(prompt_ids)
    else:
        formatted_prompt = _format_prompt(tokenizer, problem, system_prompt, messages)
        full_text = formatted_prompt + cot_text
        inputs = tokenizer(full_text, return_tensors="pt")
        prompt_len = _find_prompt_length(tokenizer, formatted_prompt, full_text)
    assert inputs["input_ids"].shape[0] == 1, (
        "extract_logs_single_pass only supports batch_size=1"
    )
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    # Router and hidden-state extraction only needs the decoder backbone. The
    # CausalLM wrapper also computes an auxiliary load-balancing loss when
    # router outputs are requested, materializing a large token/expert one-hot.
    from models.router_adapters import stream_router_signals
    forward_model = getattr(model, "model", model)
    forward_kwargs = {**inputs, "use_cache": False, "return_dict": True}
    if forward_model is model and "logits_to_keep" in inspect.signature(model.forward).parameters:
        forward_kwargs["logits_to_keep"] = 1  # the whole model would otherwise compute logits for every token
    return stream_router_signals(
        model, forward_model, forward_kwargs, prompt_len,
        None if layer_indices is None else set(layer_indices), extract_hidden_states, routing_details,
    )

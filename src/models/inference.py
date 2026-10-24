"""Teacher-forced forward pass over a saved prompt and completion: router signals of the generated tokens."""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def extract_logs_single_pass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cot_text: str,
    token_replay: dict,
    extract_hidden_states: bool = False,
    layer_indices: list[int] | None = None,
    routing_details: dict | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward pass over the saved prompt and completion token IDs and extract the model logs.

    `token_replay` (metadata.token_replay of the trace, checked against `tokenizer` and `cot_text`) holds the
    token IDs that are fed as they are: nothing is re-tokenised and no chat template is applied, so the extracted
    routing is conditioned on exactly the context the model generated in.

    Returns:
        router_logits: (num_layers, seq_len, num_experts)
        If extract_hidden_states is True, also returns:
        hidden_states: (num_layers, seq_len, hidden_size)  [Only for the generated part]
        NOTE: hidden_states are decoder-block inputs, matching Hugging Face's
        hidden_states[:-1], not the post-attention representations at the router.
        The signals are copied to CPU as the selected layers execute (router_adapters.stream_router_signals).
    """
    from models.token_replay import validate_token_replay

    first_device = next(model.parameters()).device
    prompt_ids, completion_ids = validate_token_replay(token_replay, tokenizer, cot_text)
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
    prompt_len = len(prompt_ids)
    inputs = {"input_ids": ids.to(first_device), "attention_mask": torch.ones_like(ids).to(first_device)}

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

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, showing all your work clearly. "
    "State your final answer at the end."
)

SYSTEM_PROMPT_SELFCHECK = (
    "You are a helpful math assistant. "
    "Solve the problem step by step. After each step, verify that it is correct "
    "before moving on. If you find an error or inconsistency, explicitly state "
    "what went wrong and correct it. Show all your work clearly. "
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


def generate_cot(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problems: list[str],
    max_new_tokens: int = 1024,
    batch_size: int = 1,
    system_prompt: str | None = None,
) -> list[str]:
    """Generate chain-of-thought text for each problem using greedy decoding.

    Returns a list of generated strings (same length as `problems`).
    Model logs (router logits, hidden states) are NOT captured here;
    that is handled in Experiment 2.
    """
    results: list[str] = []

    # Determine the device of the first model parameter so inputs land there.
    first_device = next(model.parameters()).device

    for start in tqdm(range(0, len(problems), batch_size), desc="Generating"):
        batch_problems = problems[start : start + batch_size]
        formatted = [_format_prompt(tokenizer, p, system_prompt) for p in batch_problems]

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for out in output_ids:
            generated_tokens = out[input_len:]
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(text)

    return results


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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward pass with the full prompt + CoT to extract model logs.

    system_prompt must be the SAME prompt used when the CoT was generated
    (TraceRecord.system_prompt; None = the default SYSTEM_PROMPT), so the
    extracted routing is conditioned on the generation-time context.

    Returns:
        router_logits: (num_layers, seq_len, num_experts)
        If extract_hidden_states is True, also returns:
        hidden_states: (num_layers, seq_len, hidden_size)  [Only for the generated part]
        NOTE: hidden_states are the PRE-MoE representations (input to the router),
        so that layer i hidden state is the representation the router at layer i sees.
    """
    first_device = next(model.parameters()).device
    
    formatted_prompt = _format_prompt(tokenizer, problem, system_prompt, messages)
    full_text = formatted_prompt + cot_text

    # Tokenize the full text once to avoid boundary-merge issues
    inputs = tokenizer(full_text, return_tensors="pt")
    assert inputs["input_ids"].shape[0] == 1, (
        "extract_logs_single_pass only supports batch_size=1"
    )
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    # Find the prompt length within the jointly-tokenized sequence
    prompt_len = _find_prompt_length(tokenizer, formatted_prompt, full_text)

    # Router and hidden-state extraction only needs the decoder backbone. The
    # CausalLM wrapper also computes an auxiliary load-balancing loss when
    # router outputs are requested, materializing a large token/expert one-hot.
    forward_model = getattr(model, "model", model)
    forward_kwargs = {
        **inputs,
        "use_cache": False,
        "output_router_logits": True,
        "output_hidden_states": extract_hidden_states,
        "return_dict": True,
    }
    if forward_model is model and "logits_to_keep" in inspect.signature(model.forward).parameters:
        forward_kwargs["logits_to_keep"] = 1

    with torch.inference_mode():
        outputs = forward_model(**forward_kwargs)

    extracted_logits = []
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    model_types = {
        getattr(config, "model_type", None),
        getattr(text_config, "model_type", None),
    }
    router_outputs_are_probabilities = bool(
        model_types & {"qwen3_5_moe", "qwen3_5_moe_text"}
    )
    
    if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
        requested_layers = None if layer_indices is None else set(layer_indices)
        for layer_index, layer_logits in enumerate(outputs.router_logits):
            if requested_layers is not None and layer_index not in requested_layers:
                continue
            # Hugging Face usually outputs router logits as a tuple of length num_layers.
            # Depending on the model, it might be flattened (batch_size * seq_len, num_experts).
            
            # Ensure shape is (batch_size, seq_len, num_experts)
            if layer_logits.ndim == 2:
                # Typically (batch_size * seq_len, num_experts)
                layer_logits = layer_logits.view(1, -1, layer_logits.shape[-1])
            if router_outputs_are_probabilities:
                layer_logits = layer_logits.clamp_min(torch.finfo(layer_logits.dtype).tiny).log()
                
            # Extract just the generations part
            gen_logits = layer_logits[0, prompt_len:, :].cpu()
            extracted_logits.append(gen_logits)
            
    # Stack into (num_layers, gen_seq_len, num_experts)
    if extracted_logits:
        router_tensor = torch.stack(extracted_logits, dim=0)
    else:
        router_tensor = torch.empty(0)
        
    if extract_hidden_states:
        extracted_hidden = []
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            # hidden_states is a tuple of length (num_layers + 1):
            #   [0] = embedding output = input to layer 0
            #   [1] = output of layer 0 = input to layer 1
            #   ...
            #   [L] = output of layer L-1
            #
            # In OLMoE (and standard MoE transformers), within each layer the
            # computation is: input → attention → router → MoE FFN → output.
            # The router at layer i operates on the post-attention representation
            # inside that layer. The exact post-attention state is not exposed by
            # HuggingFace's output, but hidden_states[i] (the input to layer i,
            # i.e., the output of layer i-1) is the closest available signal and
            # is highly correlated with the actual router input (they differ only
            # by the attention sublayer of layer i).
            #
            # We take hidden_states[:-1] to get indices [0..L-1], matching the
            # L router logit tensors.
            layer_hidden = outputs.hidden_states[:-1]
            for layer_index, h in enumerate(layer_hidden):
                if requested_layers is not None and layer_index not in requested_layers:
                    continue
                gen_h = h[0, prompt_len:, :].cpu()
                extracted_hidden.append(gen_h)
        if extracted_hidden:
            hidden_tensor = torch.stack(extracted_hidden, dim=0)
        else:
            hidden_tensor = torch.empty(0)
        return router_tensor, hidden_tensor

    return router_tensor

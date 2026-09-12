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


def _generated_tokens_to_cpu(tensor: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Copy the continuation without retaining the full sequence's storage."""
    if tensor.ndim == 2:
        tensor = tensor.view(1, -1, tensor.shape[-1])
    return tensor[0, prompt_len:, :].detach().to(device="cpu", copy=True)


def _extract_qwen_logs_to_cpu(
    forward_model: torch.nn.Module,
    layers: torch.nn.ModuleList,
    forward_kwargs: dict,
    prompt_len: int,
    requested_layers: set[int] | None,
    extract_hidden_states: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Capture selected Qwen3.5 MoE signals without accumulating GPU outputs.

    The gate's first output contains full routing probabilities. Decoder inputs
    match ``output_hidden_states[:-1]`` (not the post-attention router inputs).
    Only generated tokens are copied, while the forward still sees full context.
    """
    selected = [i for i in range(len(layers)) if requested_layers is None or i in requested_layers]
    routers: dict[int, torch.Tensor] = {}
    hidden: dict[int, torch.Tensor] = {}
    handles = []

    def router_hook(layer_index):
        def capture(_module, _args, output):
            probabilities = _generated_tokens_to_cpu(output[0], prompt_len)
            # Convert on CPU: a log/softmax round trip preserves the full router
            # distribution, without allocating another full sequence on GPU.
            routers[layer_index] = probabilities.clamp_min_(
                torch.finfo(probabilities.dtype).tiny
            ).log_()
        return capture

    def hidden_hook(layer_index):
        def capture(_module, args, kwargs):
            block_input = args[0] if args else kwargs["hidden_states"]
            hidden[layer_index] = _generated_tokens_to_cpu(block_input, prompt_len)
        return capture

    try:
        for layer_index in selected:
            layer = layers[layer_index]
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is None:
                raise RuntimeError(f"Cannot capture Qwen router at decoder layer {layer_index}")
            handles.append(gate.register_forward_hook(router_hook(layer_index)))
            if extract_hidden_states:
                handles.append(layer.register_forward_pre_hook(
                    hidden_hook(layer_index), with_kwargs=True
                ))

        # HF's automatic collectors would otherwise hold every layer's full
        # sequence on GPU until the end, even when only a few layers are wanted.
        with torch.inference_mode():
            forward_model(**{
                **forward_kwargs,
                "output_router_logits": False,
                "output_hidden_states": False,
                "output_attentions": False,
            })
    finally:
        for handle in handles:
            handle.remove()

    if any(i not in routers or (extract_hidden_states and i not in hidden) for i in selected):
        raise RuntimeError("Qwen forward did not capture all requested decoder layers")
    router_tensor = torch.stack([routers[i] for i in selected]) if selected else torch.empty(0)
    if extract_hidden_states:
        hidden_tensor = torch.stack([hidden[i] for i in selected]) if selected else torch.empty(0)
        return router_tensor, hidden_tensor
    return router_tensor


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

    system_prompt must be the SAME prompt used when the CoT was generated
    (TraceRecord.system_prompt; None = the default SYSTEM_PROMPT), so the
    extracted routing is conditioned on the generation-time context.

    Returns:
        router_logits: (num_layers, seq_len, num_experts)
        If extract_hidden_states is True, also returns:
        hidden_states: (num_layers, seq_len, hidden_size)  [Only for the generated part]
        NOTE: hidden_states are decoder-block inputs, matching Hugging Face's
        hidden_states[:-1], not the post-attention representations at the router.
        Qwen3.5 MoE signals are copied to CPU as the selected layers execute.
    """
    first_device = next(model.parameters()).device
    
    if token_replay is not None:
        from moe_exp.models.token_replay import validate_token_replay
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

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    model_types = {
        getattr(config, "model_type", None),
        getattr(text_config, "model_type", None),
    }
    router_outputs_are_probabilities = bool(
        model_types & {"qwen3_5_moe", "qwen3_5_moe_text"}
    )
    requested_layers = None if layer_indices is None else set(layer_indices)
    from moe_exp.models.router_adapters import available_router_layers, stream_router_signals
    if available_router_layers(model) is not None:
        return stream_router_signals(
            model, forward_model, forward_kwargs, prompt_len,
            requested_layers, extract_hidden_states, routing_details,
        )
    if router_outputs_are_probabilities:
        # Text-only backbones expose .layers; the multimodal backbone exposes
        # the same decoder through .language_model.layers.
        decoder = getattr(forward_model, "language_model", forward_model)
        layers = getattr(decoder, "layers", None)
        if isinstance(layers, torch.nn.ModuleList):
            return _extract_qwen_logs_to_cpu(
                forward_model, layers, forward_kwargs, prompt_len,
                requested_layers, extract_hidden_states,
            )

    with torch.inference_mode():
        outputs = forward_model(**forward_kwargs)

    extracted_logits = []
    
    if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
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

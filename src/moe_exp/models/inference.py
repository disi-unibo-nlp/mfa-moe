from __future__ import annotations

import torch
from tqdm import tqdm
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
) -> str:
    """Apply the tokenizer's chat template; fall back to plain text."""
    sys_msg = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": problem},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
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
    """Find the token length of the prompt within the full tokenized sequence.

    Tokenizes the full text once and finds where the prompt ends by checking
    the prefix token IDs match. This avoids the boundary-merge issue where
    tokenizing prompt and full_text separately can produce different tokens
    at the join point.
    """
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"][0]

    # Fast path: if the prompt tokens are an exact prefix of full tokens, use that length
    prompt_len = len(prompt_ids)
    if prompt_len <= len(full_ids) and torch.equal(full_ids[:prompt_len], prompt_ids):
        return prompt_len

    # Slow path: decode incrementally to find the boundary.
    # Find the shortest prefix of full_ids whose decoded text covers formatted_prompt.
    prompt_char_len = len(formatted_prompt)
    for i in range(1, len(full_ids) + 1):
        decoded = tokenizer.decode(full_ids[:i], skip_special_tokens=False)
        if len(decoded) >= prompt_char_len:
            return i

    return prompt_len  # fallback


def extract_logs_single_pass(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem: str,
    cot_text: str,
    extract_hidden_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward pass with the full prompt + CoT to extract model logs.
    Returns:
        router_logits: (num_layers, seq_len, num_experts)
        If extract_hidden_states is True, also returns:
        hidden_states: (num_layers, seq_len, hidden_size)  [Only for the generated part]
        NOTE: hidden_states are the PRE-MoE representations (input to the router),
        so that layer i hidden state is the representation the router at layer i sees.
    """
    first_device = next(model.parameters()).device
    
    formatted_prompt = _format_prompt(tokenizer, problem)
    full_text = formatted_prompt + cot_text

    # Tokenize the full text once to avoid boundary-merge issues
    inputs = tokenizer(full_text, return_tensors="pt")
    assert inputs["input_ids"].shape[0] == 1, (
        "extract_logs_single_pass only supports batch_size=1"
    )
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    # Find the prompt length within the jointly-tokenized sequence
    prompt_len = _find_prompt_length(tokenizer, formatted_prompt, full_text)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_router_logits=True,
            output_hidden_states=extract_hidden_states,
            return_dict=True
        )

    extracted_logits = []
    
    if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
        for layer_logits in outputs.router_logits:
            # Hugging Face usually outputs router logits as a tuple of length num_layers.
            # Depending on the model, it might be flattened (batch_size * seq_len, num_experts).
            
            # Ensure shape is (batch_size, seq_len, num_experts)
            if layer_logits.ndim == 2:
                # Typically (batch_size * seq_len, num_experts)
                layer_logits = layer_logits.view(1, -1, layer_logits.shape[-1])
                
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
            for h in layer_hidden:
                gen_h = h[0, prompt_len:, :].cpu()
                extracted_hidden.append(gen_h)
        if extracted_hidden:
            hidden_tensor = torch.stack(extracted_hidden, dim=0)
        else:
            hidden_tensor = torch.empty(0)
        return router_tensor, hidden_tensor

    return router_tensor

from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, showing all your work clearly. "
    "State your final answer at the end."
)

_CHAT_TEMPLATE_FALLBACK = (
    "System: {system}\n\nUser: {user}\n\nAssistant:"
)


def _format_prompt(tokenizer: PreTrainedTokenizerBase, problem: str) -> str:
    """Apply the tokenizer's chat template; fall back to plain text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
            system=SYSTEM_PROMPT,
            user=problem,
        )


def generate_cot(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problems: list[str],
    max_new_tokens: int = 1024,
    batch_size: int = 1,
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
        formatted = [_format_prompt(tokenizer, p) for p in batch_problems]

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

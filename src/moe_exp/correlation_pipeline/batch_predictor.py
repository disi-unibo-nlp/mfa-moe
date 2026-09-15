"""Batched judge predictor built on the saved GEPA program.

Prompt rendering and completion parsing are delegated to DSPy's own ChatAdapter and the
loaded program's signature, so a batched run produces byte-identical prompts and uses
identical parsing to the single-request control. Only the transport differs: N rendered
conversations are posted in one `/v1/chat/completions/batch` request.
"""

from __future__ import annotations

from typing import Any

from moe_exp.correlation_pipeline.batch_judge import (
    build_batch_payload,
    map_batch_response,
    post_batch,
)

JUDGE_INPUT_FIELDS = (
    "problem_statement",
    "previous_sentence",
    "sentence",
    "next_sentence",
)


def load_program_and_adapter(judge_program: str) -> tuple[Any, Any, Any]:
    """Load the real saved GEPA program and the adapter DSPy would use for it."""
    import dspy
    from dspy.adapters.chat_adapter import ChatAdapter

    from moe_exp.gepaLLMAsJudge.run import EpisodeJudge

    program = EpisodeJudge()
    program.load(judge_program)
    predict = program.classify
    return program, predict, ChatAdapter()


def render_conversation(adapter: Any, predict: Any, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Render one judge prompt exactly as DSPy would for the single-request path."""
    missing = [field for field in JUDGE_INPUT_FIELDS if field not in inputs]
    if missing:
        raise ValueError(f"missing judge input fields: {missing}")
    return adapter.format(predict.signature, predict.demos, inputs)


def parse_completion(adapter: Any, predict: Any, completion: str) -> str:
    """Parse a completion with DSPy's parser, then the project's label normaliser."""
    from moe_exp.gepaLLMAsJudge.metrics import parse_sentence_label

    parsed = adapter.parse(predict.signature, completion)
    return parse_sentence_label(parsed)


def classify_batch(
    items: list[dict[str, Any]],
    *,
    adapter: Any,
    predict: Any,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    timeout: float = 600.0,
) -> list[str]:
    """Classify `items` in a single batched generation request, order preserved."""
    conversations = [render_conversation(adapter, predict, item) for item in items]
    payload = build_batch_payload(
        conversations,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    response = post_batch(payload, base_url=base_url, api_key=api_key, timeout=timeout)
    mapped = map_batch_response(response, expected=len(items))
    return [parse_completion(adapter, predict, mapped[i]) for i in range(len(items))]

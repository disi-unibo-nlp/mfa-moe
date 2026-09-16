"""Batched judge predictor built on the saved GEPA program.

Prompt rendering and completion parsing are delegated to DSPy's own ChatAdapter and the
loaded program's signature, so a batched run produces byte-identical prompts and uses
identical parsing to the single-request control. Only the transport differs: N rendered
conversations are posted in one `/v1/chat/completions/batch` request.

Two APIs share that one request path:

* `classify_batch` is the strict compatibility API used by the benchmark and probe
  callers. It returns canonical label strings and raises on anything unusual.
* `classify_batch_outcomes` is the tolerant API used by the production annotation batch.
  A malformed or non-stop choice comes back as an `unknown` outcome carrying the exact
  raw completion, so one bad choice cannot discard its valid siblings.
"""

from __future__ import annotations

from typing import Any

from moe_exp.correlation_pipeline.batch_judge import (
    build_batch_payload,
    map_batch_choices,
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


def parse_failure_types() -> tuple[type[BaseException], ...]:
    """Exceptions that mean "this one completion could not be turned into a label".

    The project's normaliser raises `LabelParseError` (a `ValueError`); DSPy's adapters
    raise `AdapterParseError`, which is not a `ValueError`, so it is added when DSPy is
    importable. Transport and programming errors are deliberately not included.
    """
    types: tuple[type[BaseException], ...] = (ValueError, TypeError)
    try:
        from dspy.utils.exceptions import AdapterParseError
    except ImportError:
        return types
    return (*types, AdapterParseError)


def _request_items(
    items: list[dict[str, Any]],
    *,
    mapper: Any,
    adapter: Any,
    predict: Any,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    timeout: float,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
) -> Any:
    """Render, post and index one batch; shared by the strict and tolerant APIs."""
    conversations = [render_conversation(adapter, predict, item) for item in items]
    payload = build_batch_payload(
        conversations,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    response = post_batch(payload, base_url=base_url, api_key=api_key, timeout=timeout)
    return mapper(response, expected=len(items))


def _choice_content(choice: dict[str, Any]) -> Any:
    message = choice.get("message")
    return message.get("content") if isinstance(message, dict) else None


def _unknown_outcome(
    kind: str,
    *,
    error_type: str,
    message: str,
    finish_reason: Any,
    raw_completion: str | None,
) -> dict[str, Any]:
    """Build one soft per-choice failure; only JSON-serializable scalars are stored."""
    return {
        "status": "unknown",
        "raw_completion": raw_completion,
        "failure": {
            "kind": kind,
            "error_type": error_type,
            "message": message,
            "finish_reason": finish_reason,
        },
    }


def classify_batch_outcomes(
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
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> list[dict[str, Any]]:
    """Classify `items`, returning one outcome per input row, in input order.

    A clean choice yields exactly `{"label": <canonical label>}`. A per-choice problem
    yields an `unknown` outcome holding the exact raw completion and deterministic
    failure metadata, so the caller can persist it and resume without re-requesting it.
    A non-stop choice is unknown even when its text happens to contain a valid label,
    because a truncated answer is not a valid annotation. Structural batch failures
    (missing, duplicated or out-of-range indices, wrong choice count) and transport
    failures still raise.
    """
    mapped = _request_items(
        items,
        mapper=map_batch_choices,
        adapter=adapter,
        predict=predict,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    failure_types = parse_failure_types()
    outcomes: list[dict[str, Any]] = []
    for index in range(len(items)):
        choice = mapped[index]
        finish_reason = choice.get("finish_reason")
        content = _choice_content(choice)
        raw_completion = content if isinstance(content, str) else None
        if finish_reason not in (None, "stop"):
            outcomes.append(
                _unknown_outcome(
                    "non_stop",
                    error_type="ValueError",
                    message=f"finish_reason={finish_reason!r} is not a normal stop",
                    finish_reason=finish_reason,
                    raw_completion=raw_completion,
                )
            )
            continue
        if not isinstance(content, str):
            outcomes.append(
                _unknown_outcome(
                    "missing_content",
                    error_type="ValueError",
                    message="choice has no string message content",
                    finish_reason=finish_reason,
                    raw_completion=None,
                )
            )
            continue
        try:
            label = parse_completion(adapter, predict, content)
        except failure_types as error:
            outcomes.append(
                _unknown_outcome(
                    "label_parse",
                    error_type=type(error).__name__,
                    message=str(error),
                    finish_reason=finish_reason,
                    raw_completion=content,
                )
            )
            continue
        outcomes.append({"label": label})
    return outcomes


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
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> list[str]:
    """Classify `items` in a single batched generation request, order preserved."""
    mapped = _request_items(
        items,
        mapper=map_batch_response,
        adapter=adapter,
        predict=predict,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    return [parse_completion(adapter, predict, mapped[i]) for i in range(len(items))]

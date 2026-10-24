"""The judge client: label a batch of sentences with Qwen3.8-27B served by vLLM.

Each sentence becomes a chat prompt built by the saved DSPy program (the frozen GEPA judge). All
prompts of a batch go to the server in one request to vLLM's `/v1/chat/completions/batch` route, and
each answer is read with DSPy's own parser, so the labels equal what DSPy's one-request-at-a-time
path would give. An answer that cannot be read comes back as an `unknown` outcome that keeps the raw
text; one bad answer never discards its siblings.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from models.roster import JUDGE_MODEL
from store import WORK

MAX_TOKENS = 16384
REASONING_EFFORT = "low"
# Decoding settings of every labelling run (the model card's); they are part of the checkpoint key.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0}
# The same settings as the request's chat template and the checkpoint key use.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True, "reasoning_effort": REASONING_EFFORT,
                        "preserve_thinking": False}
API_KEY = "local-vllm-key"  # the key generation/serve_flags.py starts the server with
# The frozen judge: the GEPA program every label set was made with.
JUDGE_PROGRAM = (WORK / "mfa-moe/repo/results/gepaLLMAsJudge/qwen3.8-27b-medium-final-s42-v3"
                 / "selected_program_20260827_173300.json")

JUDGE_INPUT_FIELDS = ("problem_statement", "previous_sentence", "sentence", "next_sentence")


def question_of(messages: list[dict[str, Any]] | None, prompt: str) -> str:
    """The question the judge sees: the user message(s) of the generation prompt, or else the bare
    prompt."""
    users = [str(m["content"]) for m in (messages or []) if m.get("role") == "user"]
    return "\n\n".join(users) or prompt


def judge_inputs(question: str, units: list[dict[str, Any]], index: int) -> dict[str, str]:
    """What the judge sees for sentence `index`: the question, one neighbouring sentence each side,
    edge markers."""
    return {
        "problem_statement": question,
        "previous_sentence": units[index - 1]["text"] if index else "<START OF RESPONSE>",
        "sentence": units[index]["text"],
        "next_sentence": (units[index + 1]["text"] if index + 1 < len(units)
                          else "<END OF RESPONSE>"),
    }


def load_program_and_adapter(judge_program: str) -> tuple[Any, Any]:
    """Load the saved GEPA program's predictor and the adapter DSPy would use for it."""
    from dspy.adapters.chat_adapter import ChatAdapter

    from labeling.gepa.run import EpisodeJudge

    program = EpisodeJudge()
    program.load(judge_program)
    return program.classify, ChatAdapter()


def render_conversation(adapter: Any, predict: Any, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Render one judge prompt exactly as DSPy would."""
    missing = [field for field in JUDGE_INPUT_FIELDS if field not in inputs]
    if missing:
        raise ValueError(f"missing judge input fields: {missing}")
    return adapter.format(predict.signature, predict.demos, inputs)


def parse_completion(adapter: Any, predict: Any, completion: str) -> str:
    """Parse a completion with DSPy's parser, then the project's label normaliser."""
    from labeling.gepa.metrics import parse_sentence_label

    return parse_sentence_label(adapter.parse(predict.signature, completion))


def parse_failure_types() -> tuple[type[BaseException], ...]:
    """Exceptions that mean "this one completion cannot be turned into a label".

    The label normaliser raises `LabelParseError` (a `ValueError`); DSPy's adapters raise
    `AdapterParseError`, which is not a `ValueError`, so it is added when DSPy is importable.
    Transport and programming errors are deliberately not included.
    """
    try:
        from dspy.utils.exceptions import AdapterParseError
    except ImportError:
        return (ValueError, TypeError)
    return (ValueError, TypeError, AdapterParseError)


def batch_payload(conversations: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """One batch request carrying every conversation. The route supports neither streaming, tools
    nor `n` > 1."""
    return {
        "model": JUDGE_MODEL,
        "n": 1,
        "stream": False,
        "messages": conversations,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        **SAMPLING,
    }


def post_batch(payload: dict[str, Any], *, base_url: str, timeout: float = 600.0) -> dict[str, Any]:
    """POST one batch request and return the decoded JSON body."""
    request = urllib.request.Request(
        base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions/batch",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read().decode("utf-8"))


def map_batch_choices(response: dict[str, Any], *, expected: int) -> dict[int, dict[str, Any]]:
    """Map the raw `choices` back to conversation positions without judging their content.

    A structural problem (missing, repeated or out-of-range index, wrong count) stays fatal because
    such a response cannot be attributed to the input rows. Per-choice problems are left to the
    caller.
    """
    choices = response.get("choices")
    if not isinstance(choices, list):
        raise ValueError("batch response has no choices list")
    if len(choices) != expected:
        raise ValueError(f"batch response returned {len(choices)} choices, expected {expected}")
    mapped: dict[int, dict[str, Any]] = {}
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError("batch choice must be an object")
        index = choice.get("index")
        if type(index) is not int:
            raise ValueError("batch choice is missing an integer index")
        if index in mapped:
            raise ValueError(f"batch response repeated index {index}")
        if not 0 <= index < expected:
            raise ValueError(f"batch choice index {index} outside 0..{expected - 1}")
        mapped[index] = choice
    return mapped


def _unknown_outcome(kind: str, *, error_type: str, message: str, finish_reason: Any,
                     raw_completion: str | None) -> dict[str, Any]:
    return {
        "status": "unknown",
        "raw_completion": raw_completion,
        "failure": {"kind": kind, "error_type": error_type, "message": message,
                    "finish_reason": finish_reason},
    }


def classify_batch_outcomes(
    items: list[dict[str, Any]], *, adapter: Any, predict: Any, base_url: str
) -> list[dict[str, Any]]:
    """Classify `items`, returning one outcome per input row, in input order.

    A clean choice gives exactly `{"label": <label>}`. A choice that did not stop normally, has no
    text or cannot be parsed gives an `unknown` outcome holding the raw completion, even when the
    text happens to contain a valid label (a truncated answer is not an annotation). Transport
    failures and structural batch failures raise. An HTTP 400 rejects the whole batch when one
    conversation does not fit the judge context (a degenerate, unpunctuated "sentence" of a looping
    trace); the batch is bisected so that only the rejected conversation becomes `unknown`.
    """
    conversations = [render_conversation(adapter, predict, item) for item in items]
    try:
        response = post_batch(batch_payload(conversations), base_url=base_url)
    except urllib.error.HTTPError as error:
        if error.code != 400:
            raise
        if len(items) > 1:
            middle = len(items) // 2
            first = classify_batch_outcomes(items[:middle], adapter=adapter, predict=predict,
                                            base_url=base_url)
            return first + classify_batch_outcomes(items[middle:], adapter=adapter, predict=predict,
                                                   base_url=base_url)
        body = error.read().decode("utf-8", "replace") if error.fp is not None else ""
        return [_unknown_outcome("request_rejected", error_type="HTTPError",
                                 message=f"HTTP 400: {body[:300]}", finish_reason=None,
                                 raw_completion=None)]
    choices = map_batch_choices(response, expected=len(items))
    failure_types = parse_failure_types()
    outcomes = []
    for index in range(len(items)):
        choice = choices[index]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        raw_completion = content if isinstance(content, str) else None
        if finish_reason not in (None, "stop"):
            outcomes.append(_unknown_outcome(
                "non_stop", error_type="ValueError",
                message=f"finish_reason={finish_reason!r} is not a normal stop",
                finish_reason=finish_reason, raw_completion=raw_completion))
        elif raw_completion is None:
            outcomes.append(_unknown_outcome(
                "missing_content", error_type="ValueError",
                message="choice has no string message content",
                finish_reason=finish_reason, raw_completion=None))
        else:
            try:
                outcomes.append({"label": parse_completion(adapter, predict, raw_completion)})
            except failure_types as error:
                outcomes.append(_unknown_outcome(
                    "label_parse", error_type=type(error).__name__, message=str(error),
                    finish_reason=finish_reason, raw_completion=raw_completion))
    return outcomes

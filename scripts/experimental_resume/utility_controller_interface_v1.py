"""CPU-only request boundary for a future original-prompt utility controller.

This module selects no transition, side-reader threshold, or routing action.
It checks the token and information boundaries that an eventual GPU controller
must satisfy. It does not qualify replay, batching, or the utility policy.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Callable, Sequence


MAX_OUTPUT_TOKENS = 16_384
SIDE_INPUT_KEYS = frozenset({"problem", "emitted_prefix", "triggering_sentence"})
TRANSITIONS = frozenset({"candidate_to_verify", "approach_to_commit",
                         "failed_check_to_revise"})


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _ids(value: Sequence[int]) -> tuple[int, ...]:
    result = tuple(value)
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError("token IDs must be nonnegative integers")
    return result


@dataclass(frozen=True)
class RequestState:
    """One assignment's immutable completion history and 16k output charge."""

    uid: str
    problem: str
    prompt_token_ids_sha256: str
    completion_token_ids: tuple[int, ...] = ()
    token_sources: tuple[str, ...] = ()
    finish: str | None = None

    def __post_init__(self) -> None:
        if not self.uid or not isinstance(self.problem, str) or not self.problem:
            raise ValueError("assignment UID and original problem required")
        if len(self.prompt_token_ids_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in self.prompt_token_ids_sha256):
            raise ValueError("original prompt token hash required")
        _ids(self.completion_token_ids)
        if len(self.completion_token_ids) != len(self.token_sources) or any(
                source not in {"emitted", "injected"} for source in self.token_sources):
            raise ValueError("token-source trace differs from completion")
        if len(self.completion_token_ids) > MAX_OUTPUT_TOKENS:
            raise ValueError("emitted plus injected tokens exceed 16k")
        if self.finish not in (None, "stop", "length", "error"):
            raise ValueError("invalid finish status")

    @property
    def tokens_charged(self) -> int:
        return len(self.completion_token_ids)

    @property
    def remaining_tokens(self) -> int:
        return MAX_OUTPUT_TOKENS - self.tokens_charged

    @property
    def emitted_token_ids(self) -> tuple[int, ...]:
        return tuple(token for token, source in zip(self.completion_token_ids,
                                                     self.token_sources)
                     if source == "emitted")

    def append(self, ids: Sequence[int], *, source: str) -> RequestState:
        if self.finish is not None or source not in {"emitted", "injected"}:
            raise ValueError("cannot append after finish or with unknown source")
        addition = _ids(ids)
        if not addition or len(addition) > self.remaining_tokens:
            raise ValueError("empty output or 16k output budget exhausted")
        return replace(self,
                       completion_token_ids=self.completion_token_ids + addition,
                       token_sources=self.token_sources + (source,) * len(addition))

    def close(self, finish: str) -> RequestState:
        if self.finish is not None or finish not in {"stop", "length", "error"}:
            raise ValueError("invalid or repeated finish")
        if finish == "length" and self.remaining_tokens:
            raise ValueError("length finish requires all 16k charged tokens")
        return replace(self, finish=finish)


@dataclass(frozen=True)
class SideRequest:
    """Request metadata stays outside the three-field arm-blind reader input."""

    request_id: str
    assignment_uid: str
    transition: str
    reader_input: dict[str, str]
    emitted_token_ids_sha256: str
    charged_output_tokens_at_query: int


def make_side_request(state: RequestState, *, transition: str,
                      triggering_sentence: str,
                      decode: Callable[[Sequence[int]], str]) -> SideRequest:
    if state.finish is not None or transition not in TRANSITIONS:
        raise ValueError("finished request or unsupported transition")
    prefix = decode(state.emitted_token_ids)
    if not isinstance(prefix, str) or not isinstance(triggering_sentence, str) or not (
            triggering_sentence.strip() and prefix.rstrip().endswith(triggering_sentence.rstrip())):
        raise ValueError("side request must end at the current emitted sentence")
    if "</think>" in prefix:
        raise ValueError("reasoning has closed")
    reader_input = {"problem": state.problem, "emitted_prefix": prefix,
                    "triggering_sentence": triggering_sentence}
    if set(reader_input) != SIDE_INPUT_KEYS:
        raise ValueError("side-reader input allowlist differs")
    return SideRequest(
        request_id=digest(["utility-side-request-v1", state.uid, transition,
                           state.completion_token_ids, state.token_sources,
                           triggering_sentence]),
        assignment_uid=state.uid, transition=transition, reader_input=reader_input,
        emitted_token_ids_sha256=digest(state.emitted_token_ids),
        charged_output_tokens_at_query=state.tokens_charged)


def strict_start_vote(text: str | None, finish_reason: str) -> bool | None:
    """Only a natural-stop, exact one-key JSON boolean is a usable vote."""
    if finish_reason != "stop" or not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"start"} or type(parsed["start"]) is not bool:
        return None
    return parsed["start"]


@dataclass(frozen=True)
class SideResult:
    request_id: str
    reader: int
    finish_reason: str
    response_text: str | None
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not self.request_id or self.reader not in (0, 1):
            raise ValueError("side request ID and two-reader index required")
        if (type(self.prompt_tokens) is not int or self.prompt_tokens < 0 or
                type(self.generated_tokens) is not int or self.generated_tokens < 0 or
                not isinstance(self.elapsed_seconds, (int, float)) or
                not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0):
            raise ValueError("actual side-reader cost required, including failures")

    @property
    def vote(self) -> bool | None:
        return strict_start_vote(self.response_text, self.finish_reason)


def two_reader_accept(request: SideRequest, results: Sequence[SideResult]) -> bool:
    """An absent, ambiguous, truncated, or failed side response abstains."""
    if len(results) != 2 or {result.reader for result in results} != {0, 1} or any(
            result.request_id != request.request_id for result in results):
        raise ValueError("side-reader receipt coverage or request binding differs")
    return all(result.vote is True for result in results)

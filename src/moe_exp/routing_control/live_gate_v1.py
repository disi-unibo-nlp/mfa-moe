"""Past-only proposal boundary for a prospective routing controller.

This module makes no semantic decision.  A discovery-frozen cheap screen must
approve a lexical proposal before an optional side query is requested.  The
caller supplies the native tokenizer's decode function and records actual
side-query tokens through QueryCostLedger.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence

from .transitions_v2 import StreamingTransitionDetectorV2


TRANSITIONS = frozenset({
    "candidate_to_verify", "approach_to_commit", "failed_check_to_revise",
})


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class SideQuery:
    request_id: str
    transition: str
    token_ids: tuple[int, ...]
    problem: str
    emitted_prefix: str
    triggering_sentence: str

    def reader_input(self) -> dict[str, str]:
        """The complete allowlist for the semantic reader."""
        return {"problem": self.problem, "emitted_prefix": self.emitted_prefix,
                "triggering_sentence": self.triggering_sentence}


class LivePrefixGate:
    """Return query opportunities from only the current emitted token prefix.

    The screen is deliberately required and may inspect only a SideQuery.  A
    missing screen abstains.  A lexical proposal inside a token that already
    emits the next sentence is skipped, because branching after that token
    would change the frozen sentence-boundary estimand.
    """

    def __init__(self, supported: Sequence[str],
                 screen: Callable[[SideQuery], bool] | None = None):
        if any(label not in TRANSITIONS for label in supported):
            raise ValueError("unsupported transition")
        self.supported = frozenset(supported)
        self.screen = screen
        self.detector = StreamingTransitionDetectorV2()
        self.seen: set[str] = set()
        self.closed = False

    def observe_record(self, record: Mapping, decode: Callable[[Sequence[int]], str]) -> list[SideQuery]:
        # Never inspect emitted_text, future completion, gold, correctness,
        # class labels or any other field in a trace-like record.
        return self.observe_tokens(record["problem"], record["emitted_token_ids"], decode)

    def observe_tokens(self, problem: str, token_ids: Sequence[int],
                       decode: Callable[[Sequence[int]], str]) -> list[SideQuery]:
        ids = tuple(token_ids)
        if any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("emitted token IDs must be nonnegative integers")
        text = decode(ids)
        if not isinstance(text, str):
            raise ValueError("native decoder did not return text")
        result = self.detector.observe({"problem": problem, "emitted_token_ids": ids,
                                        "emitted_text": text})
        if result["closure"]:
            self.closed = True
        if self.closed:
            return []
        queries = []
        for event in result["events"]:
            if event.transition not in self.supported:
                continue
            # The current token prefix must stop at this completed sentence.
            if text[event.evidence_end:].strip():
                continue
            request_id = _fingerprint(["live-prefix-side-query-v1", event.transition,
                                       problem, ids, event.evidence_end])
            if request_id in self.seen:
                continue
            self.seen.add(request_id)
            query = SideQuery(request_id, event.transition, ids, problem,
                              text, event.text)
            if self.screen is not None and self.screen(query):
                queries.append(query)
        return queries


class QueryCostLedger:
    """Count actual side-model usage, including vetoes, errors and nonfires."""

    def __init__(self):
        self.records: dict[str, dict] = {}

    def record(self, request_id: str, status: str, prompt_tokens: int,
               generated_tokens: int, elapsed_seconds: float) -> None:
        if status not in {"accepted", "vetoed", "unresolved", "failed"}:
            raise ValueError("unrecognized side-query outcome")
        if request_id in self.records:
            raise ValueError("duplicate side-query request ID")
        if (type(prompt_tokens) is not int or prompt_tokens < 0
                or type(generated_tokens) is not int or generated_tokens < 0
                or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0):
            raise ValueError("invalid actual side-query cost")
        self.records[request_id] = {"status": status, "prompt_tokens": prompt_tokens,
                                    "generated_tokens": generated_tokens,
                                    "elapsed_seconds": float(elapsed_seconds)}

    def totals(self) -> dict[str, object]:
        return {"queries": len(self.records),
                "prefill_tokens": sum(x["prompt_tokens"] for x in self.records.values()),
                "generated_tokens": sum(x["generated_tokens"] for x in self.records.values()),
                "elapsed_seconds": sum(x["elapsed_seconds"] for x in self.records.values()),
                "statuses": {status: sum(x["status"] == status for x in self.records.values())
                             for status in ("accepted", "vetoed", "unresolved", "failed")}}

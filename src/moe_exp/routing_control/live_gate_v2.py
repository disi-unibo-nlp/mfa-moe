"""Versioned past-only side-query gate for a future online controller.

This binds the corrected v2.4 discovery detector.  Its lexical proposals are
not semantic judgments; a discovery-frozen screen still has to approve them.
The held same-prefix pilot does not import this gate.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .live_gate_v1 import QueryCostLedger, SideQuery, TRANSITIONS, _fingerprint
from .transitions_v24 import StreamingTransitionDetectorV2


class LivePrefixGate:
    """Offer a side query only at a completed, current-prefix sentence edge."""

    def __init__(self, supported: Sequence[str],
                 screen: Callable[[SideQuery], bool] | None = None):
        if any(label not in TRANSITIONS for label in supported):
            raise ValueError("unsupported transition")
        self.supported = frozenset(supported)
        self.screen = screen
        self.detector = StreamingTransitionDetectorV2()
        self.seen: set[str] = set()
        self.closed = False

    def observe_record(self, record: Mapping,
                       decode: Callable[[Sequence[int]], str]) -> list[SideQuery]:
        # Only the original problem and already emitted token IDs enter the
        # detector.  Do not inspect emitted_text or any future/gold fields.
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
        queries: list[SideQuery] = []
        for event in result["events"]:
            if event.transition not in self.supported:
                continue
            # Skip a boundary already followed by another emitted sentence.
            if text[event.evidence_end:].strip():
                continue
            request_id = _fingerprint(["live-prefix-side-query-v2", event.transition,
                                       problem, ids, event.evidence_end])
            if request_id in self.seen:
                continue
            self.seen.add(request_id)
            query = SideQuery(request_id, event.transition, ids, problem,
                              text, event.text)
            if self.screen is not None and self.screen(query):
                queries.append(query)
        return queries


__all__ = ["LivePrefixGate", "SideQuery", "QueryCostLedger"]

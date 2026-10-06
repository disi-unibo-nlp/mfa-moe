"""Discovery-only narrow proposal-language gate after the v2.2 candidate screen.

This is an exploratory high-precision candidate for evaluation, not a qualified
online controller. It deliberately sacrifices math-only candidate coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

VERSION = "explicit-proposal-gate-v0-discovery"
ALLOWLIST = frozenset({"problem", "emitted_prefix", "triggering_sentence"})
PROPOSAL = re.compile(
    r"\b(?:state becomes|(?:candidate|proposed|tentative|possible)\s+(?:answer|value|solution)\s*(?:is|:)|"
    r"(?:we|i)\s+(?:get|obtain|find|propose)\b)", re.I)
REJECT = re.compile(r"\b(?:the problem asks|we need to find|let'?s calculate|set up the equation)\b", re.I)


@dataclass(frozen=True)
class StartInput:
    problem: str
    emitted_prefix: str
    triggering_sentence: str

    @classmethod
    def from_record(cls, record: Mapping) -> "StartInput":
        # Only these already-visible fields are read; metadata, gold and later
        # sentences cannot affect the decision.
        problem, prefix, sentence = (record[k] for k in
                                     ("problem", "emitted_prefix", "triggering_sentence"))
        if not all(isinstance(v, str) for v in (problem, prefix, sentence)):
            raise ValueError("visible prefix input must be text")
        if not prefix.rstrip().endswith(sentence.rstrip()):
            raise ValueError("trigger must end the emitted prefix")
        return cls(problem, prefix, sentence)


def passes(record: Mapping) -> bool:
    current = StartInput.from_record(record).triggering_sentence
    return bool(PROPOSAL.search(current) and not REJECT.search(current))

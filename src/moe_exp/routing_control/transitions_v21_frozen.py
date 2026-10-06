"""Discovery-only prefix transition proposals, version 2.

This expands *starting-condition* coverage without consulting a later sentence,
class label, gold answer, correctness, or a completion field.  A proposal is
not a semantic transition and must pass a separately frozen audit before use.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from .prefix import PrefixInput, StreamingAdapter, candidates

VERSION = "prefix-transition-candidates-v2.1-discovery"

_RELATION = re.compile(r"(?<![<>=])=(?![=])|[<>≈≠]|\\(?:approx|simeq|leq?|geq?|equiv|to|Rightarrow)\b|->")
_NUMBER = re.compile(r"\d")
_MATH = re.compile(
    r"\$\$(.{1,240}?)\$\$|(?<!\$)\$([^$\n]{1,240})(?<!\\)\$(?!\$)"
    r"|\\\((.{1,240}?)\\\)|\\\[(.{1,240}?)\\\)", re.S)
_PROPOSAL_CUE = re.compile(r"\b(?:answer|result|solution|value|therefore|thus|hence|so|we\s+(?:get|find|obtain|have)|this\s+gives)\b", re.I)
_APPROACH_CUE = re.compile(r"\b(?:try|could|might|perhaps|maybe|consider|suppose|we\s+(?:can|could|should|might)|let'?s|one\s+(?:way|approach)|start\s+by)\b", re.I)
_APPROACH_OBJECT = re.compile(r"\b(?:formula|equation|identity|substitution|factor|case|split|induct|bound|count|enumerat|construct|represent|differentiat|integrat|symmetr|recurren|coordinate|vector|set\s+up|rewrite|solve|express|define|assum)\w*\b", re.I)
_FAILURE = re.compile(r"\b(?:contradict\w*|inconsisten\w*|impossible|invalid|incorrect|wrong|fails?\s+(?:the\s+)?(?:check|condition|constraint)|does(?:\s+not|n'?t)\s+(?:satisfy|work|hold|match)|violat\w*|not\s+equal)\b|≠", re.I)
_CHECK = re.compile(r"\b(?:check|substitut\w*|condition|constraint|equation|expected|require\w*|must|gives?|yields?)\b|[=<>≠≈]", re.I)
_SENTENCE_END = re.compile(r"[.!?](?=\s)|\n+")


def _completed_sentences(reasoning: str):
    """Wait for a following delimiter, so a trailing decimal point is not final."""
    left = 0
    protected = [match.span() for match in _MATH.finditer(reasoning)]
    span_index = 0
    for match in _SENTENCE_END.finditer(reasoning):
        while span_index < len(protected) and protected[span_index][1] <= match.start():
            span_index += 1
        if span_index < len(protected) and protected[span_index][0] <= match.start() < protected[span_index][1]:
            continue
        end = match.end()
        sentence = reasoning[left:end].strip()
        if sentence:
            yield left, end, sentence
        left = end


def _closed_math(sentence: str) -> list[str] | None:
    """Return short, closed math spans; reject dangling delimiters and braces."""
    if len(sentence) > 1200 or sentence.count("$") % 2 or sentence.count("$$") % 2:
        return None
    if sentence.count(r"\(") != sentence.count(r"\)") or sentence.count(r"\[") != sentence.count(r"\]"):
        return None
    if sentence.count("{") != sentence.count("}"):
        return None
    return [next(group for group in match.groups() if group is not None)
            for match in _MATH.finditer(sentence)]


def classify_sentence_v2(sentence: str) -> tuple[str, ...]:
    """Conservative lexical proposals from one completed, already emitted unit."""
    math = _closed_math(sentence)
    if math is None:
        return ()
    proposed: list[str] = []
    # A completed intermediate equality is a proposed value, even when it is
    # not introduced by "answer is".  Mere formulas without a numeric or
    # comparison claim abstain.  Do not infer that any check happened.
    math_claim = any(_RELATION.search(expr) and _NUMBER.search(expr)
                     and "..." not in expr and r"\ldots" not in expr
                     for expr in math)
    boxed = bool(re.search(r"\\boxed\s*\{", sentence)) and sentence.count("{") == sentence.count("}")
    plain_result = bool(_PROPOSAL_CUE.search(sentence) and re.search(r"\b\d+(?:\.\d+)?\b", sentence))
    if math_claim or boxed or plain_result:
        proposed.append("candidate_to_verify")
    if _APPROACH_CUE.search(sentence) and _APPROACH_OBJECT.search(sentence):
        proposed.append("approach_to_commit")
    # Require both an explicit negative judgment and visible check evidence;
    # arithmetic by itself is not a failed check.
    if _FAILURE.search(sentence) and _CHECK.search(sentence) and (math or _NUMBER.search(sentence)):
        proposed.append("failed_check_to_revise")
    return tuple(proposed)


@dataclass(frozen=True)
class TransitionProposalV2:
    transition: str
    evidence_end: int
    text: str
    basis: str


class StreamingTransitionDetectorV2:
    """Append-only proposal stream; emits only after a complete sentence."""

    def __init__(self):
        self.adapter = StreamingAdapter()
        self.last_sentence_end = 0

    def observe_tokens(self, problem: str, token_ids, decode):
        ids = tuple(token_ids)
        return self.observe({"problem": problem, "emitted_token_ids": ids,
                             "emitted_text": decode(ids)})

    def observe(self, record: Mapping) -> dict:
        prefix = PrefixInput.from_record(record)
        state = self.adapter.observe({"problem": prefix.problem,
                                      "emitted_token_ids": prefix.emitted_token_ids,
                                      "emitted_text": prefix.emitted_text})
        if state["closure"]:
            return {"version": VERSION, "closure": True, "events": [],
                    "status": "UNQUALIFIED_DISCOVERY_PROPOSALS"}
        reasoning = prefix.emitted_text
        complete_candidates = candidates(reasoning)
        events = []
        for _, end, sentence in _completed_sentences(reasoning):
            if end <= self.last_sentence_end:
                continue
            labels = set(classify_sentence_v2(sentence))
            if any(0 <= item.end <= end and item.end >= end - len(sentence) - 4
                   for item in complete_candidates):
                labels.add("candidate_to_verify")
            for label in sorted(labels):
                events.append(TransitionProposalV2(label, end, sentence,
                                                    "completed_prefix_sentence_v2"))
            self.last_sentence_end = end
        return {"version": VERSION, "closure": False, "events": events,
                "status": "UNQUALIFIED_DISCOVERY_PROPOSALS"}

"""Versioned, conservative prefix-only proposals for three registered transitions.

These are prospective trigger *candidates*. The action controller must not use a
transition until its discovery-only semantic qualification has been sealed.
Only the original problem and already emitted token IDs/text are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from .prefix import PrefixInput, StreamingAdapter, candidates


VERSION = 'prefix-transition-candidates-v1'

# The conservative patterns are fixed before discovery ratings. They seek
# explicit operations, not words like "maybe" or "verify" by themselves.
METHOD = re.compile(
    r'\b(?:substitut(?:e|ion)|factor(?:ing)?|induct(?:ion|ive)|modular|'
    r'case\s+split|enumerat(?:e|ion)|differentiat(?:e|ion)|integrat(?:e|ion)|'
    r'construct(?:ion)?|inequalit(?:y|ies)|equation|recurren(?:ce|t)|'
    r'count(?:ing)?|bound(?:ing)?|symmetr(?:y|ic)|generat(?:ing)?\s+function)\b', re.I)
INTENT = re.compile(r'\b(?:we\s+(?:can|could|will|should)|let\s+us|let\'s|I\s+(?:can|will)|try|use|apply)\b', re.I)
OPERATION = re.compile(
    r'\b(?:solve|compute|derive|check|compare|factor|substitute|bound|construct|'
    r'count|enumerate|differentiate|integrate|show|prove|rewrite)\b', re.I)
FAILED = re.compile(
    r'\b(?:does\s+not\s+satisfy|doesn\'t\s+satisfy|fails?\s+(?:the\s+)?check|'
    r'contradict(?:s|ion)|invalid|incorrect|not\s+equal|wrong\s+because|'
    r'violat(?:es|ion))\b|≠', re.I)
EVIDENCE = re.compile(r'(?:\d\s*[=<>≠]\s*\d|[A-Za-z]\s*[=<>≠]\s*[-+]?\d|\b(?:condition|constraint|equation)\b)', re.I)
SENTENCE_END = re.compile(r'[.!?](?=\s|$)|\n+')


@dataclass(frozen=True)
class TransitionCandidate:
    transition: str
    evidence_end: int
    text: str
    basis: str


def completed_sentences(reasoning: str):
    """Yield terminated text spans; never use an unfinished tail as evidence."""
    left = 0
    for match in SENTENCE_END.finditer(reasoning):
        end = match.end()
        sentence = reasoning[left:end].strip()
        if sentence:
            yield left, end, sentence
        left = end


def classify_sentence(sentence: str):
    """Return one explicit pattern proposal or abstain; no correctness inference."""
    if FAILED.search(sentence) and EVIDENCE.search(sentence):
        return 'failed_check_to_revise'
    if METHOD.search(sentence) and INTENT.search(sentence) and OPERATION.search(sentence):
        return 'approach_to_commit'
    return None


class StreamingTransitionDetector:
    """Append-only candidate stream with chunk-invariant emitted event identities."""

    def __init__(self):
        self.adapter = StreamingAdapter()
        self.last_sentence_end = 0
        self.last_candidate_end = -1

    def observe_tokens(self, problem: str, token_ids, decode):
        ids = tuple(token_ids)
        return self.observe({'problem': problem, 'emitted_token_ids': ids,
                             'emitted_text': decode(ids)})

    def observe(self, record: Mapping) -> dict:
        prefix = PrefixInput.from_record(record)
        parsed = self.adapter.observe({'problem': prefix.problem,
                                       'emitted_token_ids': prefix.emitted_token_ids,
                                       'emitted_text': prefix.emitted_text})
        if parsed['closure']:
            return {'version': VERSION, 'closure': True, 'events': [],
                    'status': 'UNQUALIFIED_CANDIDATES_ONLY'}
        reasoning = prefix.emitted_text
        events = []
        for candidate in candidates(reasoning):
            if candidate.end <= self.last_candidate_end:
                continue
            events.append(TransitionCandidate('candidate_to_verify', candidate.end,
                                              candidate.expression, 'complete_supported_numeric_candidate'))
            self.last_candidate_end = candidate.end
        for _, end, sentence in completed_sentences(reasoning):
            if end <= self.last_sentence_end:
                continue
            transition = classify_sentence(sentence)
            if transition is not None:
                events.append(TransitionCandidate(transition, end, sentence,
                                                  'explicit_pattern_pending_semantic_audit'))
            self.last_sentence_end = end
        events.sort(key=lambda item: (item.evidence_end, item.transition))
        return {'version': VERSION, 'closure': False, 'events': events,
                'status': 'UNQUALIFIED_CANDIDATES_ONLY'}

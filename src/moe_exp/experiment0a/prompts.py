from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence

from .data import EpisodeDocument, SENTENCE_LABELS


SEED_INSTRUCTIONS = r"""You are an expert reviewer of mathematical reasoning traces. Classify the single
input sentence according to its primary local function in the adapted Schoenfeld Episode Theory.

Choose exactly one label:
- Read: restates only information or the goal given by the problem, without inference.
- Analyze: recalls concepts, introduces notation, or makes a certain deduction, without executing a
  pre-announced calculation. A small analytic calculation may be Analyze when it establishes a relation.
- Plan: commits to a concrete next mathematical action before executing it.
- Implement: carries out a chosen procedure, substitution, calculation, enumeration, or its direct result.
- Explore: tentatively proposes an option, hypothesis, guess, or trial without commitment.
- Verify: evaluates or confirms correctness, consistency, reasonableness, or a candidate result.
- Monitor: a short content-light hesitation, pause, self-monitoring interjection, or transition.

Important distinctions:
- Classify the sentence's function, not isolated keywords.
- "Let's verify" is Verify, not Plan.
- A tentative substantive idea is Explore; a content-light "Wait" or "Let me think" is Monitor.
- A declarative result without an actual check is not Verify.

Return only the label name, with no Markdown, explanation, or additional text."""


@dataclass(frozen=True)
class FewShotSentence:
    question_id: str
    unit_id: int
    sentence: str
    label: str

    def metadata(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "unit_id": self.unit_id,
            "label": self.label,
        }


def select_few_shot_sentences(
    documents: Sequence[EpisodeDocument],
    *,
    count: int = 7,
) -> tuple[FewShotSentence, ...]:
    """Select deterministic train-only examples, cycling through all seven classes."""
    if count <= 0:
        raise ValueError("Few-shot example count must be positive")

    by_label: dict[str, list[FewShotSentence]] = {label: [] for label in SENTENCE_LABELS}
    for document in documents:
        for unit_id, unit in enumerate(document.units):
            by_label[unit.sentence_label].append(
                FewShotSentence(document.question_id, unit_id, unit.text, unit.sentence_label)
            )

    missing = [label for label, candidates in by_label.items() if not candidates]
    if missing:
        raise ValueError(f"Training split has no examples for labels: {', '.join(missing)}")

    # Put representative-length examples first; prefer different source responses.
    for candidates in by_label.values():
        target_length = median(len(candidate.sentence) for candidate in candidates)
        candidates.sort(
            key=lambda item: (
                abs(len(item.sentence) - target_length),
                item.question_id,
                item.unit_id,
            )
        )

    selected: list[FewShotSentence] = []
    selected_keys: set[tuple[str, int]] = set()
    used_documents: set[str] = set()
    label_offsets = {label: 0 for label in SENTENCE_LABELS}

    while len(selected) < count:
        label = SENTENCE_LABELS[len(selected) % len(SENTENCE_LABELS)]
        candidates = by_label[label]
        offset = label_offsets[label]
        available = [
            item
            for item in candidates[offset:]
            if (item.question_id, item.unit_id) not in selected_keys
        ]
        if not available:
            break
        choice = next(
            (item for item in available if item.question_id not in used_documents),
            available[0],
        )
        label_offsets[label] = candidates.index(choice) + 1
        selected.append(choice)
        selected_keys.add((choice.question_id, choice.unit_id))
        used_documents.add(choice.question_id)

    return tuple(selected)


def build_few_shot_instructions(examples: Sequence[FewShotSentence]) -> str:
    if not examples:
        raise ValueError("At least one few-shot example is required")

    sections = [SEED_INSTRUCTIONS, "\n\nGold training examples:"]
    for index, example in enumerate(examples, start=1):
        sections.append(
            f"\n\nExample {index}\nSentence: {example.sentence}\nCorrect label: {example.label}"
        )
    sections.append("\n\nNow classify the new sentence. Return only its label.")
    return "".join(sections)

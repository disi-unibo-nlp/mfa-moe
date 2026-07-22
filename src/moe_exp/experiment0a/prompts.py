from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from .data import EpisodeDocument, EpisodeUnit, PARAGRAPH_LABELS


SEED_INSTRUCTIONS = r"""You are an expert reviewer of mathematical reasoning traces. Apply the adapted
Schoenfeld Episode Theory annotation protocol. The response is a JSON array of already segmented units.
Use the whole response and the original problem as context, but assign exactly one paragraph-level label
and one sentence-level label to every unit.

Paragraph-level labels describe the broader episode containing the unit:
- General: the initial/main solution path, including reading, analysis, planning, calculation, and a final
  answer when these are not part of a broader exploration or verification routine.
- Explore: a broader uncertain, trial-and-error investigation, alternative route, conjecture, or tangent.
- Verify: a broader retrospective routine whose purpose is checking a candidate result. Once such a
  routine starts, its calculations and conclusions remain Verify at paragraph level.

Sentence-level labels describe the unit's primary local function:
- Read: restates only information or the goal given by the problem, without inference.
- Analyze: recalls concepts, introduces notation, or makes a certain deduction, without executing a
  pre-announced calculation. A small analytic calculation may be Analyze when it establishes a relation.
- Plan: commits to a concrete next mathematical action before executing it.
- Implement: carries out a chosen procedure, substitution, calculation, enumeration, or its direct result.
- Explore: tentatively proposes an option, hypothesis, guess, or trial without commitment.
- Verify: evaluates or confirms correctness, consistency, reasonableness, or a candidate result.
- Monitor: a short content-light hesitation, pause, self-monitoring interjection, or transition.

Important distinctions:
- Paragraph and sentence labels are independent: a Verify paragraph can contain Plan or Implement units.
- Do not classify from keywords alone; use purpose and neighboring units.
- "Let's verify" is Verify, not Plan. A declarative final answer without an actual check is not Verify.
- A tentative substantive idea is Explore; a content-light "Wait" or "Let me think" is Monitor.
- Return one object for every input id, in the same order, with no missing or duplicate ids.

The annotations output must be only a valid JSON array with this exact shape:
[{"id": 0, "paragraph_label": "General", "sentence_label": "Read"}]
Do not add Markdown, explanations, confidence scores, or any other keys."""


@dataclass(frozen=True)
class FewShotWindow:
    """A short, contiguous gold excerpt used as an in-context example."""

    question_id: str
    problem: str
    start: int
    units: tuple[EpisodeUnit, ...]

    @property
    def paragraph_label(self) -> str:
        return self.units[0].paragraph_label

    def response_json(self) -> str:
        return json.dumps(
            [{"id": index, "text": unit.text} for index, unit in enumerate(self.units)],
            ensure_ascii=False,
        )

    def annotations_json(self) -> str:
        return json.dumps(
            [
                {
                    "id": index,
                    "paragraph_label": unit.paragraph_label,
                    "sentence_label": unit.sentence_label,
                }
                for index, unit in enumerate(self.units)
            ],
            ensure_ascii=False,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "source_unit_start": self.start,
            "source_unit_stop": self.start + len(self.units),
            "paragraph_label": self.paragraph_label,
            "sentence_labels": sorted({unit.sentence_label for unit in self.units}),
        }


def _candidate_windows(
    documents: Sequence[EpisodeDocument],
    max_units: int,
) -> list[tuple[int, FewShotWindow]]:
    candidates: list[tuple[int, FewShotWindow]] = []
    for document_index, document in enumerate(documents):
        run_start = 0
        while run_start < len(document.units):
            paragraph_label = document.units[run_start].paragraph_label
            run_stop = run_start + 1
            while (
                run_stop < len(document.units)
                and document.units[run_stop].paragraph_label == paragraph_label
            ):
                run_stop += 1

            window_size = min(max_units, run_stop - run_start)
            for start in range(run_start, run_stop - window_size + 1):
                candidates.append(
                    (
                        document_index,
                        FewShotWindow(
                            question_id=document.question_id,
                            problem=document.problem,
                            start=start,
                            units=document.units[start : start + window_size],
                        ),
                    )
                )
            run_start = run_stop
    return candidates


def select_few_shot_windows(
    documents: Sequence[EpisodeDocument],
    *,
    count: int = 3,
    max_units: int = 8,
) -> tuple[FewShotWindow, ...]:
    """Select deterministic, label-diverse demonstrations from training documents only."""
    if count <= 0:
        raise ValueError("Few-shot example count must be positive")
    if max_units <= 0:
        raise ValueError("Few-shot unit count must be positive")

    candidates = _candidate_windows(documents, max_units)
    if not candidates:
        raise ValueError("Cannot build few-shot examples from an empty training split")

    selected: list[FewShotWindow] = []
    used_keys: set[tuple[str, int]] = set()
    covered_sentence_labels: set[str] = set()

    def choose(paragraph_label: str | None = None) -> FewShotWindow | None:
        eligible = [
            (document_index, window)
            for document_index, window in candidates
            if (window.question_id, window.start) not in used_keys
            and (paragraph_label is None or window.paragraph_label == paragraph_label)
        ]
        if not eligible:
            return None

        used_documents = {window.question_id for window in selected}

        def rank(item: tuple[int, FewShotWindow]) -> tuple[int, int, int, int, int]:
            document_index, window = item
            labels = {unit.sentence_label for unit in window.units}
            return (
                len(labels - covered_sentence_labels),
                len(labels),
                int(window.question_id not in used_documents),
                -document_index,
                -window.start,
            )

        return max(eligible, key=rank)[1]

    # The first three demonstrations mirror the three paragraph-level classes.
    for paragraph_label in PARAGRAPH_LABELS:
        if len(selected) >= count:
            break
        window = choose(paragraph_label)
        if window is None:
            continue
        selected.append(window)
        used_keys.add((window.question_id, window.start))
        covered_sentence_labels.update(unit.sentence_label for unit in window.units)

    while len(selected) < count:
        window = choose()
        if window is None:
            break
        selected.append(window)
        used_keys.add((window.question_id, window.start))
        covered_sentence_labels.update(unit.sentence_label for unit in window.units)

    return tuple(selected)


def build_few_shot_instructions(windows: Sequence[FewShotWindow]) -> str:
    """Append gold demonstrations to the guidebook-based seed instructions."""
    if not windows:
        raise ValueError("At least one few-shot window is required")

    sections = [
        SEED_INSTRUCTIONS,
        (
            "\n\nGold worked examples follow. Each response is a short contiguous excerpt from a "
            "training response; ids are local to that example. Apply the demonstrated functional "
            "distinctions, but classify every new unit from its own content and context."
        ),
    ]
    for index, window in enumerate(windows, start=1):
        sections.append(
            f"\n\nWorked example {index}\n"
            f"Problem:\n{window.problem}\n\n"
            f"Response:\n{window.response_json()}\n\n"
            f"Correct annotations:\n{window.annotations_json()}"
        )
    sections.append("\n\nNow annotate the new problem and response using the same JSON contract.")
    return "".join(sections)

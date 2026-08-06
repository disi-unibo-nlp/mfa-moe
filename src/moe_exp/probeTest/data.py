"""Load and align the gold Schoenfeld sentence annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPISODE_LABELS = (
    "Read",
    "Analyze",
    "Plan",
    "Implement",
    "Explore",
    "Verify",
    "Monitor",
)


@dataclass(frozen=True)
class GoldUnit:
    """One exactly aligned sentence-level gold annotation."""

    unit_index: int
    text: str
    label: str
    paragraph_label: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class GoldResponse:
    """A SAT prompt, its original DeepSeek-R1 response, and gold units."""

    response_id: str
    instruction: str
    response_text: str
    units: tuple[GoldUnit, ...]


def _numbered_json_files(directory: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, int | str]:
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.name)

    return sorted(directory.glob("*.json"), key=key)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _align_units(
    response_id: str,
    response_text: str,
    rows: list[dict[str, Any]],
    *,
    include_think_boundary_units: bool,
) -> tuple[GoldUnit, ...]:
    """Locate every annotated sentence verbatim and monotonically in the source response."""
    cursor = 0
    units: list[GoldUnit] = []
    for unit_index, row in enumerate(rows):
        text = row.get("text")
        label = row.get("gt-class-2")
        paragraph_label = row.get("gt-class-1")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{response_id}: unit {unit_index} has empty/non-string text")
        if label not in EPISODE_LABELS:
            raise ValueError(f"{response_id}: unit {unit_index} has unknown label {label!r}")
        if not isinstance(paragraph_label, str) or not paragraph_label:
            raise ValueError(f"{response_id}: unit {unit_index} has no paragraph label")

        char_start = response_text.find(text, cursor)
        if char_start < 0:
            excerpt = text[:100].replace("\n", "\\n")
            raise ValueError(
                f"{response_id}: unit {unit_index} cannot be aligned after character "
                f"{cursor}: {excerpt!r}"
            )
        char_end = char_start + len(text)
        # The current release contains exactly one synthetic closing-thinking
        # unit per response (38 total), all labeled Monitor.  The paper's 3,087
        # sentence count excludes them; retaining them would introduce a trivial
        # surface-marker shortcut for the Monitor probe.
        if include_think_boundary_units or "</think>" not in text:
            units.append(
                GoldUnit(
                    unit_index=unit_index,
                    text=text,
                    label=label,
                    paragraph_label=paragraph_label,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        cursor = char_end
    return tuple(units)


def load_gold_responses(
    dataset_dir: Path,
    *,
    include_think_boundary_units: bool = False,
) -> list[GoldResponse]:
    """Load the 38 released responses and validate exact sentence alignment.

    The release stores original traces and labels separately.  Exact alignment
    is mandatory: silently joining segmented sentences would alter whitespace,
    tokenization, and therefore the boundary activations.
    """
    originals_path = dataset_dir / "responses_original" / "SAT_deepseekR1_results.json"
    labels_dir = dataset_dir / "responses_labeled"
    if not originals_path.is_file():
        raise FileNotFoundError(f"Missing original responses: {originals_path}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labeled responses directory: {labels_dir}")

    originals = _load_json(originals_path)
    if not isinstance(originals, list):
        raise ValueError(f"Expected a JSON list in {originals_path}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in originals:
        response_id = row.get("Question ID")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError(f"Original response has invalid Question ID: {response_id!r}")
        if response_id in by_id:
            raise ValueError(f"Duplicate original response ID: {response_id}")
        by_id[response_id] = row

    label_files = _numbered_json_files(labels_dir)
    if not label_files:
        raise ValueError(f"No annotation JSON files found in {labels_dir}")

    responses: list[GoldResponse] = []
    seen_ids: set[str] = set()
    for path in label_files:
        labeled = _load_json(path)
        response_id = labeled.get("Question ID")
        if response_id in seen_ids:
            raise ValueError(f"Duplicate labeled response ID: {response_id}")
        if response_id not in by_id:
            raise ValueError(f"{path}: no original response for Question ID {response_id!r}")
        seen_ids.add(response_id)

        original = by_id[response_id]
        instruction = original.get("Instruction")
        response_text = original.get("deepseek-reasoner (response)")
        rows = labeled.get("data")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(f"{response_id}: missing Instruction")
        if not isinstance(response_text, str) or not response_text:
            raise ValueError(f"{response_id}: missing DeepSeek-R1 response")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{response_id}: missing sentence annotations")

        responses.append(
            GoldResponse(
                response_id=response_id,
                instruction=instruction,
                response_text=response_text,
                units=_align_units(
                    response_id,
                    response_text,
                    rows,
                    include_think_boundary_units=include_think_boundary_units,
                ),
            )
        )

    missing_labels = set(by_id) - seen_ids
    if missing_labels:
        missing = ", ".join(sorted(missing_labels))
        raise ValueError(f"Original responses without label files: {missing}")
    return responses


def label_counts(responses: list[GoldResponse]) -> dict[str, int]:
    counts = {label: 0 for label in EPISODE_LABELS}
    for response in responses:
        for unit in response.units:
            counts[unit.label] += 1
    return counts

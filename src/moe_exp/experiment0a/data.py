from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SENTENCE_LABELS = (
    "Read",
    "Analyze",
    "Plan",
    "Implement",
    "Explore",
    "Verify",
    "Monitor",
)


@dataclass(frozen=True)
class EpisodeUnit:
    text: str
    sentence_label: str
    parent_label: str = ""


@dataclass(frozen=True)
class EpisodeDocument:
    question_id: str
    units: tuple[EpisodeUnit, ...]
    problem_statement: str = ""


def _resolve_labels_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.resolve()
    if dataset_dir.name == "responses_labeled":
        labels_dir = dataset_dir
    else:
        labels_dir = dataset_dir / "responses_labeled"
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected responses_labeled under {dataset_dir}, or pass that directory directly."
        )
    return labels_dir


def _dataset_root(labels_dir: Path) -> Path:
    return labels_dir.parent


def _nonempty_text(value: Any) -> str:
    return str(value or "").strip()


def _format_problem_statement(row: dict[str, Any]) -> str:
    """Build model-visible problem context without leaking answers or rationales."""
    sections: list[str] = []
    stem = _nonempty_text(row.get("Item Stem"))
    question = _nonempty_text(row.get("Question"))
    if stem:
        sections.append(stem)
    if question:
        sections.append(question)

    choices = [
        f"{letter}. {value}"
        for letter in ("A", "B", "C", "D")
        if (value := _nonempty_text(row.get(f"Choice {letter}")))
    ]
    if choices:
        sections.append("Choices:\n" + "\n".join(choices))

    table = _nonempty_text(row.get("Table"))
    if table:
        sections.append("Table:\n" + table)
    figure = _nonempty_text(row.get("Figure"))
    if figure:
        sections.append("Figure description:\n" + figure)
    return "\n\n".join(sections)


def _load_problem_statements(dataset_root: Path) -> dict[str, str]:
    sat_path = dataset_root / "SAT.json"
    if not sat_path.is_file():
        return {}
    payload = json.loads(sat_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON list in {sat_path}")
    problems: dict[str, str] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise TypeError(f"Malformed SAT row {index} in {sat_path}")
        question_id = _nonempty_text(row.get("Question ID"))
        if question_id:
            problems[question_id] = _format_problem_statement(row)
    return problems


def _natural_json_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 2**31 - 1, path.name


def load_documents(dataset_dir: Path) -> list[EpisodeDocument]:
    """Load sentence-level gold labels, retaining response membership for splitting."""
    labels_dir = _resolve_labels_dir(dataset_dir)
    problems = _load_problem_statements(_dataset_root(labels_dir))
    documents: list[EpisodeDocument] = []

    for path in sorted(labels_dir.glob("*.json"), key=_natural_json_key):
        payload = json.loads(path.read_text(encoding="utf-8"))
        question_id = str(payload.get("Question ID", "")).strip()
        raw_units = payload.get("data")
        if not question_id or not isinstance(raw_units, list) or not raw_units:
            raise ValueError(f"Malformed annotation document: {path}")

        units: list[EpisodeUnit] = []
        for index, row in enumerate(raw_units):
            text = str(row.get("text", "")).strip()
            sentence_label = str(row.get("gt-class-2", "")).strip()
            if not text:
                raise ValueError(f"Empty text in {path}, unit {index}")
            if sentence_label not in SENTENCE_LABELS:
                raise ValueError(
                    f"Unknown sentence label {sentence_label!r} in {path}, unit {index}"
                )
            units.append(
                EpisodeUnit(
                    text=text,
                    sentence_label=sentence_label,
                    parent_label=_nonempty_text(row.get("gt-class-1")),
                )
            )

        documents.append(
            EpisodeDocument(
                question_id=question_id,
                units=tuple(units),
                problem_statement=problems.get(question_id, ""),
            )
        )

    if not documents:
        raise RuntimeError(f"No annotation JSON files found in {labels_dir}")
    return documents


def split_documents(
    documents: Iterable[EpisodeDocument],
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[list[EpisodeDocument], list[EpisodeDocument], list[EpisodeDocument]]:
    """Shuffle deterministically and split only at response/document boundaries."""
    shuffled = list(documents)
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_size and val_size must both be positive")
    if train_size + val_size >= len(shuffled):
        raise ValueError(
            "train_size + val_size must leave at least one held-out test document "
            f"(got {train_size} + {val_size} for {len(shuffled)} documents)"
        )
    random.Random(seed).shuffle(shuffled)
    train = shuffled[:train_size]
    val = shuffled[train_size : train_size + val_size]
    test = shuffled[train_size + val_size :]
    return train, val, test


def documents_to_dspy_examples(documents: Iterable[EpisodeDocument], dspy: Any) -> list[Any]:
    """Flatten documents with local context and split-local class weights."""
    documents = list(documents)
    label_counts = Counter(unit.sentence_label for document in documents for unit in document.units)
    total = sum(label_counts.values())
    present_classes = len(label_counts)
    class_weights = {
        label: total / (present_classes * count) for label, count in label_counts.items()
    }

    examples: list[Any] = []
    for document in documents:
        for unit_id, unit in enumerate(document.units):
            previous_sentence = (
                document.units[unit_id - 1].text if unit_id else "<START OF RESPONSE>"
            )
            next_sentence = (
                document.units[unit_id + 1].text
                if unit_id + 1 < len(document.units)
                else "<END OF RESPONSE>"
            )
            examples.append(
                dspy.Example(
                    question_id=document.question_id,
                    unit_id=unit_id,
                    problem_statement=document.problem_statement or "<PROBLEM UNAVAILABLE>",
                    previous_sentence=previous_sentence,
                    sentence=unit.text,
                    next_sentence=next_sentence,
                    gold_label=unit.sentence_label,
                    class_weight=class_weights[unit.sentence_label],
                ).with_inputs(
                    "problem_statement",
                    "previous_sentence",
                    "sentence",
                    "next_sentence",
                )
            )
    return examples


def make_nested_group_folds(
    documents: Iterable[EpisodeDocument],
    *,
    folds: int,
    locked_test_documents: int,
    inner_val_documents: int,
    seed: int,
) -> tuple[
    list[tuple[list[EpisodeDocument], list[EpisodeDocument], list[EpisodeDocument]]],
    list[EpisodeDocument],
]:
    """Create outer-test/inner-validation folds while preserving whole responses."""
    shuffled = list(documents)
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if locked_test_documents <= 0 or locked_test_documents >= len(shuffled):
        raise ValueError("locked_test_documents must leave at least two development documents")
    random.Random(seed).shuffle(shuffled)
    development = shuffled[:-locked_test_documents]
    locked_test = shuffled[-locked_test_documents:]
    if folds > len(development):
        raise ValueError("folds cannot exceed the number of development documents")

    outer_groups = [development[index::folds] for index in range(folds)]
    result = []
    for fold_index, outer_test in enumerate(outer_groups):
        outer_ids = {document.question_id for document in outer_test}
        remaining = [document for document in development if document.question_id not in outer_ids]
        if inner_val_documents <= 0 or inner_val_documents >= len(remaining):
            raise ValueError(
                "inner_val_documents must leave at least one training document in every fold"
            )
        random.Random(seed + fold_index + 1).shuffle(remaining)
        inner_val = remaining[:inner_val_documents]
        train = remaining[inner_val_documents:]
        result.append((train, inner_val, outer_test))
    return result, locked_test


_STRUCTURAL_MARKER = re.compile(
    r"(?i)(?:\*\*\s*final answer\s*\*\*|</?think>|^\s*(?:solution|answer)\s*:)"
)


def annotation_audit(documents: Iterable[EpisodeDocument]) -> dict[str, Any]:
    """Flag annotation units that merit human review without changing gold labels."""
    documents = list(documents)
    flagged: list[dict[str, Any]] = []
    missing_problem_ids = [
        document.question_id for document in documents if not document.problem_statement
    ]
    for document in documents:
        for unit_id, unit in enumerate(document.units):
            flags: list[str] = []
            if "\n" in unit.text:
                flags.append("multiline_unit")
            if _STRUCTURAL_MARKER.search(unit.text):
                flags.append("structural_or_control_marker")
            sentence_endings = len(re.findall(r"[.!?](?:\s|$)", unit.text))
            if sentence_endings > 1:
                flags.append("multiple_sentence_endings")
            if "structural_or_control_marker" in flags and re.search(
                r"\\boxed|[=+\-*/]|\d", unit.text
            ):
                flags.append("mixed_structural_and_substantive_content")
            if flags:
                flagged.append(
                    {
                        "question_id": document.question_id,
                        "unit_id": unit_id,
                        "label": unit.sentence_label,
                        "parent_label": unit.parent_label,
                        "flags": flags,
                        "text": unit.text,
                    }
                )

    flag_counts = Counter(flag for row in flagged for flag in row["flags"])
    return {
        "documents": len(documents),
        "units": sum(len(document.units) for document in documents),
        "documents_missing_problem_context": len(missing_problem_ids),
        "missing_problem_question_ids": missing_problem_ids,
        "flagged_units": len(flagged),
        "flag_counts": dict(sorted(flag_counts.items())),
        "items": flagged,
    }

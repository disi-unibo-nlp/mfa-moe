from __future__ import annotations

import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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


@dataclass(frozen=True)
class EpisodeDocument:
    question_id: str
    units: tuple[EpisodeUnit, ...]


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


def _natural_json_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 2**31 - 1, path.name


def load_documents(dataset_dir: Path) -> list[EpisodeDocument]:
    """Load sentence-level gold labels, retaining response membership for splitting."""
    labels_dir = _resolve_labels_dir(dataset_dir)
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
            units.append(EpisodeUnit(text, sentence_label))

        documents.append(EpisodeDocument(question_id, tuple(units)))

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
    """Flatten split documents into sentence examples without crossing split boundaries."""
    examples = []
    for document in documents:
        for unit_id, unit in enumerate(document.units):
            examples.append(
                dspy.Example(
                    question_id=document.question_id,
                    unit_id=unit_id,
                    sentence=unit.text,
                    gold_label=unit.sentence_label,
                ).with_inputs("sentence")
            )
    return examples

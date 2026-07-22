from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PARAGRAPH_LABELS = ("General", "Explore", "Verify")
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
    paragraph_label: str
    sentence_label: str


@dataclass(frozen=True)
class EpisodeDocument:
    question_id: str
    problem: str
    units: tuple[EpisodeUnit, ...]

    def response_json(self) -> str:
        return json.dumps(
            [{"id": index, "text": unit.text} for index, unit in enumerate(self.units)],
            ensure_ascii=False,
        )

    def gold_annotations_json(self) -> str:
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


def _resolve_dataset_paths(dataset_dir: Path) -> tuple[Path, Path | None]:
    dataset_dir = dataset_dir.resolve()
    if dataset_dir.name == "responses_labeled":
        labels_dir = dataset_dir
        sat_path = dataset_dir.parent / "SAT.json"
    else:
        labels_dir = dataset_dir / "responses_labeled"
        sat_path = dataset_dir / "SAT.json"
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected responses_labeled under {dataset_dir}, or pass that directory directly."
        )
    return labels_dir, sat_path if sat_path.is_file() else None


def _load_problems(sat_path: Path | None) -> dict[str, str]:
    if sat_path is None:
        return {}
    rows = json.loads(sat_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{sat_path} must contain a JSON array")

    problems: dict[str, str] = {}
    fields = (
        "Item Stem",
        "Question",
        "Choice A",
        "Choice B",
        "Choice C",
        "Choice D",
        "Table",
        "Figure",
    )
    for row in rows:
        question_id = str(row.get("Question ID", "")).strip()
        if not question_id:
            continue
        parts = [f"{field}: {row[field]}" for field in fields if str(row.get(field, "")).strip()]
        problems[question_id] = "\n".join(parts)
    return problems


def _natural_json_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 2**31 - 1, path.name


def load_documents(dataset_dir: Path) -> list[EpisodeDocument]:
    """Load and validate the paper's gold annotations, one document per response."""
    labels_dir, sat_path = _resolve_dataset_paths(dataset_dir)
    problems = _load_problems(sat_path)
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
            paragraph_label = str(row.get("gt-class-1", "")).strip()
            sentence_label = str(row.get("gt-class-2", "")).strip()
            if not text:
                raise ValueError(f"Empty text in {path}, unit {index}")
            if paragraph_label not in PARAGRAPH_LABELS:
                raise ValueError(
                    f"Unknown paragraph label {paragraph_label!r} in {path}, unit {index}"
                )
            if sentence_label not in SENTENCE_LABELS:
                raise ValueError(
                    f"Unknown sentence label {sentence_label!r} in {path}, unit {index}"
                )
            units.append(EpisodeUnit(text, paragraph_label, sentence_label))

        problem = problems.get(question_id, f"Question ID: {question_id}")
        documents.append(EpisodeDocument(question_id, problem, tuple(units)))

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
    """Convert documents lazily so data/metric tests do not require DSPy."""
    return [
        dspy.Example(
            question_id=document.question_id,
            problem=document.problem,
            response=document.response_json(),
            gold_annotations=document.gold_annotations_json(),
        ).with_inputs("problem", "response")
        for document in documents
    ]

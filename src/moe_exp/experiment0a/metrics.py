from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from scipy.stats import kendalltau

from .data import PARAGRAPH_LABELS, SENTENCE_LABELS


@dataclass(frozen=True)
class Annotation:
    unit_id: int
    paragraph_label: str
    sentence_label: str


@dataclass(frozen=True)
class Agreement:
    paragraph_kappa: float
    paragraph_kendall_tau: float
    sentence_kappa: float
    sentence_kendall_tau: float
    paragraph_accuracy: float
    sentence_accuracy: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class AnnotationParseError(ValueError):
    pass


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _canonical_label(value: Any, allowed: Sequence[str], field: str) -> str:
    normalized = str(value).strip().casefold()
    mapping = {label.casefold(): label for label in allowed}
    if normalized not in mapping:
        raise AnnotationParseError(
            f"Invalid {field} {value!r}; expected one of {', '.join(allowed)}"
        )
    return mapping[normalized]


def parse_annotations(value: Any, expected_count: int | None = None) -> list[Annotation]:
    """Parse the judge output and enforce a complete one-to-one unit mapping."""
    if hasattr(value, "annotations"):
        value = value.annotations
    if isinstance(value, str):
        try:
            value = json.loads(_strip_json_fence(value))
        except json.JSONDecodeError as exc:
            raise AnnotationParseError(f"annotations is not valid JSON: {exc}") from exc
    if isinstance(value, dict) and "annotations" in value:
        value = value["annotations"]
    if not isinstance(value, list):
        raise AnnotationParseError("annotations must be a JSON array")

    parsed: list[Annotation] = []
    seen: set[int] = set()
    for position, row in enumerate(value):
        if not isinstance(row, dict):
            raise AnnotationParseError(f"annotation at position {position} is not an object")
        try:
            unit_id = int(row["id"])
            paragraph = _canonical_label(
                row["paragraph_label"], PARAGRAPH_LABELS, "paragraph_label"
            )
            sentence = _canonical_label(row["sentence_label"], SENTENCE_LABELS, "sentence_label")
        except KeyError as exc:
            raise AnnotationParseError(f"annotation {position} is missing {exc.args[0]}") from exc
        if unit_id in seen:
            raise AnnotationParseError(f"duplicate unit id {unit_id}")
        seen.add(unit_id)
        parsed.append(Annotation(unit_id, paragraph, sentence))

    parsed.sort(key=lambda annotation: annotation.unit_id)
    if expected_count is not None:
        expected_ids = list(range(expected_count))
        actual_ids = [annotation.unit_id for annotation in parsed]
        if actual_ids != expected_ids:
            raise AnnotationParseError(
                f"expected exactly unit ids 0..{expected_count - 1}; got {actual_ids}"
            )
    return parsed


def _safe_kappa(gold: Sequence[str], predicted: Sequence[str]) -> float:
    if list(gold) == list(predicted):
        return 1.0
    count = len(gold)
    observed = sum(a == b for a, b in zip(gold, predicted)) / count
    gold_counts = Counter(gold)
    predicted_counts = Counter(predicted)
    expected = sum(
        gold_counts[label] * predicted_counts[label]
        for label in gold_counts.keys() | predicted_counts.keys()
    ) / (count * count)
    denominator = 1.0 - expected
    return 0.0 if denominator == 0.0 else (observed - expected) / denominator


def _safe_kendall(gold: Sequence[str], predicted: Sequence[str], order: Sequence[str]) -> float:
    if list(gold) == list(predicted):
        return 1.0
    rank = {label: index for index, label in enumerate(order)}
    value = float(
        kendalltau(
            [rank[label] for label in gold],
            [rank[label] for label in predicted],
            variant="b",
        ).statistic
    )
    return 0.0 if math.isnan(value) else value


def compute_agreement(
    gold: Sequence[Annotation],
    predicted: Sequence[Annotation],
    *,
    kappa_weight: float = 0.5,
    paragraph_weight: float = 0.5,
) -> Agreement:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and predicted must be non-empty and have equal length")
    if [item.unit_id for item in gold] != [item.unit_id for item in predicted]:
        raise ValueError("gold and predicted unit ids must align")
    if not 0.0 <= kappa_weight <= 1.0 or not 0.0 <= paragraph_weight <= 1.0:
        raise ValueError("metric weights must be in [0, 1]")

    gold_paragraph = [item.paragraph_label for item in gold]
    pred_paragraph = [item.paragraph_label for item in predicted]
    gold_sentence = [item.sentence_label for item in gold]
    pred_sentence = [item.sentence_label for item in predicted]

    paragraph_kappa = _safe_kappa(gold_paragraph, pred_paragraph)
    paragraph_tau = _safe_kendall(gold_paragraph, pred_paragraph, PARAGRAPH_LABELS)
    sentence_kappa = _safe_kappa(gold_sentence, pred_sentence)
    sentence_tau = _safe_kendall(gold_sentence, pred_sentence, SENTENCE_LABELS)
    paragraph_accuracy = sum(a == b for a, b in zip(gold_paragraph, pred_paragraph)) / len(gold)
    sentence_accuracy = sum(a == b for a, b in zip(gold_sentence, pred_sentence)) / len(gold)

    # GEPA maximizes a [0, 1] reward. Negative agreement is therefore clipped,
    # while the raw coefficients remain available in the report.
    paragraph_score = kappa_weight * max(0.0, paragraph_kappa) + (1.0 - kappa_weight) * max(
        0.0, paragraph_tau
    )
    sentence_score = kappa_weight * max(0.0, sentence_kappa) + (1.0 - kappa_weight) * max(
        0.0, sentence_tau
    )
    score = paragraph_weight * paragraph_score + (1.0 - paragraph_weight) * sentence_score

    return Agreement(
        paragraph_kappa=paragraph_kappa,
        paragraph_kendall_tau=paragraph_tau,
        sentence_kappa=sentence_kappa,
        sentence_kendall_tau=sentence_tau,
        paragraph_accuracy=paragraph_accuracy,
        sentence_accuracy=sentence_accuracy,
        score=score,
    )


def concatenate_annotations(groups: Iterable[Sequence[Annotation]]) -> list[Annotation]:
    """Flatten documents and assign fresh ids for corpus-level agreement."""
    flattened: list[Annotation] = []
    for group in groups:
        for item in group:
            flattened.append(Annotation(len(flattened), item.paragraph_label, item.sentence_label))
    return flattened

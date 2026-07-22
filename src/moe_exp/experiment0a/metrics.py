from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from scipy.stats import kendalltau

from .data import SENTENCE_LABELS


class LabelParseError(ValueError):
    pass


@dataclass(frozen=True)
class SentenceAgreement:
    cohen_kappa: float
    kendall_tau_b: float
    accuracy: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_sentence_label(value: Any) -> str:
    """Parse exactly one sentence-level label from a DSPy prediction or string."""
    if hasattr(value, "label"):
        value = value.label
    if isinstance(value, dict):
        value = value.get("label")
    if not isinstance(value, str):
        raise LabelParseError("label must be a string")

    text = _strip_json_fence(value)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = text
    if isinstance(decoded, dict):
        decoded = decoded.get("label")
    if not isinstance(decoded, str):
        raise LabelParseError("label must be a string or a JSON object with a label key")

    normalized = re.sub(r"^label\s*:\s*", "", decoded.strip(), flags=re.IGNORECASE).casefold()
    mapping = {label.casefold(): label for label in SENTENCE_LABELS}
    if normalized not in mapping:
        raise LabelParseError(
            f"invalid label {decoded!r}; expected one of {', '.join(SENTENCE_LABELS)}"
        )
    return mapping[normalized]


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


def _safe_kendall(gold: Sequence[str], predicted: Sequence[str]) -> float:
    if list(gold) == list(predicted):
        return 1.0
    rank = {label: index for index, label in enumerate(SENTENCE_LABELS)}
    value = float(
        kendalltau(
            [rank[label] for label in gold],
            [rank[label] for label in predicted],
            variant="b",
        ).statistic
    )
    return 0.0 if math.isnan(value) else value


def compute_sentence_agreement(
    gold: Sequence[str],
    predicted: Sequence[str],
    *,
    kappa_weight: float = 0.5,
) -> SentenceAgreement:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and predicted must be non-empty and have equal length")
    if not 0.0 <= kappa_weight <= 1.0:
        raise ValueError("kappa_weight must be in [0, 1]")
    unknown = (set(gold) | set(predicted)) - set(SENTENCE_LABELS)
    if unknown:
        raise ValueError(f"unknown sentence labels: {sorted(unknown)}")

    kappa = _safe_kappa(gold, predicted)
    kendall = _safe_kendall(gold, predicted)
    accuracy = sum(a == b for a, b in zip(gold, predicted)) / len(gold)

    # Preserve the ordering of negative coefficients while mapping the GEPA/report
    # composite to [0, 1]. Raw coefficients remain available in the result.
    scaled_kappa = (kappa + 1.0) / 2.0
    scaled_kendall = (kendall + 1.0) / 2.0
    score = kappa_weight * scaled_kappa + (1.0 - kappa_weight) * scaled_kendall
    return SentenceAgreement(kappa, kendall, accuracy, score)

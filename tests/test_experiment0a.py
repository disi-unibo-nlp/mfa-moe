import json
from pathlib import Path

import pytest

from moe_exp.experiment0a.data import load_documents, split_documents
from moe_exp.experiment0a.metrics import (
    Annotation,
    AnnotationParseError,
    compute_agreement,
    parse_annotations,
)


def test_parse_annotations_is_case_tolerant_but_structurally_strict() -> None:
    parsed = parse_annotations(
        '[{"id": 0, "paragraph_label": "general", "sentence_label": "READ"}]',
        expected_count=1,
    )
    assert parsed == [Annotation(0, "General", "Read")]

    with pytest.raises(AnnotationParseError, match="exactly unit ids"):
        parse_annotations(
            '[{"id": 1, "paragraph_label": "General", "sentence_label": "Read"}]',
            expected_count=1,
        )


def test_exact_agreement_scores_one_even_for_constant_sequences() -> None:
    annotations = [
        Annotation(0, "General", "Read"),
        Annotation(1, "General", "Read"),
    ]
    metrics = compute_agreement(annotations, annotations)
    assert metrics.paragraph_kappa == 1.0
    assert metrics.paragraph_kendall_tau == 1.0
    assert metrics.sentence_kappa == 1.0
    assert metrics.sentence_kendall_tau == 1.0
    assert metrics.score == 1.0


def test_load_and_split_documents_without_cross_document_leakage(tmp_path: Path) -> None:
    labels_dir = tmp_path / "responses_labeled"
    labels_dir.mkdir()
    sat_rows = []
    for index in range(4):
        question_id = f"q{index}"
        sat_rows.append(
            {"Question ID": question_id, "Item Stem": f"Stem {index}", "Question": "Solve."}
        )
        payload = {
            "Question ID": question_id,
            "data": [
                {"text": "We are given x.", "gt-class-1": "General", "gt-class-2": "Read"},
                {"text": "Thus x=1.", "gt-class-1": "General", "gt-class-2": "Implement"},
            ],
        }
        (labels_dir / f"{index + 1}.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "SAT.json").write_text(json.dumps(sat_rows), encoding="utf-8")

    documents = load_documents(tmp_path)
    train, val, test = split_documents(documents, train_size=2, val_size=1, seed=7)
    split_ids = [{document.question_id for document in group} for group in (train, val, test)]

    assert len(documents) == 4
    assert "Item Stem: Stem" in documents[0].problem
    assert not (split_ids[0] & split_ids[1])
    assert not (split_ids[0] & split_ids[2])
    assert not (split_ids[1] & split_ids[2])

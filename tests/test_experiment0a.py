import json
from pathlib import Path

import pytest

from moe_exp.experiment0a.data import (
    SENTENCE_LABELS,
    EpisodeDocument,
    EpisodeUnit,
    documents_to_dspy_examples,
    load_documents,
    split_documents,
)
from moe_exp.experiment0a.metrics import (
    LabelParseError,
    compute_sentence_agreement,
    parse_sentence_label,
)
from moe_exp.experiment0a.prompts import (
    SEED_INSTRUCTIONS,
    build_few_shot_instructions,
    select_few_shot_sentences,
)


def test_parse_sentence_label_is_case_tolerant_but_strict() -> None:
    assert parse_sentence_label("read") == "Read"
    assert parse_sentence_label('{"label": "VERIFY"}') == "Verify"
    assert parse_sentence_label("Label: Monitor") == "Monitor"

    with pytest.raises(LabelParseError, match="invalid label"):
        parse_sentence_label("The answer is Read because it repeats the problem.")


def test_exact_sentence_agreement_scores_one_for_constant_sequences() -> None:
    agreement = compute_sentence_agreement(["Read", "Read"], ["Read", "Read"])
    assert agreement.cohen_kappa == 1.0
    assert agreement.kendall_tau_b == 1.0
    assert agreement.accuracy == 1.0
    assert agreement.score == 1.0


def test_agreement_rescales_instead_of_clipping_negative_coefficients() -> None:
    agreement = compute_sentence_agreement(
        ["Read", "Analyze", "Plan", "Implement"],
        ["Implement", "Plan", "Analyze", "Read"],
    )
    assert agreement.kendall_tau_b < 0.0
    assert 0.0 <= agreement.score < 0.5


def test_load_split_and_flatten_without_cross_document_leakage(tmp_path: Path) -> None:
    labels_dir = tmp_path / "responses_labeled"
    labels_dir.mkdir()
    for index in range(4):
        question_id = f"q{index}"
        payload = {
            "Question ID": question_id,
            "data": [
                {"text": "We are given x.", "gt-class-1": "General", "gt-class-2": "Read"},
                {
                    "text": "Thus x=1.",
                    "gt-class-1": "General",
                    "gt-class-2": "Implement",
                },
            ],
        }
        (labels_dir / f"{index + 1}.json").write_text(json.dumps(payload), encoding="utf-8")
    documents = load_documents(tmp_path)
    train, val, test = split_documents(documents, train_size=2, val_size=1, seed=7)
    split_ids = [{document.question_id for document in group} for group in (train, val, test)]

    class FakeExample:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)
            self.input_fields: tuple[str, ...] = ()

        def with_inputs(self, *fields: str) -> "FakeExample":
            self.input_fields = fields
            return self

    class FakeDSPy:
        Example = FakeExample

    examples = documents_to_dspy_examples(train, FakeDSPy)
    assert len(examples) == 4
    assert all(example.input_fields == ("sentence",) for example in examples)
    assert not (split_ids[0] & split_ids[1])
    assert not (split_ids[0] & split_ids[2])
    assert not (split_ids[1] & split_ids[2])


def test_few_shot_prompt_covers_all_sentence_labels() -> None:
    documents = [
        EpisodeDocument(
            question_id=f"q{index}",
            units=(
                EpisodeUnit(
                    text=f"Representative {label} sentence.",
                    sentence_label=label,
                ),
            ),
        )
        for index, label in enumerate(SENTENCE_LABELS)
    ]

    examples = select_few_shot_sentences(documents, count=7)
    prompt = build_few_shot_instructions(examples)

    assert {example.label for example in examples} == set(SENTENCE_LABELS)
    assert prompt.startswith(SEED_INSTRUCTIONS)
    assert prompt.count("Correct label:") == 7

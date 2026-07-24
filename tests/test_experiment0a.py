import json
from pathlib import Path

import pytest

from moe_exp.experiment0a.data import (
    SENTENCE_LABELS,
    EpisodeDocument,
    EpisodeUnit,
    annotation_audit,
    documents_to_dspy_examples,
    load_documents,
    make_nested_group_folds,
    split_documents,
)
from moe_exp.experiment0a.metrics import (
    LabelParseError,
    compute_classification_metrics,
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
    (tmp_path / "SAT.json").write_text(
        json.dumps(
            [
                {
                    "Question ID": f"q{index}",
                    "Item Stem": f"Given value {index}.",
                    "Question": "What is x?",
                    "Choice A": "1",
                    "Choice B": "2",
                    "Correct Answer": "B",
                    "Rationale": "This must never be model-visible.",
                }
                for index in range(4)
            ]
        ),
        encoding="utf-8",
    )
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
    assert all(
        example.input_fields
        == ("problem_statement", "previous_sentence", "sentence", "next_sentence")
        for example in examples
    )
    assert all("What is x?" in example.problem_statement for example in examples)
    assert all("Rationale" not in example.problem_statement for example in examples)
    assert all("This must never" not in example.problem_statement for example in examples)
    assert examples[0].previous_sentence == "<START OF RESPONSE>"
    assert examples[0].next_sentence == "Thus x=1."
    assert examples[1].next_sentence == "<END OF RESPONSE>"
    for label in ("Read", "Implement"):
        label_examples = [example for example in examples if example.gold_label == label]
        assert sum(example.class_weight for example in label_examples) == pytest.approx(
            len(examples) / 2
        )
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

    examples = select_few_shot_sentences(documents, count=21)
    prompt = build_few_shot_instructions(examples)

    assert {example.label for example in examples} == set(SENTENCE_LABELS)
    assert prompt.startswith(SEED_INSTRUCTIONS)
    assert prompt.count("Correct label:") == 21
    assert all(
        sum(example.label == label for example in examples) == 3 for label in SENTENCE_LABELS
    )
    assert "Previous unit:" in prompt
    assert "Next unit:" in prompt


def test_strict_classification_metrics_include_invalid_predictions() -> None:
    report = compute_classification_metrics(
        ["Read", "Read", "Analyze", "Analyze"],
        ["Read", None, "Analyze", "Read"],
    )
    assert report["accuracy"] == 0.5
    assert report["balanced_accuracy"] == pytest.approx(0.5)
    assert report["per_class"]["Read"]["recall"] == 0.5
    assert report["per_class"]["Analyze"]["precision"] == 1.0


def test_nested_group_folds_never_expose_locked_test() -> None:
    documents = [
        EpisodeDocument(
            question_id=f"q{index}",
            units=(EpisodeUnit("Sentence.", "Read"),),
        )
        for index in range(12)
    ]
    folds, locked = make_nested_group_folds(
        documents,
        folds=3,
        locked_test_documents=3,
        inner_val_documents=2,
        seed=42,
    )
    locked_ids = {document.question_id for document in locked}
    outer_ids: list[str] = []
    for train, inner_val, outer_test in folds:
        groups = [
            {document.question_id for document in group} for group in (train, inner_val, outer_test)
        ]
        assert not (groups[0] & groups[1])
        assert not (groups[0] & groups[2])
        assert not (groups[1] & groups[2])
        assert not (locked_ids & set.union(*groups))
        outer_ids.extend(document.question_id for document in outer_test)
    assert len(outer_ids) == len(set(outer_ids)) == 9


def test_annotation_audit_flags_compound_structural_units() -> None:
    documents = [
        EpisodeDocument(
            question_id="q1",
            problem_statement="What is 2 + 3?",
            units=(
                EpisodeUnit(
                    "**Final Answer**\n\\boxed{B}</think>",
                    "Monitor",
                    "General",
                ),
            ),
        )
    ]
    report = annotation_audit(documents)
    assert report["documents_missing_problem_context"] == 0
    assert report["flagged_units"] == 1
    flags = report["items"][0]["flags"]
    assert "structural_or_control_marker" in flags
    assert "mixed_structural_and_substantive_content" in flags

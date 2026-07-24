from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dspy

from .data import (
    SENTENCE_LABELS,
    EpisodeDocument,
    annotation_audit,
    documents_to_dspy_examples,
    load_documents,
    make_nested_group_folds,
    split_documents,
)
from .metrics import (
    LabelParseError,
    compute_classification_metrics,
    compute_sentence_agreement,
    parse_sentence_label,
)
from .prompts import (
    SEED_INSTRUCTIONS,
    build_few_shot_instructions,
    select_few_shot_sentences,
)


class EpisodeJudge(dspy.Module):
    """Context-aware sentence-classification prompt optimized by GEPA."""

    def __init__(self, instructions: str = SEED_INSTRUCTIONS) -> None:
        super().__init__()
        signature = dspy.Signature(
            "problem_statement, previous_sentence, sentence, next_sentence -> label",
            instructions=instructions,
        )
        self.classify = dspy.Predict(signature)

    def forward(
        self,
        problem_statement: str,
        previous_sentence: str,
        sentence: str,
        next_sentence: str,
    ) -> Any:
        return self.classify(
            problem_statement=problem_statement,
            previous_sentence=previous_sentence,
            sentence=sentence,
            next_sentence=next_sentence,
        )


def create_metric(*, class_balanced: bool) -> Callable[..., dspy.Prediction]:
    """Return decomposable exact-match or class-balanced feedback for GEPA."""

    def metric(
        example: Any,
        pred: Any,
        trace: Any = None,
        pred_name: Any = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        del trace, pred_name, pred_trace
        gold = str(example.gold_label)
        try:
            predicted = parse_sentence_label(pred)
        except LabelParseError as exc:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"OUTPUT CONTRACT ERROR: {exc}. Return exactly one of the seven label names. "
                    f"The gold label for this training example is {gold}."
                ),
            )

        correct = predicted == gold
        feedback = (
            f"Correct classification: {gold}."
            if correct
            else (
                f"Misclassified as {predicted}; the gold label is {gold}. Improve the instruction "
                f"distinguishing {gold} from {predicted} for sentences with this function."
            )
        )
        reward = float(example.class_weight) if class_balanced else 1.0
        return dspy.Prediction(score=reward if correct else 0.0, feedback=feedback)

    return metric


def _make_lm(
    args: argparse.Namespace,
    *,
    max_tokens: int,
    temperature: float,
) -> dspy.LM:
    model = args.model if args.model.startswith("openai/") else f"openai/{args.model}"
    return dspy.LM(
        model=model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        cache=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def _configure_lm(lm: dspy.LM) -> None:
    if hasattr(dspy, "configure"):
        dspy.configure(lm=lm)
    else:
        dspy.settings.configure(lm=lm)


def _gepa_class() -> Any:
    if hasattr(dspy, "GEPA"):
        return dspy.GEPA
    from dspy.teleprompt import GEPA

    return GEPA


def _extract_instructions(program: Any) -> str:
    for path in (("classify", "predict", "signature"), ("classify", "signature")):
        current = program
        try:
            for attribute in path:
                current = getattr(current, attribute)
        except AttributeError:
            continue
        instructions = getattr(current, "instructions", None)
        if instructions:
            return str(instructions)
    return "Could not extract optimized instructions from this DSPy version."


def _evaluate_and_collect(
    program: Any,
    examples: Sequence[Any],
    *,
    kappa_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_all: list[str] = []
    predicted_all: list[str | None] = []
    gold_valid: list[str] = []
    predicted_valid: list[str] = []
    rows: list[dict[str, Any]] = []
    correct = 0

    for example in examples:
        gold = str(example.gold_label)
        gold_all.append(gold)
        try:
            prediction = program(
                problem_statement=example.problem_statement,
                previous_sentence=example.previous_sentence,
                sentence=example.sentence,
                next_sentence=example.next_sentence,
            )
            predicted = parse_sentence_label(prediction)
            predicted_all.append(predicted)
            is_correct = predicted == gold
            correct += int(is_correct)
            gold_valid.append(gold)
            predicted_valid.append(predicted)
            row = {
                "question_id": example.question_id,
                "unit_id": example.unit_id,
                "problem_statement": example.problem_statement,
                "previous_sentence": example.previous_sentence,
                "sentence": example.sentence,
                "next_sentence": example.next_sentence,
                "gold_label": gold,
                "predicted_label": predicted,
                "correct": is_correct,
            }
        except Exception as exc:  # noqa: BLE001 - failed LM calls count as invalid predictions
            predicted_all.append(None)
            row = {
                "question_id": example.question_id,
                "unit_id": example.unit_id,
                "problem_statement": example.problem_statement,
                "previous_sentence": example.previous_sentence,
                "sentence": example.sentence,
                "next_sentence": example.next_sentence,
                "gold_label": gold,
                "error": str(exc),
                "correct": False,
            }
        rows.append(row)

    total = len(examples)
    valid = len(predicted_valid)
    report: dict[str, Any] = {
        "sentences": total,
        "valid_predictions": valid,
        "failed_predictions": total - valid,
        "coverage": valid / total if total else 0.0,
        "strict_accuracy": correct / total if total else 0.0,
    }
    if gold_all:
        report["strict_classification"] = compute_classification_metrics(
            gold_all,
            predicted_all,
        )
    if predicted_valid:
        agreement = compute_sentence_agreement(
            gold_valid,
            predicted_valid,
            kappa_weight=kappa_weight,
        )
        report["corpus_agreement_valid_predictions"] = agreement.to_dict()
        report["coverage_adjusted_score"] = report["coverage"] * agreement.score
    return report, rows


def _append_stats(path: Path, row: dict[str, Any]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        old_fields = reader.fieldnames or []
        old_rows = list(reader)
    fields = old_fields + [field for field in row if field not in old_fields]
    if fields != old_fields:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerow(row)
    else:
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _selection_value(report: dict[str, Any], metric: str) -> float:
    if metric == "accuracy":
        return float(report["strict_accuracy"])
    classification = report.get("strict_classification", {})
    return float(classification.get(metric, 0.0))


def _select_program(
    seed_program: Any,
    optimized_program: Any,
    seed_report: dict[str, Any],
    optimized_report: dict[str, Any],
    *,
    selection_metric: str,
    max_class_recall_drop: float,
) -> tuple[Any, str, dict[str, Any]]:
    seed_score = _selection_value(seed_report, selection_metric)
    optimized_score = _selection_value(optimized_report, selection_metric)
    seed_classes = seed_report["strict_classification"]["per_class"]
    optimized_classes = optimized_report["strict_classification"]["per_class"]
    recall_drops = {
        label: float(seed_classes[label]["recall"]) - float(optimized_classes[label]["recall"])
        for label in SENTENCE_LABELS
        if int(seed_classes[label]["support"]) > 0
    }
    excessive_drops = {
        label: drop for label, drop in recall_drops.items() if drop > max_class_recall_drop
    }
    use_optimized = optimized_score > seed_score and not excessive_drops
    decision = {
        "metric": selection_metric,
        "seed_score": seed_score,
        "gepa_optimized_score": optimized_score,
        "max_allowed_class_recall_drop": max_class_recall_drop,
        "class_recall_drops": recall_drops,
        "excessive_class_recall_drops": excessive_drops,
        "selected": "gepa_optimized" if use_optimized else "seed",
        "reason": (
            "optimized prompt improved the selection metric without violating the recall gate"
            if use_optimized
            else (
                "seed retained because the optimized prompt violated the class-recall gate"
                if excessive_drops
                else "seed retained because the optimized prompt did not improve the selection metric"
            )
        ),
    }
    return (
        (optimized_program, "gepa_optimized", decision)
        if use_optimized
        else (seed_program, "seed", decision)
    )


def _gepa_kwargs(
    args: argparse.Namespace,
    *,
    metric: Callable[..., dspy.Prediction],
    reflection_lm: dspy.LM,
    log_dir: Path,
    seed: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "metric": metric,
        "num_threads": args.num_threads,
        "track_stats": True,
        "track_best_outputs": False,
        "reflection_lm": reflection_lm,
        "log_dir": str(log_dir),
        "seed": seed,
    }
    if args.gepa_auto:
        kwargs["auto"] = args.gepa_auto
    elif args.max_full_evals is not None:
        kwargs["max_full_evals"] = args.max_full_evals
    else:
        kwargs["max_metric_calls"] = args.max_metric_calls
    return kwargs


def _optimize_and_select(
    args: argparse.Namespace,
    *,
    trainset: Sequence[Any],
    valset: Sequence[Any],
    seed_instructions: str,
    metric: Callable[..., dspy.Prediction],
    reflection_lm: dspy.LM,
    log_dir: Path,
    seed: int,
) -> dict[str, Any]:
    seed_program = EpisodeJudge(seed_instructions)
    seed_report, seed_predictions = _evaluate_and_collect(
        seed_program,
        valset,
        kappa_weight=args.kappa_weight,
    )
    optimized_program = _gepa_class()(
        **_gepa_kwargs(
            args,
            metric=metric,
            reflection_lm=reflection_lm,
            log_dir=log_dir,
            seed=seed,
        )
    ).compile(
        seed_program,
        trainset=trainset,
        valset=valset,
    )
    optimized_report, optimized_predictions = _evaluate_and_collect(
        optimized_program,
        valset,
        kappa_weight=args.kappa_weight,
    )
    selected_program, selected_name, decision = _select_program(
        seed_program,
        optimized_program,
        seed_report,
        optimized_report,
        selection_metric=args.selection_metric,
        max_class_recall_drop=args.max_class_recall_drop,
    )
    return {
        "seed_program": seed_program,
        "seed_report": seed_report,
        "seed_predictions": seed_predictions,
        "optimized_program": optimized_program,
        "optimized_report": optimized_report,
        "optimized_predictions": optimized_predictions,
        "selected_program": selected_program,
        "selected_name": selected_name,
        "selection": decision,
    }


def _truncate_documents(
    documents: Sequence[EpisodeDocument],
    max_units: int | None,
) -> list[EpisodeDocument]:
    if max_units is None:
        return list(documents)
    if max_units <= 0:
        raise ValueError("--max-units-per-document must be positive")
    return [
        EpisodeDocument(
            question_id=document.question_id,
            units=document.units[:max_units],
            problem_statement=document.problem_statement,
        )
        for document in documents
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a seven-class Schoenfeld sentence judge with GEPA and llama.cpp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root of MingLiiii/Schoenfeld_Reasoning or its responses_labeled directory.",
    )
    parser.add_argument("--train-documents", type=int, default=26)
    parser.add_argument("--val-documents", type=int, default=6)
    parser.add_argument(
        "--test-documents",
        type=int,
        default=None,
        help="Limit held-out evaluation documents; intended only for smoke tests.",
    )
    parser.add_argument(
        "--max-units-per-document",
        type=int,
        default=None,
        help="Truncate each response to its first N sentences; intended only for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-variant",
        choices=("base", "few-shot"),
        default="few-shot",
        help="Use the contextual guide alone or append audited contrastive demonstrations.",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=int,
        default=21,
        help="Number of balanced examples drawn from the 21-example curated bank.",
    )
    parser.add_argument(
        "--gepa-reward",
        choices=("balanced", "exact"),
        default="balanced",
        help="Use inverse-frequency exact match (balanced accuracy) or plain exact match.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("balanced_accuracy", "macro_f1", "accuracy"),
        default="balanced_accuracy",
        help="Validation metric used by the seed-versus-optimized safety gate.",
    )
    parser.add_argument(
        "--max-class-recall-drop",
        type=float,
        default=0.10,
        help="Reject the optimized prompt if any validation class recall drops by more than this.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Run nested response-grouped CV with this many outer folds; 0 runs a final fit.",
    )
    parser.add_argument(
        "--cv-inner-val-documents",
        type=int,
        default=5,
        help="Inner validation responses used for GEPA and the safety gate in each CV fold.",
    )
    parser.add_argument(
        "--locked-test-documents",
        type=int,
        default=6,
        help="Responses excluded entirely from nested CV.",
    )
    parser.add_argument(
        "--evaluate-locked-test",
        action="store_true",
        help="Explicitly authorize one evaluation of the locked test split after final selection.",
    )

    parser.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key", default=os.getenv("LLAMA_API_KEY", "local-llamacpp-key"))
    parser.add_argument("--model", default="local-llamacpp")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--reflection-temperature", type=float, default=0.7)
    parser.add_argument("--reflection-max-tokens", type=int, default=2048)
    parser.add_argument("--num-threads", type=int, default=1)

    parser.add_argument(
        "--kappa-weight",
        type=float,
        default=0.5,
        help="Reporting weight of scaled Cohen kappa versus scaled Kendall tau-b.",
    )
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--gepa-auto", choices=("light", "medium", "heavy"))
    budget.add_argument("--max-full-evals", type=int)
    budget.add_argument("--max-metric-calls", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp0a"))
    args = parser.parse_args()
    if not 0.0 <= args.max_class_recall_drop <= 1.0:
        parser.error("--max-class-recall-drop must be in [0, 1]")
    if args.cv_folds == 1:
        parser.error("--cv-folds must be 0 or at least 2")
    if args.cv_folds and args.evaluate_locked_test:
        parser.error("nested CV never evaluates the locked test; remove --evaluate-locked-test")
    return args


def _report_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    kappa_weight: float,
) -> dict[str, Any]:
    gold = [str(row["gold_label"]) for row in rows]
    predicted = [row.get("predicted_label") for row in rows]
    valid_pairs = [
        (expected, actual) for expected, actual in zip(gold, predicted) if isinstance(actual, str)
    ]
    report: dict[str, Any] = {
        "sentences": len(rows),
        "valid_predictions": len(valid_pairs),
        "failed_predictions": len(rows) - len(valid_pairs),
        "coverage": len(valid_pairs) / len(rows) if rows else 0.0,
        "strict_accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else 0.0,
    }
    if rows:
        report["strict_classification"] = compute_classification_metrics(gold, predicted)
    if valid_pairs:
        agreement = compute_sentence_agreement(
            [pair[0] for pair in valid_pairs],
            [pair[1] for pair in valid_pairs],
            kappa_weight=kappa_weight,
        )
        report["corpus_agreement_valid_predictions"] = agreement.to_dict()
        report["coverage_adjusted_score"] = report["coverage"] * agreement.score
    return report


def _save_program(program: Any, path: Path) -> None:
    try:
        program.save(path)
    except Exception as exc:  # noqa: BLE001 - serialization support varies by DSPy version
        print(f"Warning: DSPy could not serialize {path.name}: {exc}")


def _few_shot_configuration(
    args: argparse.Namespace,
    documents: Sequence[EpisodeDocument],
) -> tuple[tuple[Any, ...], str]:
    if args.prompt_variant != "few-shot":
        return (), SEED_INSTRUCTIONS
    examples = select_few_shot_sentences(documents, count=args.few_shot_examples)
    print(
        "Curated demonstrations: "
        + ", ".join(f"{example.example_id}={example.label}" for example in examples)
    )
    return examples, build_few_shot_instructions(examples)


def _common_result_metadata(
    args: argparse.Namespace,
    *,
    timestamp: str,
    few_shot_examples: Sequence[Any],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "model": args.model,
        "api_base": args.api_base,
        "seed": args.seed,
        "task": "context-aware seven-class Schoenfeld episode classification",
        "context_fields": [
            "problem_statement",
            "previous_sentence",
            "current_sentence",
            "next_sentence",
        ],
        "prompt_variant": args.prompt_variant,
        "few_shot": {
            "requested_examples": args.few_shot_examples,
            "selected_examples": [example.metadata() for example in few_shot_examples],
        },
        "metric_definition": {
            "gepa_reward": (
                "inverse-frequency weighted exact match; split mean equals balanced accuracy"
                if args.gepa_reward == "balanced"
                else "per-unit exact match"
            ),
            "selection_metric": args.selection_metric,
            "max_class_recall_drop": args.max_class_recall_drop,
            "reporting": (
                "strict accuracy, balanced accuracy, macro-F1, per-class metrics, "
                "Cohen kappa, and Kendall tau-b"
            ),
            "kappa_weight": args.kappa_weight,
            "kendall_tau_b_weight": 1.0 - args.kappa_weight,
            "kendall_label_order": list(SENTENCE_LABELS),
        },
    }


def _run_cross_validation(
    args: argparse.Namespace,
    *,
    documents: Sequence[EpisodeDocument],
    timestamp: str,
    seed_instructions: str,
    few_shot_examples: Sequence[Any],
    metric: Callable[..., dspy.Prediction],
    reflection_lm: dspy.LM,
) -> dict[str, Any]:
    folds, locked_test = make_nested_group_folds(
        documents,
        folds=args.cv_folds,
        locked_test_documents=args.locked_test_documents,
        inner_val_documents=args.cv_inner_val_documents,
        seed=args.seed,
    )
    all_outer_predictions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    prompts_dir = args.output_dir / f"cv_prompts_{timestamp}"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for fold_index, (train_docs, inner_val_docs, outer_test_docs) in enumerate(folds):
        fold_number = fold_index + 1
        print(
            f"Nested CV fold {fold_number}/{len(folds)}: "
            f"documents={len(train_docs)}/{len(inner_val_docs)}/{len(outer_test_docs)}"
        )
        trainset = documents_to_dspy_examples(train_docs, dspy)
        inner_valset = documents_to_dspy_examples(inner_val_docs, dspy)
        outer_testset = documents_to_dspy_examples(outer_test_docs, dspy)
        outcome = _optimize_and_select(
            args,
            trainset=trainset,
            valset=inner_valset,
            seed_instructions=seed_instructions,
            metric=metric,
            reflection_lm=reflection_lm,
            log_dir=args.output_dir / "gepa_logs" / f"fold_{fold_number}",
            seed=args.seed + fold_index,
        )
        outer_report, outer_predictions = _evaluate_and_collect(
            outcome["selected_program"],
            outer_testset,
            kappa_weight=args.kappa_weight,
        )
        for row in outer_predictions:
            row["outer_fold"] = fold_number
        all_outer_predictions.extend(outer_predictions)

        selected_instructions = _extract_instructions(outcome["selected_program"])
        (prompts_dir / f"fold_{fold_number}_selected.txt").write_text(
            selected_instructions,
            encoding="utf-8",
        )
        _save_program(
            outcome["selected_program"],
            prompts_dir / f"fold_{fold_number}_selected_program.json",
        )
        fold_reports.append(
            {
                "fold": fold_number,
                "train_question_ids": [document.question_id for document in train_docs],
                "inner_validation_question_ids": [
                    document.question_id for document in inner_val_docs
                ],
                "outer_test_question_ids": [document.question_id for document in outer_test_docs],
                "sentence_counts": {
                    "train": len(trainset),
                    "inner_validation": len(inner_valset),
                    "outer_test": len(outer_testset),
                },
                "seed_inner_validation": outcome["seed_report"],
                "gepa_optimized_inner_validation": outcome["optimized_report"],
                "selection": outcome["selection"],
                "outer_test": outer_report,
            }
        )

    result = _common_result_metadata(
        args,
        timestamp=timestamp,
        few_shot_examples=few_shot_examples,
    )
    result.update(
        {
            "mode": "nested_response_grouped_cross_validation",
            "cv_folds": args.cv_folds,
            "inner_validation_documents": args.cv_inner_val_documents,
            "folds": fold_reports,
            "aggregate_outer_test": _report_from_rows(
                all_outer_predictions,
                kappa_weight=args.kappa_weight,
            ),
            "locked_test_evaluated": False,
            "locked_test_question_ids": [document.question_id for document in locked_test],
        }
    )
    _write_jsonl(
        args.output_dir / f"cross_validation_predictions_{timestamp}.jsonl",
        all_outer_predictions,
    )
    (args.output_dir / f"cross_validation_{timestamp}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    raw_documents = load_documents(args.dataset_dir)
    documents = _truncate_documents(raw_documents, args.max_units_per_document)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    audit = annotation_audit(raw_documents)
    audit_path = args.output_dir / f"annotation_audit_{timestamp}.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if audit["documents_missing_problem_context"]:
        print(
            "Warning: problem context is unavailable for "
            f"{audit['documents_missing_problem_context']} documents."
        )

    task_lm = _make_lm(args, max_tokens=args.max_tokens, temperature=args.temperature)
    reflection_lm = _make_lm(
        args,
        max_tokens=args.reflection_max_tokens,
        temperature=args.reflection_temperature,
    )
    _configure_lm(task_lm)
    metric = create_metric(class_balanced=args.gepa_reward == "balanced")
    few_shot_examples, seed_instructions = _few_shot_configuration(args, documents)
    (args.output_dir / f"seed_prompt_{timestamp}.txt").write_text(
        seed_instructions,
        encoding="utf-8",
    )

    if args.cv_folds:
        result = _run_cross_validation(
            args,
            documents=documents,
            timestamp=timestamp,
            seed_instructions=seed_instructions,
            few_shot_examples=few_shot_examples,
            metric=metric,
            reflection_lm=reflection_lm,
        )
        result["annotation_audit_file"] = audit_path.name
        (args.output_dir / f"cross_validation_{timestamp}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    train_docs, val_docs, test_docs = split_documents(
        documents,
        train_size=args.train_documents,
        val_size=args.val_documents,
        seed=args.seed,
    )
    if args.test_documents is not None:
        if args.test_documents <= 0:
            raise ValueError("--test-documents must be positive")
        test_docs = test_docs[: args.test_documents]
    trainset = documents_to_dspy_examples(train_docs, dspy)
    valset = documents_to_dspy_examples(val_docs, dspy)
    testset = documents_to_dspy_examples(test_docs, dspy)
    print(
        f"Loaded response-grouped split: documents={len(train_docs)}/{len(val_docs)}/"
        f"{len(test_docs)}, sentences={len(trainset)}/{len(valset)}/{len(testset)}"
    )
    print(
        "Running GEPA with "
        + (
            "class-balanced exact-match feedback..."
            if args.gepa_reward == "balanced"
            else "plain exact-match feedback..."
        )
    )
    outcome = _optimize_and_select(
        args,
        trainset=trainset,
        valset=valset,
        seed_instructions=seed_instructions,
        metric=metric,
        reflection_lm=reflection_lm,
        log_dir=args.output_dir / "gepa_logs",
        seed=args.seed,
    )
    selected_report = (
        outcome["optimized_report"]
        if outcome["selected_name"] == "gepa_optimized"
        else outcome["seed_report"]
    )
    selected_predictions = (
        outcome["optimized_predictions"]
        if outcome["selected_name"] == "gepa_optimized"
        else outcome["seed_predictions"]
    )

    test_report = None
    test_predictions: list[dict[str, Any]] = []
    if args.evaluate_locked_test:
        print("Explicit authorization received; evaluating the selected prompt on locked test.")
        test_report, test_predictions = _evaluate_and_collect(
            outcome["selected_program"],
            testset,
            kappa_weight=args.kappa_weight,
        )
    else:
        print("Locked test not evaluated. Re-run once with --evaluate-locked-test when ready.")

    optimized_instructions = _extract_instructions(outcome["optimized_program"])
    selected_instructions = _extract_instructions(outcome["selected_program"])
    result = _common_result_metadata(
        args,
        timestamp=timestamp,
        few_shot_examples=few_shot_examples,
    )
    result.update(
        {
            "mode": "final_fit",
            "annotation_audit_file": audit_path.name,
            "smoke_limits": {
                "max_units_per_document": args.max_units_per_document,
                "test_documents": args.test_documents,
            },
            "split_question_ids": {
                "train": [document.question_id for document in train_docs],
                "validation": [document.question_id for document in val_docs],
                "locked_test": [document.question_id for document in test_docs],
            },
            "split_sentence_counts": {
                "train": len(trainset),
                "validation": len(valset),
                "locked_test": len(testset),
            },
            "seed_validation": outcome["seed_report"],
            "gepa_optimized_validation": outcome["optimized_report"],
            "optimized_validation": outcome["optimized_report"],
            "selection": outcome["selection"],
            "selected_validation": selected_report,
            "selected_program": outcome["selected_name"],
            "locked_test_evaluated": args.evaluate_locked_test,
            "test": test_report,
            "gepa_optimized_instructions": optimized_instructions,
            "selected_instructions": selected_instructions,
        }
    )

    (args.output_dir / f"results_{timestamp}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(
        args.output_dir / f"seed_validation_predictions_{timestamp}.jsonl",
        outcome["seed_predictions"],
    )
    _write_jsonl(
        args.output_dir / f"gepa_optimized_validation_predictions_{timestamp}.jsonl",
        outcome["optimized_predictions"],
    )
    _write_jsonl(
        args.output_dir / f"selected_validation_predictions_{timestamp}.jsonl",
        selected_predictions,
    )
    if args.evaluate_locked_test:
        _write_jsonl(
            args.output_dir / f"test_predictions_{timestamp}.jsonl",
            test_predictions,
        )
    (args.output_dir / f"gepa_optimized_prompt_{timestamp}.txt").write_text(
        optimized_instructions,
        encoding="utf-8",
    )
    (args.output_dir / f"selected_prompt_{timestamp}.txt").write_text(
        selected_instructions,
        encoding="utf-8",
    )
    _save_program(
        outcome["optimized_program"],
        args.output_dir / f"gepa_optimized_program_{timestamp}.json",
    )
    _save_program(
        outcome["selected_program"],
        args.output_dir / f"selected_program_{timestamp}.json",
    )

    selected_nominal = selected_report["strict_classification"]
    test_agreement = (
        test_report.get("corpus_agreement_valid_predictions", {}) if test_report else {}
    )
    _append_stats(
        args.output_dir / "stats.csv",
        {
            "timestamp": timestamp,
            "model": args.model,
            "prompt_variant": args.prompt_variant,
            "gepa_reward": args.gepa_reward,
            "selected_program": outcome["selected_name"],
            "train_sentences": len(trainset),
            "val_sentences": len(valset),
            "locked_test_sentences": len(testset),
            "seed_val_accuracy": outcome["seed_report"]["strict_accuracy"],
            "gepa_val_accuracy": outcome["optimized_report"]["strict_accuracy"],
            "selected_val_accuracy": selected_report["strict_accuracy"],
            "selected_val_balanced_accuracy": selected_nominal["balanced_accuracy"],
            "selected_val_macro_f1": selected_nominal["macro_f1"],
            "locked_test_evaluated": args.evaluate_locked_test,
            "test_accuracy": test_report["strict_accuracy"] if test_report else "",
            "test_balanced_accuracy": (
                test_report["strict_classification"]["balanced_accuracy"] if test_report else ""
            ),
            "test_macro_f1": (
                test_report["strict_classification"]["macro_f1"] if test_report else ""
            ),
            "test_kappa": test_agreement.get("cohen_kappa", ""),
            "test_kendall": test_agreement.get("kendall_tau_b", ""),
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

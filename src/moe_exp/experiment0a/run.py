from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import dspy

from .data import EpisodeDocument, documents_to_dspy_examples, load_documents, split_documents
from .metrics import (
    Annotation,
    AnnotationParseError,
    compute_agreement,
    concatenate_annotations,
    parse_annotations,
)
from .prompts import SEED_INSTRUCTIONS


class EpisodeJudge(dspy.Module):
    """The single prompt-bearing module optimized by GEPA."""

    def __init__(self) -> None:
        super().__init__()
        signature = dspy.Signature(
            "problem, response -> annotations",
            instructions=SEED_INSTRUCTIONS,
        )
        self.annotate = dspy.Predict(signature)

    def forward(self, problem: str, response: str) -> Any:
        return self.annotate(problem=problem, response=response)


def _gold_from_example(example: Any) -> list[Annotation]:
    return parse_annotations(example.gold_annotations)


def create_metric(
    kappa_weight: float,
    paragraph_weight: float,
) -> Callable[..., dspy.Prediction]:
    """Create an agreement metric with feedback suitable for GEPA reflection."""

    def metric(
        example: Any,
        pred: Any,
        trace: Any = None,
        pred_name: Any = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        del trace, pred_name, pred_trace
        gold = _gold_from_example(example)
        try:
            predicted = parse_annotations(pred, expected_count=len(gold))
        except AnnotationParseError as exc:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"OUTPUT CONTRACT ERROR: {exc}. Return only the complete JSON array, with one "
                    "object per input id and valid paragraph_label and sentence_label values."
                ),
            )

        agreement = compute_agreement(
            gold,
            predicted,
            kappa_weight=kappa_weight,
            paragraph_weight=paragraph_weight,
        )
        paragraph_errors = [
            item.unit_id
            for item, target in zip(predicted, gold)
            if item.paragraph_label != target.paragraph_label
        ]
        sentence_errors = [
            item.unit_id
            for item, target in zip(predicted, gold)
            if item.sentence_label != target.sentence_label
        ]
        feedback = (
            f"Agreement: paragraph kappa={agreement.paragraph_kappa:.3f}, "
            f"paragraph Kendall tau-b={agreement.paragraph_kendall_tau:.3f}, "
            f"sentence kappa={agreement.sentence_kappa:.3f}, "
            f"sentence Kendall tau-b={agreement.sentence_kendall_tau:.3f}. "
            f"Misclassified paragraph ids: {paragraph_errors[:20]}; "
            f"misclassified sentence ids: {sentence_errors[:20]}. "
            "Improve the distinctions in the annotation instructions that caused these errors."
        )
        return dspy.Prediction(score=agreement.score, feedback=feedback)

    return metric


def _make_lm(args: argparse.Namespace) -> dspy.LM:
    model = args.model if args.model.startswith("openai/") else f"openai/{args.model}"
    return dspy.LM(
        model=model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
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


def _evaluate_class() -> Any:
    if hasattr(dspy, "Evaluate"):
        return dspy.Evaluate
    from dspy.evaluate import Evaluate

    return Evaluate


def _score_to_fraction(value: Any) -> float:
    score = float(value)
    return score / 100.0 if score > 1.0 else score


def _extract_instructions(program: Any) -> str:
    for path in (("annotate", "predict", "signature"), ("annotate", "signature")):
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
    paragraph_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_groups: list[list[Annotation]] = []
    predicted_groups: list[list[Annotation]] = []
    rows: list[dict[str, Any]] = []
    failures = 0

    for example in examples:
        gold = _gold_from_example(example)
        try:
            prediction = program(problem=example.problem, response=example.response)
            predicted = parse_annotations(prediction, expected_count=len(gold))
            agreement = compute_agreement(
                gold,
                predicted,
                kappa_weight=kappa_weight,
                paragraph_weight=paragraph_weight,
            )
            gold_groups.append(gold)
            predicted_groups.append(predicted)
            row = {
                "question_id": example.question_id,
                "metrics": agreement.to_dict(),
                "prediction": [item.__dict__ for item in predicted],
                "gold": [item.__dict__ for item in gold],
            }
        except Exception as exc:
            failures += 1
            row = {"question_id": example.question_id, "error": str(exc)}
        rows.append(row)

    report: dict[str, Any] = {
        "documents": len(examples),
        "valid_documents": len(predicted_groups),
        "failed_documents": failures,
    }
    if predicted_groups:
        corpus = compute_agreement(
            concatenate_annotations(gold_groups),
            concatenate_annotations(predicted_groups),
            kappa_weight=kappa_weight,
            paragraph_weight=paragraph_weight,
        )
        report["corpus_agreement"] = corpus.to_dict()
        report["mean_document_score"] = sum(
            row["metrics"]["score"] for row in rows if "metrics" in row
        ) / len(predicted_groups)
    return report, rows


def _append_stats(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a Schoenfeld episode-annotation judge with GEPA and llama.cpp.",
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
        help="Truncate each response to its first N units; intended only for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key", default=os.getenv("LLAMA_API_KEY", "local-llamacpp-key"))
    parser.add_argument("--model", default="local-llamacpp")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--num-threads", type=int, default=3)

    parser.add_argument(
        "--kappa-weight",
        type=float,
        default=0.5,
        help="Weight of Cohen's kappa versus Kendall tau-b within each annotation level.",
    )
    parser.add_argument(
        "--paragraph-weight",
        type=float,
        default=0.5,
        help="Weight of paragraph-level versus sentence-level agreement.",
    )
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--gepa-auto", choices=("light", "medium", "heavy"))
    budget.add_argument("--max-full-evals", type=int)
    budget.add_argument("--max-metric-calls", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp0a"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.dataset_dir)
    if args.max_units_per_document is not None:
        if args.max_units_per_document <= 0:
            raise ValueError("--max-units-per-document must be positive")
        documents = [
            EpisodeDocument(
                document.question_id,
                document.problem,
                document.units[: args.max_units_per_document],
            )
            for document in documents
        ]
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lm = _make_lm(args)
    _configure_lm(lm)
    metric = create_metric(args.kappa_weight, args.paragraph_weight)
    seed_program = EpisodeJudge()

    Evaluate = _evaluate_class()
    evaluator = Evaluate(
        devset=valset,
        metric=lambda example, pred, trace=None: metric(example, pred, trace).score,
        num_threads=args.num_threads,
        display_progress=True,
    )
    print(f"Loaded document split: train={len(trainset)}, val={len(valset)}, test={len(testset)}")
    print("Evaluating seed prompt...")
    seed_score = _score_to_fraction(evaluator(seed_program))

    gepa_kwargs: dict[str, Any] = {
        "metric": metric,
        "num_threads": args.num_threads,
        "track_stats": True,
        "track_best_outputs": False,
        "reflection_lm": lm,
        "log_dir": str(args.output_dir / "gepa_logs"),
        "seed": args.seed,
    }
    if args.gepa_auto:
        gepa_kwargs["auto"] = args.gepa_auto
    elif args.max_full_evals is not None:
        gepa_kwargs["max_full_evals"] = args.max_full_evals
    else:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls

    print("Running GEPA optimization...")
    optimized = _gepa_class()(**gepa_kwargs).compile(
        seed_program,
        trainset=trainset,
        valset=valset,
    )
    optimized_val_score = _score_to_fraction(evaluator(optimized))
    print("Evaluating the optimized prompt once on the untouched test split...")
    test_report, predictions = _evaluate_and_collect(
        optimized,
        testset,
        kappa_weight=args.kappa_weight,
        paragraph_weight=args.paragraph_weight,
    )

    split_ids = {
        "train": [document.question_id for document in train_docs],
        "validation": [document.question_id for document in val_docs],
        "test": [document.question_id for document in test_docs],
    }
    instructions = _extract_instructions(optimized)
    result = {
        "timestamp": timestamp,
        "model": args.model,
        "api_base": args.api_base,
        "seed": args.seed,
        "smoke_limits": {
            "max_units_per_document": args.max_units_per_document,
            "test_documents": args.test_documents,
        },
        "weights": {
            "kappa": args.kappa_weight,
            "kendall_tau_b": 1.0 - args.kappa_weight,
            "paragraph": args.paragraph_weight,
            "sentence": 1.0 - args.paragraph_weight,
        },
        "split_question_ids": split_ids,
        "seed_validation_score": seed_score,
        "optimized_validation_score": optimized_val_score,
        "test": test_report,
        "optimized_instructions": instructions,
    }

    stem = f"{timestamp}"
    (args.output_dir / f"results_{stem}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / f"test_predictions_{stem}.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / f"optimized_prompt_{stem}.txt").write_text(instructions, encoding="utf-8")
    try:
        optimized.save(args.output_dir / f"optimized_program_{stem}.json")
    except Exception as exc:
        print(f"Warning: DSPy could not serialize the optimized program: {exc}")
    _append_stats(
        args.output_dir / "stats.csv",
        {
            "timestamp": timestamp,
            "model": args.model,
            "train_documents": len(trainset),
            "val_documents": len(valset),
            "test_documents": len(testset),
            "seed_validation_score": seed_score,
            "optimized_validation_score": optimized_val_score,
            "test_score": test_report.get("corpus_agreement", {}).get("score", ""),
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

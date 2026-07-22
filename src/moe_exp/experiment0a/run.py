from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import dspy

from .data import (
    SENTENCE_LABELS,
    EpisodeDocument,
    documents_to_dspy_examples,
    load_documents,
    split_documents,
)
from .metrics import LabelParseError, compute_sentence_agreement, parse_sentence_label
from .prompts import (
    SEED_INSTRUCTIONS,
    build_few_shot_instructions,
    select_few_shot_sentences,
)


class EpisodeJudge(dspy.Module):
    """The single sentence-classification prompt optimized by GEPA."""

    def __init__(self, instructions: str = SEED_INSTRUCTIONS) -> None:
        super().__init__()
        signature = dspy.Signature("sentence -> label", instructions=instructions)
        self.classify = dspy.Predict(signature)

    def forward(self, sentence: str) -> Any:
        return self.classify(sentence=sentence)


def create_metric() -> Callable[..., dspy.Prediction]:
    """Return decomposable exact-match feedback for GEPA's per-example API."""

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
        return dspy.Prediction(score=1.0 if correct else 0.0, feedback=feedback)

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
    gold_valid: list[str] = []
    predicted_valid: list[str] = []
    rows: list[dict[str, Any]] = []
    correct = 0

    for example in examples:
        gold = str(example.gold_label)
        try:
            prediction = program(sentence=example.sentence)
            predicted = parse_sentence_label(prediction)
            is_correct = predicted == gold
            correct += int(is_correct)
            gold_valid.append(gold)
            predicted_valid.append(predicted)
            row = {
                "question_id": example.question_id,
                "unit_id": example.unit_id,
                "sentence": example.sentence,
                "gold_label": gold,
                "predicted_label": predicted,
                "correct": is_correct,
            }
        except Exception as exc:
            row = {
                "question_id": example.question_id,
                "unit_id": example.unit_id,
                "sentence": example.sentence,
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
        default="base",
        help="Use the label guide alone or append train-only single-sentence demonstrations.",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=int,
        default=7,
        help="Number of gold training sentences appended by the few-shot variant.",
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

    task_lm = _make_lm(args, max_tokens=args.max_tokens, temperature=args.temperature)
    reflection_lm = _make_lm(
        args,
        max_tokens=args.reflection_max_tokens,
        temperature=args.reflection_temperature,
    )
    _configure_lm(task_lm)
    metric = create_metric()

    few_shot_examples = ()
    seed_instructions = SEED_INSTRUCTIONS
    if args.prompt_variant == "few-shot":
        few_shot_examples = select_few_shot_sentences(
            train_docs,
            count=args.few_shot_examples,
        )
        seed_instructions = build_few_shot_instructions(few_shot_examples)
        print(
            "Few-shot demonstrations: "
            + ", ".join(
                f"{example.question_id}:{example.unit_id}={example.label}"
                for example in few_shot_examples
            )
        )
    seed_program = EpisodeJudge(seed_instructions)

    print(
        f"Loaded response-grouped split: documents={len(train_docs)}/{len(val_docs)}/{len(test_docs)}, "
        f"sentences={len(trainset)}/{len(valset)}/{len(testset)}"
    )
    print("Evaluating the seed prompt on validation sentences...")
    seed_val_report, seed_val_predictions = _evaluate_and_collect(
        seed_program,
        valset,
        kappa_weight=args.kappa_weight,
    )

    gepa_kwargs: dict[str, Any] = {
        "metric": metric,
        "num_threads": args.num_threads,
        "track_stats": True,
        "track_best_outputs": False,
        "reflection_lm": reflection_lm,
        "log_dir": str(args.output_dir / "gepa_logs"),
        "seed": args.seed,
    }
    if args.gepa_auto:
        gepa_kwargs["auto"] = args.gepa_auto
    elif args.max_full_evals is not None:
        gepa_kwargs["max_full_evals"] = args.max_full_evals
    else:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls

    print("Running GEPA with per-sentence exact-match feedback...")
    optimized = _gepa_class()(**gepa_kwargs).compile(
        seed_program,
        trainset=trainset,
        valset=valset,
    )
    print("Evaluating global validation agreement for the optimized prompt...")
    optimized_val_report, optimized_val_predictions = _evaluate_and_collect(
        optimized,
        valset,
        kappa_weight=args.kappa_weight,
    )
    print("Evaluating the optimized prompt once on the untouched test split...")
    test_report, test_predictions = _evaluate_and_collect(
        optimized,
        testset,
        kappa_weight=args.kappa_weight,
    )

    instructions = _extract_instructions(optimized)
    result = {
        "timestamp": timestamp,
        "model": args.model,
        "api_base": args.api_base,
        "seed": args.seed,
        "task": "single-sentence seven-class Schoenfeld episode classification",
        "prompt_variant": args.prompt_variant,
        "few_shot": {
            "requested_examples": args.few_shot_examples,
            "selected_examples": [example.metadata() for example in few_shot_examples],
        },
        "metric_definition": {
            "gepa_reward": "per-sentence exact match (1 correct, 0 incorrect)",
            "reporting": "corpus-level Cohen kappa and Kendall tau-b",
            "kappa_weight": args.kappa_weight,
            "kendall_tau_b_weight": 1.0 - args.kappa_weight,
            "kendall_label_order": list(SENTENCE_LABELS),
        },
        "smoke_limits": {
            "max_units_per_document": args.max_units_per_document,
            "test_documents": args.test_documents,
        },
        "split_question_ids": {
            "train": [document.question_id for document in train_docs],
            "validation": [document.question_id for document in val_docs],
            "test": [document.question_id for document in test_docs],
        },
        "split_sentence_counts": {
            "train": len(trainset),
            "validation": len(valset),
            "test": len(testset),
        },
        "seed_validation": seed_val_report,
        "optimized_validation": optimized_val_report,
        "test": test_report,
        "optimized_instructions": instructions,
    }

    stem = timestamp
    (args.output_dir / f"results_{stem}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_jsonl(
        args.output_dir / f"seed_validation_predictions_{stem}.jsonl", seed_val_predictions
    )
    _write_jsonl(
        args.output_dir / f"optimized_validation_predictions_{stem}.jsonl",
        optimized_val_predictions,
    )
    _write_jsonl(args.output_dir / f"test_predictions_{stem}.jsonl", test_predictions)
    (args.output_dir / f"optimized_prompt_{stem}.txt").write_text(instructions, encoding="utf-8")
    (args.output_dir / f"seed_prompt_{stem}.txt").write_text(seed_instructions, encoding="utf-8")
    try:
        optimized.save(args.output_dir / f"optimized_program_{stem}.json")
    except Exception as exc:
        print(f"Warning: DSPy could not serialize the optimized program: {exc}")

    _append_stats(
        args.output_dir / "stats.csv",
        {
            "timestamp": timestamp,
            "model": args.model,
            "prompt_variant": args.prompt_variant,
            "train_sentences": len(trainset),
            "val_sentences": len(valset),
            "test_sentences": len(testset),
            "seed_val_accuracy": seed_val_report["strict_accuracy"],
            "optimized_val_accuracy": optimized_val_report["strict_accuracy"],
            "optimized_val_kappa": optimized_val_report.get(
                "corpus_agreement_valid_predictions", {}
            ).get("cohen_kappa", ""),
            "optimized_val_kendall": optimized_val_report.get(
                "corpus_agreement_valid_predictions", {}
            ).get("kendall_tau_b", ""),
            "test_accuracy": test_report["strict_accuracy"],
            "test_kappa": test_report.get("corpus_agreement_valid_predictions", {}).get(
                "cohen_kappa", ""
            ),
            "test_kendall": test_report.get("corpus_agreement_valid_predictions", {}).get(
                "kendall_tau_b", ""
            ),
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

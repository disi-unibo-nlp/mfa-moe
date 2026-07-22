#!/usr/bin/env python3
"""Standalone GEPA prompt optimization for legal Q&A generation.

This script has no imports from the surrounding repository. It expects an
OpenAI-compatible endpoint, such as vLLM, for task generation and judging.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import dspy
from datasets import load_dataset
from dotenv import load_dotenv


load_dotenv()


SEED_INSTRUCTIONS = (
    "Generate substantive legal Q&A that requires both documents to answer completely. "
    "CRITICAL: DO NOT ask about document relationships or changes. "
    "FORBIDDEN PATTERNS - Never ask: 'Does X [verb] Y?', 'What changes...?', "
    "'When do provisions start/stop?', 'How does X affect Y?'. "
    "FORBIDDEN REFERENCES - Never use: 'Text1', 'Text2', 'earlier/later document', "
    "'first/second regulation', regulation numbers. "
    "FORBIDDEN ANSWERS - Never answer with: 'Yes', 'No', or relationship statements like 'X repeals Y'. "
    "REQUIRED FOCUS: Ask about substantive procedures, requirements, definitions, criteria, obligations. "
    "REQUIRED CONTEXT: Include specific legal domain terms, such as crop declarations, mutual assistance, "
    "export refunds, notifications, competent authorities, or eligibility criteria when supported by the documents. "
    "ANSWER FORMAT: Write a comprehensive explanation of what the law requires, defines, or establishes, "
    "synthesized from both documents with sufficient detail."
)


JUDGE_PROMPT = """You are an expert evaluator assessing legal Q&A generation quality.

Critical auto-fail constraints:
- No yes/no questions.
- No questions about document relationships, changes, amendments, repeals, extensions, temporal validity, or effects between documents.
- No references to Text1, Text2, Document 1, Document 2, earlier/later document, first/second regulation, or regulation/directive identifiers.
- Answers must not be one-word yes/no answers and must not describe document relationships.

Required high-quality output:
- The reasoning explains why both documents are substantively needed.
- The question asks about substantive legal content: procedures, requirements, definitions, criteria, obligations, or rights.
- The question has enough domain context to stand alone.
- The answer synthesizes supported content from both documents.

Score on three 1-5 metrics:
1. CONSTRAINT_COMPLIANCE
5: no forbidden patterns and clear substantive focus.
3: borderline vague language.
1: any critical violation.

2. SUBSTANTIVE_QUALITY
5: requires both documents and has rich domain-specific content.
3: only partially needs both documents or lacks detail.
1: answerable from one document or lacks substantive content.

3. FACTUAL_ACCURACY
5: all claims are supported by the documents.
3: some unsupported but plausible claims.
1: major fabrication.

Return exactly:
REASONING: [2-3 sentences]
CONSTRAINT_COMPLIANCE: [1-5]
SUBSTANTIVE_QUALITY: [1-5]
FACTUAL_ACCURACY: [1-5]
"""


BILL_REFERENCE_PATTERNS = [
    r"\bRegulation\s+\d+/\d+\b",
    r"\bRegulation\s+\(EEC\)\s+No\s+\d+/\d+\b",
    r"\bDirective\s+\d+/\d+\b",
    r"\bDecision\s+\d+/\d+\b",
    r"\bCouncil\s+Decision\s+\d+",
    r"\bCouncil\s+Regulation\s+\d+",
    r"\bDocument\s+[12]\b",
    r"\bText\s*[12]\b",
    r"\bthe\s+first\s+document\b",
    r"\bthe\s+second\s+document\b",
    r"\bfirst\s+text\b",
    r"\bsecond\s+text\b",
    r"\bearlier\s+(document|regulation|text|provision|obligation|act)\b",
    r"\blater\s+(document|regulation|text|provision|act)\b",
    r"\bthe\s+new\s+(regulation|rule|provision|act)\b",
    r"\bthe\s+old\s+(regulation|rule|provision|act)\b",
    r"\bfirst\s+regulation\b",
    r"\bsecond\s+regulation\b",
]


RELATIONSHIP_LANGUAGE_PATTERNS = [
    r"\bdoes\s+.*\b(repeal|amend|modify|extend|replace|supersede|affect|impact)\b",
    r"\bis\s+.*\b(obsolete|superseded|repealed|amended|modified)\b",
    r"\bwhat\s+change(s|d)?\b",
    r"\bwhat\s+(modifications?|amendments?)\b",
    r"\bwhen\s+do(es)?\s+.*\b(start|stop|cease|begin|end)\s+(applying|to\s+apply)\b",
    r"\bwhen\s+.*\b(become|became)\s+(valid|effective|applicable)\b",
    r"\bhow\s+do(es)?\s+.*\b(affect|modify|change|impact)\b",
    r"\bwhat\s+happens\s+to\s+.*\bprovisions?\b",
    r"\b(extend|repeal|amend|modify|override|supersede)(s|ed|ing)?\s+(the\s+)?validity\b",
    r"\bcease(s|d)?\s+to\s+(be\s+)?in\s+force\b",
    r"\bno\s+longer\s+in\s+force\b",
    r"\bstop(s|ped)?\s+applying\b",
]


class UnifiedQAGenerationModule(dspy.Module):
    """DSPy module whose instructions are optimized by GEPA."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(
            "text1, text2, relation_types -> reasoning, question, answer",
            instructions=SEED_INSTRUCTIONS,
        )

    def forward(self, text1: str, text2: str, relation_types: str) -> Any:
        return self.generate(text1=text1, text2=text2, relation_types=relation_types)


def has_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text or "", re.IGNORECASE) for pattern in patterns)


def validate_output_format(pred: Any) -> tuple[bool, str]:
    for field in ("reasoning", "question", "answer"):
        value = getattr(pred, field, None)
        if not isinstance(value, str) or not value.strip():
            return False, f"Missing or empty field: {field}"
    return True, ""


def lm_call(lm: dspy.LM, prompt: str) -> str:
    response = lm(prompt)
    if isinstance(response, list):
        return str(response[0]) if response else ""
    return str(response)


def parse_judge_scores(response: str) -> tuple[str, int, int, int]:
    reasoning_match = re.search(r"REASONING:\s*(.*?)(?=CONSTRAINT_COMPLIANCE:|$)", response, re.DOTALL)
    constraint_match = re.search(r"CONSTRAINT_COMPLIANCE:\s*([1-5])", response)
    quality_match = re.search(r"SUBSTANTIVE_QUALITY:\s*([1-5])", response)
    accuracy_match = re.search(r"FACTUAL_ACCURACY:\s*([1-5])", response)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Judge response could not be parsed."
    constraint = int(constraint_match.group(1)) if constraint_match else 1
    quality = int(quality_match.group(1)) if quality_match else 1
    accuracy = int(accuracy_match.group(1)) if accuracy_match else 1
    return reasoning, constraint, quality, accuracy


def create_metric_fn(evaluation_lm: dspy.LM, max_chars_per_doc: int) -> Callable[..., dspy.Prediction]:
    """Return a GEPA-compatible metric that includes actionable feedback."""

    def metric_fn(example: Any, pred: Any, trace: Any = None, pred_name: Any = None, pred_trace: Any = None) -> dspy.Prediction:
        del trace, pred_name, pred_trace

        valid, error = validate_output_format(pred)
        if not valid:
            return dspy.Prediction(
                score=0.0,
                feedback=f"FORMAT ERROR: {error}. Return reasoning, question, and answer as non-empty strings.",
            )

        reasoning = pred.reasoning.strip()
        question = pred.question.strip()
        answer = pred.answer.strip()
        combined = "\n".join([reasoning, question, answer])

        if answer.lower() in {"yes", "no"}:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    "HARD CONSTRAINT VIOLATION: one-word yes/no answer. "
                    "The answer must explain substantive legal requirements or procedures."
                ),
            )

        if has_pattern(combined, BILL_REFERENCE_PATTERNS):
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    "HARD CONSTRAINT VIOLATION: detected document or bill references. "
                    "Avoid Text1/Text2, earlier/later document, and regulation/directive identifiers. "
                    "Ask about substantive legal content without naming the source documents."
                ),
            )

        if has_pattern(combined, RELATIONSHIP_LANGUAGE_PATTERNS):
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    "HARD CONSTRAINT VIOLATION: detected relationship/change language. "
                    "Do not ask what changed, whether one act repeals another, or when provisions cease. "
                    "Ask about requirements, procedures, definitions, criteria, obligations, or rights."
                ),
            )

        relation_types = getattr(example, "relation_types", "")
        if isinstance(relation_types, list):
            relation_types = ", ".join(str(item) for item in relation_types)

        eval_input = f"""{JUDGE_PROMPT}

DOCUMENT 1:
{str(example.text1)[:max_chars_per_doc]}

DOCUMENT 2:
{str(example.text2)[:max_chars_per_doc]}

RELATION TYPE: {relation_types}

GENERATED REASONING:
{reasoning}

GENERATED QUESTION:
{question}

GENERATED ANSWER:
{answer}

Evaluate this Q&A pair."""

        try:
            with dspy.settings.context(lm=evaluation_lm):
                judge_response = lm_call(evaluation_lm, eval_input)
            judge_reasoning, constraint_score, quality_score, accuracy_score = parse_judge_scores(judge_response)
        except Exception as exc:
            return dspy.Prediction(score=0.0, feedback=f"Judge evaluation failed: {exc}")

        score = (constraint_score + quality_score + accuracy_score) / 15.0
        feedback_parts = [
            f"JUDGE: {judge_reasoning}",
            f"SCORES: CONSTRAINT_COMPLIANCE={constraint_score}/5, SUBSTANTIVE_QUALITY={quality_score}/5, FACTUAL_ACCURACY={accuracy_score}/5",
        ]
        if constraint_score <= 3:
            feedback_parts.append(
                "Improve constraint compliance: remove relationship/change framing, document labels, regulation identifiers, and yes/no wording."
            )
        if quality_score <= 3:
            feedback_parts.append(
                "Improve substantive quality: make the question require both documents and include concrete legal-domain context."
            )
        if accuracy_score <= 3:
            feedback_parts.append("Improve factual accuracy: only state claims directly supported by the two documents.")

        feedback_parts.append(f"Generated question: {question[:300]}")
        return dspy.Prediction(score=score, feedback="\n".join(feedback_parts))

    return metric_fn


def normalize_relation(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def load_examples(args: argparse.Namespace) -> tuple[list[dspy.Example], list[dspy.Example]]:
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    rows = list(dataset)
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    required_count = args.train_samples + args.val_samples
    if len(rows) < required_count:
        raise ValueError(f"Dataset split has {len(rows)} rows, but {required_count} are requested.")

    def make_example(row: dict[str, Any]) -> dspy.Example:
        try:
            payload = {
                "text1": str(row[args.text1_field]),
                "text2": str(row[args.text2_field]),
                "relation_types": normalize_relation(row.get(args.relation_field, "")),
            }
        except KeyError as exc:
            available = ", ".join(sorted(row.keys()))
            raise KeyError(f"Missing dataset field {exc}. Available fields: {available}") from exc
        return dspy.Example(**payload).with_inputs("text1", "text2", "relation_types")

    train_rows = rows[: args.train_samples]
    val_rows = rows[args.train_samples : required_count]
    return [make_example(row) for row in train_rows], [make_example(row) for row in val_rows]


def make_lm(model: str, api_base: str, max_tokens: int, temperature: float) -> dspy.LM:
    return dspy.LM(
        model=f"openai/{model}" if not model.startswith("openai/") else model,
        api_base=api_base,
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def configure_dspy(lm: dspy.LM) -> None:
    if hasattr(dspy, "configure"):
        dspy.configure(lm=lm)
    else:
        dspy.settings.configure(lm=lm)


def get_evaluator_class() -> Any:
    if hasattr(dspy, "Evaluate"):
        return dspy.Evaluate
    from dspy.evaluate import Evaluate

    return Evaluate


def get_gepa_class() -> Any:
    if hasattr(dspy, "GEPA"):
        return dspy.GEPA
    try:
        from dspy.teleprompt import GEPA

        return GEPA
    except ImportError as exc:
        raise ImportError("GEPA is not available. Install/upgrade dspy and gepa from requirements.txt.") from exc


def evaluation_to_float(result: Any) -> float:
    value = float(result)
    return value / 100.0 if value > 1.0 else value


def extract_instructions(program: Any) -> str:
    candidates = [
        ("generate", "predict", "signature"),
        ("generate", "signature"),
        ("predict", "signature"),
        ("signature",),
    ]
    for path in candidates:
        obj = program
        try:
            for attr in path:
                obj = getattr(obj, attr)
            instructions = getattr(obj, "instructions", None)
            if instructions:
                return str(instructions)
        except AttributeError:
            continue
    return "Could not extract optimized instructions from this DSPy version."


def append_stats(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(stats)


def save_generated_qa(path: Path, program: Any, examples: list[dspy.Example]) -> None:
    with path.open("w") as handle:
        for index, example in enumerate(examples):
            try:
                pred = program(text1=example.text1, text2=example.text2, relation_types=example.relation_types)
                row = {
                    "example_id": index,
                    "relation_types": example.relation_types,
                    "reasoning": getattr(pred, "reasoning", ""),
                    "question": getattr(pred, "question", ""),
                    "answer": getattr(pred, "answer", ""),
                }
            except Exception as exc:
                row = {"example_id": index, "relation_types": example.relation_types, "error": str(exc)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone GEPA optimization for legal Q&A prompt generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name", default="disi-unibo-nlp/eurlex_relations")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--text1_field", default="text1")
    parser.add_argument("--text2_field", default="text2")
    parser.add_argument("--relation_field", default="relation_types")
    parser.add_argument("--train_samples", type=int, default=50)
    parser.add_argument("--val_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--task_model", required=True)
    parser.add_argument("--eval_model", required=True)
    parser.add_argument("--reflection_model", default=None)
    parser.add_argument("--vllm_url", default="http://127.0.0.1")
    parser.add_argument("--task_port", type=int, default=8000)
    parser.add_argument("--eval_port", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--eval_temperature", type=float, default=0.3)
    parser.add_argument("--eval_max_tokens", type=int, default=2048)

    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--gepa_auto", choices=["light", "medium", "heavy"])
    budget.add_argument("--max_full_evals", type=int)
    budget.add_argument("--max_metric_calls", type=int)

    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--max_chars_per_doc", type=int, default=2000)
    parser.add_argument("--output_dir", default="results/gepa_optimization")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="gepa-legal-qa")
    parser.add_argument("--wandb_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reflection_model is None:
        args.reflection_model = args.eval_model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("STANDALONE GEPA OPTIMIZATION FOR LEGAL Q&A")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name} [{args.dataset_split}]")
    print(f"Train/val samples: {args.train_samples}/{args.val_samples}")
    print(f"Task model: {args.task_model}")
    print(f"Eval model: {args.eval_model}")
    print(f"Reflection model: {args.reflection_model}")
    print(f"Output: {output_dir}")

    trainset, valset = load_examples(args)
    print(f"Loaded {len(trainset)} train examples and {len(valset)} validation examples.")

    task_api_base = f"{args.vllm_url}:{args.task_port}/v1"
    eval_api_base = f"{args.vllm_url}:{args.eval_port}/v1"
    task_lm = make_lm(args.task_model, task_api_base, args.max_tokens, args.temperature)
    eval_lm = make_lm(args.eval_model, eval_api_base, args.eval_max_tokens, args.eval_temperature)
    reflection_lm = make_lm(args.reflection_model, eval_api_base, args.eval_max_tokens, args.eval_temperature)
    configure_dspy(task_lm)

    metric_fn = create_metric_fn(eval_lm, max_chars_per_doc=args.max_chars_per_doc)
    seed_program = UnifiedQAGenerationModule()
    Evaluate = get_evaluator_class()

    print("Evaluating seed program...")
    evaluator = Evaluate(
        devset=valset,
        metric=lambda example, pred, trace=None: metric_fn(example, pred, trace).score,
        num_threads=args.num_threads,
        display_progress=True,
    )
    seed_score = evaluation_to_float(evaluator(seed_program))
    print(f"Seed score: {seed_score:.2%}")

    GEPA = get_gepa_class()
    gepa_kwargs: dict[str, Any] = {
        "metric": metric_fn,
        "num_threads": args.num_threads,
        "track_stats": True,
        "track_best_outputs": False,
        "reflection_lm": reflection_lm,
    }
    if args.gepa_auto:
        gepa_kwargs["auto"] = args.gepa_auto
    elif args.max_full_evals is not None:
        gepa_kwargs["max_full_evals"] = args.max_full_evals
    elif args.max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls
    if args.log_dir:
        gepa_kwargs["log_dir"] = args.log_dir
    if args.use_wandb:
        gepa_kwargs["use_wandb"] = True
        gepa_kwargs["wandb_init_kwargs"] = {
            "project": args.wandb_project,
            "name": args.wandb_name or f"gepa_legal_qa_{timestamp}",
        }

    print("Running GEPA optimization...")
    optimizer = GEPA(**gepa_kwargs)
    optimized_program = optimizer.compile(seed_program, trainset=trainset, valset=valset)

    print("Evaluating optimized program...")
    optimized_score = evaluation_to_float(evaluator(optimized_program))
    instructions = extract_instructions(optimized_program)

    stats = {
        "timestamp": timestamp,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "seed": args.seed,
        "task_model": args.task_model,
        "eval_model": args.eval_model,
        "reflection_model": args.reflection_model,
        "seed_score": seed_score,
        "optimized_score": optimized_score,
        "improvement": optimized_score - seed_score,
    }

    serialized_path = output_dir / f"optimized_program_{timestamp}.json"
    try:
        optimized_program.save(serialized_path)
        print(f"Saved serialized program: {serialized_path}")
    except Exception as exc:
        print(f"Warning: could not serialize optimized program with DSPy save(): {exc}")

    instructions_path = output_dir / f"optimized_program_{timestamp}.txt"
    instructions_path.write_text(
        "# Optimized Instructions\n\n"
        f"{instructions}\n\n"
        "# Seed Instructions\n\n"
        f"{SEED_INSTRUCTIONS}\n",
        encoding="utf-8",
    )

    results_path = output_dir / f"optimization_results_{timestamp}.json"
    results_path.write_text(
        json.dumps({"stats": stats, "optimized_instructions": instructions}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stats_path = output_dir / "gepa_optimization_stats.csv"
    append_stats(stats_path, stats)

    generated_path = output_dir / f"generated_qa_{timestamp}.jsonl"
    save_generated_qa(generated_path, optimized_program, valset)

    print("=" * 80)
    print("GEPA optimization complete")
    print(f"Seed score: {seed_score:.2%}")
    print(f"Optimized score: {optimized_score:.2%}")
    print(f"Improvement: {optimized_score - seed_score:.2%}")
    print(f"Instructions: {instructions_path}")
    print(f"Results: {results_path}")
    print(f"Stats: {stats_path}")
    print(f"Generated QA: {generated_path}")


if __name__ == "__main__":
    main()

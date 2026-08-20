from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    source: str
    split: str
    default_samples: int
    answer_type: str
    loader: Callable[[int | None], list[dict[str, Any]]]


def _hf() -> Any:
    import datasets

    return datasets


def _limit(rows: list[dict[str, Any]], max_items: int | None) -> list[dict[str, Any]]:
    return rows if max_items is None else rows[:max_items]


def _normalize_math_rows(
    rows: Any,
    *,
    dataset: str,
    question_fields: tuple[str, ...] = ("problem", "question"),
    answer_fields: tuple[str, ...] = ("answer", "final_answer"),
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = next((row.get(field) for field in question_fields if row.get(field)), "")
        answer = next((row.get(field) for field in answer_fields if row.get(field) is not None), "")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not str(question).strip():
            continue
        source_id = row.get("unique_id", row.get("id", index))
        normalized.append(
            {
                "problem_id": f"{dataset}_{source_id}",
                "prompt": str(question).strip(),
                "gold_answer": str(answer).strip().strip("$"),
                "metadata": {
                    "source_id": str(source_id),
                    "subject": row.get("subject", row.get("type")),
                    "level": row.get("level"),
                    "url": row.get("url"),
                },
            }
        )
    return normalized


def load_math500(max_items: int | None = None) -> list[dict[str, Any]]:
    rows = _hf().load_dataset("HuggingFaceH4/MATH-500", split="test")
    return _limit(_normalize_math_rows(rows, dataset="math500"), max_items)


def _load_math_ai(name: str, max_items: int | None) -> list[dict[str, Any]]:
    rows = _hf().load_dataset(f"math-ai/{name}", split="test")
    return _limit(_normalize_math_rows(rows, dataset=name), max_items)


def load_aime24(max_items: int | None = None) -> list[dict[str, Any]]:
    return _load_math_ai("aime24", max_items)


def load_aime25(max_items: int | None = None) -> list[dict[str, Any]]:
    return _load_math_ai("aime25", max_items)


def load_amc23(max_items: int | None = None) -> list[dict[str, Any]]:
    return _load_math_ai("amc23", max_items)


def load_olympiad(max_items: int | None = None) -> list[dict[str, Any]]:
    rows = _hf().load_dataset(
        "Hothan/OlympiadBench",
        "OE_TO_maths_en_COMP",
        split="train",
    )
    return _limit(_normalize_math_rows(rows, dataset="olympiad"), max_items)


def load_minerva(max_items: int | None = None) -> list[dict[str, Any]]:
    rows = _hf().load_dataset("math-ai/minervamath", split="test")
    return _limit(_normalize_math_rows(rows, dataset="minerva"), max_items)


def load_gpqa_diamond(max_items: int | None = None) -> list[dict[str, Any]]:
    import pandas as pd

    url = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
    rows = pd.read_csv(url).to_dict(orient="records")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        choices = [
            str(row["Correct Answer"]),
            str(row["Incorrect Answer 1"]),
            str(row["Incorrect Answer 2"]),
            str(row["Incorrect Answer 3"]),
        ]
        permutation = random.Random(f"{index}:0").sample(range(4), 4)
        options = [choices[position] for position in permutation]
        correct_index = options.index(str(row["Correct Answer"]))
        record_id = str(row.get("Record ID") or index)
        normalized.append(
            {
                "problem_id": f"gpqa_diamond_{record_id}",
                "prompt": str(row["Question"]).strip(),
                "gold_answer": "ABCD"[correct_index],
                "options": options,
                "_canonical_options": choices,
                "metadata": {
                    "source_id": record_id,
                    "domain": row.get("High-level domain"),
                    "subdomain": row.get("Subdomain"),
                    "option_permutation": permutation,
                },
            }
        )
    return _limit(normalized, max_items)


def sample_variant(
    dataset: str,
    example: dict[str, Any],
    sample_id: int,
) -> dict[str, Any]:
    """Return the exact per-attempt benchmark prompt and gold answer."""
    variant = dict(example)
    variant["metadata"] = dict(example.get("metadata") or {})
    if dataset != "gpqa_diamond":
        return variant
    canonical_options = list(example["_canonical_options"])
    permutation = random.Random(f"{example['problem_id']}:{sample_id}").sample(range(4), 4)
    options = [canonical_options[position] for position in permutation]
    variant["options"] = options
    variant["gold_answer"] = "ABCD"[options.index(canonical_options[0])]
    variant["metadata"]["option_permutation"] = permutation
    variant.pop("_canonical_options", None)
    return variant


def load_mmlu_pro(max_items: int | None = None) -> list[dict[str, Any]]:
    rows = _hf().load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question_id = str(row.get("question_id", index))
        normalized.append(
            {
                "problem_id": f"mmlu_pro_{question_id}",
                "prompt": str(row["question"]).strip(),
                "gold_answer": str(row["answer"]).strip().upper(),
                "options": [str(option) for option in row["options"]],
                "metadata": {
                    "source_id": question_id,
                    "category": row.get("category"),
                    "source": row.get("src"),
                },
            }
        )
    return _limit(normalized, max_items)


def _load_existing(name: str, max_items: int | None) -> list[dict[str, Any]]:
    from moe_exp.datasets.loaders import load_dataset_by_name

    return load_dataset_by_name(name, max_items=max_items)


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "math500": BenchmarkSpec("math500", "HuggingFaceH4/MATH-500", "test", 1, "math", load_math500),
    "aime24": BenchmarkSpec("aime24", "math-ai/aime24", "test", 32, "math", load_aime24),
    "aime25": BenchmarkSpec("aime25", "math-ai/aime25", "test", 32, "math", load_aime25),
    "olympiad": BenchmarkSpec(
        "olympiad", "Hothan/OlympiadBench:OE_TO_maths_en_COMP", "train", 1, "math", load_olympiad
    ),
    "amc23": BenchmarkSpec("amc23", "math-ai/amc23", "test", 32, "math", load_amc23),
    "minerva": BenchmarkSpec("minerva", "math-ai/minervamath", "test", 1, "math", load_minerva),
    "gpqa_diamond": BenchmarkSpec(
        "gpqa_diamond", "simple-evals/gpqa_diamond.csv", "all", 10, "choice", load_gpqa_diamond
    ),
    "mmlu_pro": BenchmarkSpec(
        "mmlu_pro", "TIGER-Lab/MMLU-Pro", "test", 1, "choice", load_mmlu_pro
    ),
    "gsm8k": BenchmarkSpec(
        "gsm8k", "openai/gsm8k:main", "test", 1, "math", lambda limit: _load_existing("gsm8k", limit)
    ),
    "math": BenchmarkSpec(
        "math", "EleutherAI/hendrycks_math", "test", 1, "math", lambda limit: _load_existing("math", limit)
    ),
    "prm800k": BenchmarkSpec(
        "prm800k", "tasksource/PRM800K", "train", 1, "math", lambda limit: _load_existing("prm800k", limit)
    ),
    "processbench": BenchmarkSpec(
        "processbench", "Qwen/ProcessBench", "test", 1, "unscored", lambda limit: _load_existing("processbench", limit)
    ),
}

DEFAULT_BENCHMARKS = tuple(BENCHMARKS)


def format_user_prompt(example: dict[str, Any]) -> str:
    options = example.get("options")
    if options:
        letters = "ABCDEFGHIJ"[: len(options)]
        option_lines = "\n".join(
            f"{letter}) {option}" for letter, option in zip(letters, options, strict=True)
        )
        return (
            "Please reason step by step, and put your final answer within \\boxed{}. "
            "Your final answer should be of the following format: \\boxed{LETTER} "
            f"where LETTER is one of {letters}.\n\n"
            f"{example['prompt']}\n\n{option_lines}"
        )
    return (
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        f"Question: {example['prompt']}"
    )


def generation_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": format_user_prompt(example)}]
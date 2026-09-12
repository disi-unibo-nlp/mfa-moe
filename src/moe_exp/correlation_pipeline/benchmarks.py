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


def _extract_boxed_answer(text: Any) -> str:
    """Extract the last balanced ``\\boxed{...}`` expression from a solution."""
    solution = str(text or "")
    marker = "\\boxed{"
    start = solution.rfind(marker)
    if start < 0:
        return ""
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(solution)):
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            depth -= 1
            if depth == 0:
                return solution[content_start:index].strip()
    return ""


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
        answer: Any = ""
        for field in answer_fields:
            candidate = row.get(field)
            if candidate is None:
                continue
            if isinstance(candidate, (str, list, tuple)) and not candidate:
                continue
            answer = candidate
            break
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not str(answer).strip():
            answer = _extract_boxed_answer(row.get("solution"))
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
    rows = list(
        _hf().load_dataset(
            "Hothan/OlympiadBench",
            "OE_TO_maths_en_COMP",
            split="train",
        )
    )
    # SPIRAL's frozen evaluator contains problem 1965, which is absent from the
    # current Hub conversion. Restore it so this remains the paper's 675-item set.
    if not any(str(row.get("id")) == "1965" for row in rows):
        insert_at = next(
            (index for index, row in enumerate(rows) if int(row.get("id", 0)) > 1965),
            len(rows),
        )
        rows.insert(insert_at, _SPIRAL_OLYMPIAD_1965)
    return _limit(_normalize_math_rows(rows, dataset="olympiad"), max_items)


_SPIRAL_OLYMPIAD_1965 = {
    "id": 1965,
    "subfield": "Number Theory",
    "question": r"""For every positive integer $n$ with prime factorization $n=\prod_{i=1}^{k} p_{i}^{\alpha_{i}}$, define

$$
\mho(n)=\sum_{i: p_{i}>10^{100}} \alpha_{i}\tag{1}
$$

That is, $\mho(n)$ is the number of prime factors of $n$ greater than $10^{100}$, counted with multiplicity.

Find all strictly increasing functions $f: \mathbb{Z} \rightarrow \mathbb{Z}$ such that

$$
\mho(f(a)-f(b)) \leqslant \mho(a-b) \quad \text { for all integers } a \text { and } b \text { with } a>b \text {. }
$$""",
    "final_answer": [
        (
            r"$f(x)=a x+b$, where $b$ is an arbitrary integer, and $a$ is an arbitrary "
            r"positive integer with $\mho(a)=0$"
        )
    ],
    "answer_type": "Expression",
}


def load_minerva(max_items: int | None = None) -> list[dict[str, Any]]:
    rows = _hf().load_dataset("math-ai/minervamath", split="test")
    return _limit(_normalize_math_rows(rows, dataset="minerva"), max_items)


def load_gpqa_diamond(max_items: int | None = None) -> list[dict[str, Any]]:
    import pandas as pd

    url = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
    rows = pd.read_csv(url).to_dict(orient="records")
    normalized: list[dict[str, Any]] = []
    source_size = len(rows)
    for index, row in enumerate(rows):
        choices = [
            str(row["Correct Answer"]),
            str(row["Incorrect Answer 1"]),
            str(row["Incorrect Answer 2"]),
            str(row["Incorrect Answer 3"]),
        ]
        permutation = _gpqa_permutation(index, source_size, sample_id=0)
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
                    "prompt_style": "gpqa",
                    "source_index": index,
                    "source_size": source_size,
                    "option_permutation": permutation,
                },
            }
        )
    return _limit(normalized, max_items)


def _gpqa_permutation(source_index: int, source_size: int, sample_id: int) -> list[int]:
    """Reproduce SPIRAL Simple Evals' seed-0 permutation stream."""
    rng = random.Random(0)
    target_position = sample_id * source_size + source_index
    permutation: list[int] = []
    for _ in range(target_position + 1):
        permutation = rng.sample(range(4), 4)
    return permutation


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
    metadata = variant["metadata"]
    source_index = metadata.get("source_index")
    source_size = metadata.get("source_size")
    if source_index is None or source_size is None:
        # Retain deterministic behavior for hand-built/custom GPQA records.
        permutation = random.Random(f"{example['problem_id']}:{sample_id}").sample(range(4), 4)
    else:
        permutation = _gpqa_permutation(int(source_index), int(source_size), sample_id)
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

SPIRAL_BENCHMARKS = (
    "math500",
    "aime24",
    "aime25",
    "olympiad",
    "amc23",
    "minerva",
    "gpqa_diamond",
    "mmlu_pro",
)
DEFAULT_BENCHMARKS = tuple(
    name for name in SPIRAL_BENCHMARKS if name not in {"gpqa_diamond", "mmlu_pro"}
)


def format_user_prompt(example: dict[str, Any]) -> str:
    options = example.get("options")
    if options:
        letters = "ABCDEFGHIJ"[: len(options)]
        option_lines = "\n".join(
            f"{letter}) {option}" for letter, option in zip(letters, options, strict=True)
        )
        question = str(example["prompt"])
        if (example.get("metadata") or {}).get("prompt_style") == "gpqa":
            body = f"Question: {question}\n\nOptions:\n{option_lines}"
        else:
            body = f"{question}\n\n{option_lines}"
        return (
            "Please reason step by step, and put your final answer within \\boxed{}. "
            "Your final answer should be of the following format: \\boxed{LETTER} "
            f"where LETTER is one of {letters}.\n\n"
            f"{body}"
        )
    return (
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        f"Question: {example['prompt']}"
    )


def generation_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": format_user_prompt(example)}]

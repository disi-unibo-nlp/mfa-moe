from __future__ import annotations

import json
from typing import Optional

from huggingface_hub import hf_hub_download
from rich.console import Console

from moe_exp.utils import extract_gold_answer_gsm8k, extract_model_answer

console = Console()


def _hf() -> "datasets":  # type: ignore[name-defined]
    """Lazy import of the HuggingFace datasets library.

    Importing datasets at module level on Windows causes a native crash
    (STATUS_ACCESS_VIOLATION / 0xC0000005) when torch has already been
    loaded.  Deferring the import until the first dataset is actually
    requested avoids the DLL conflict.
    """
    import datasets  # noqa: PLC0415
    return datasets

AVAILABLE_DATASETS = ["gsm8k", "processbench", "prm800k", "math"]

# Datasets that ship a pre-written reasoning chain with gold step-level labels.
# For these we analyze the GIVEN solution (its steps + first_error_step) rather
# than generating a fresh CoT, so the first-error label aligns with the chain
# whose routing we extract. Examples expose: solution_steps, first_error_step,
# solution_is_correct.
GIVEN_SOLUTION_DATASETS = {"processbench", "prm800k"}


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def load_gsm8k(max_items: Optional[int] = None) -> list[dict]:
    console.print("[blue]Loading GSM8K (test split)…")
    ds = _hf().load_dataset("openai/gsm8k", "main", split="test")
    if max_items:
        ds = ds.select(range(min(max_items, len(ds))))
    results = []
    for i, ex in enumerate(ds):
        results.append(
            {
                "problem_id": f"gsm8k_{i}",
                "prompt": ex["question"],
                "gold_answer": extract_gold_answer_gsm8k(ex["answer"]),
                "metadata": {},
            }
        )
    return results


# ---------------------------------------------------------------------------
# ProcessBench
# ---------------------------------------------------------------------------


def load_processbench(max_items: Optional[int] = None) -> list[dict]:
    """Load Qwen/ProcessBench (test split) as a given-solution dataset.

    ProcessBench is a process-level evaluation benchmark. Each example contains
    a problem and a pre-written step-by-step solution (which may contain errors),
    with a gold label for the first erroneous step (-1 = all correct). We analyze
    the given solution chain directly rather than generating a fresh CoT.

    Fields populated:
      - prompt:               the problem text
      - gold_answer:          "" (the chain itself is analyzed; no final answer)
      - solution_steps:       the pre-written steps list
      - first_error_step:     gold first-error index into solution_steps
                              (None when all steps are correct)
      - solution_is_correct:  True when label == -1
      - metadata.label:       raw label (int; -1 = all correct)
      - metadata.source:      gsm8k / math / olympiad / omnimath
    """
    console.print("[blue]Loading ProcessBench…")
    try:
        ds = _hf().load_dataset("Qwen/ProcessBench", split="test")
    except Exception:
        # Some dataset versions store everything under a default split
        data = _hf().load_dataset("Qwen/ProcessBench")
        split = list(data.keys())[0]
        ds = data[split]

    if max_items:
        ds = ds.select(range(min(max_items, len(ds))))

    results = []
    for i, ex in enumerate(ds):
        label = ex.get("label")          # int: first-error step index, -1 = all correct
        raw_steps: list[str] = ex.get("steps") or []
        steps = [str(s).strip() for s in raw_steps if str(s).strip()]
        problem = ex.get("problem") or ex.get("question") or ""

        # Given-solution example: analyze the pre-written solution chain itself.
        # first_error_step indexes `steps` directly (the gold label), so routing
        # is aligned to the chain we extract logits from. -1 means all correct.
        if not steps:
            continue
        first_error = label if (label is not None and label >= 0 and label < len(steps)) else None
        results.append(
            {
                "problem_id": f"processbench_{i}",
                "prompt": problem,
                "gold_answer": "",
                "solution_steps": steps,
                "first_error_step": first_error,
                "solution_is_correct": (label == -1) if label is not None else None,
                "metadata": {
                    "label": label,
                    "source": ex.get("source"),
                },
            }
        )
    return results


# ---------------------------------------------------------------------------
# PRM800K
# ---------------------------------------------------------------------------


def _prm800k_build_example(rec: dict) -> tuple[str, str, list[str], Optional[int], Optional[bool]]:
    """Reconstruct a reasoning chain + first-error label from one PRM800K record.

    PRM800K rates candidate next-steps (rating -1 bad, 0 neutral, 1 good) at each
    position rather than storing a single errored solution. To get a chain that is
    analogous to ProcessBench (correct up to step k, then wrong), we follow the
    good/chosen path until the first step that offers a (-1)-rated candidate, then
    append that wrong candidate as the erroneous step. `first_error_step` indexes
    that wrong step within the returned `steps`. Records with no (-1) candidate
    yield a fully-correct chain (first_error None).

    Returns: (problem, gold_answer, steps, first_error_step, is_correct).
    """
    q = rec.get("question") or {}
    problem = q.get("problem") or q.get("text") or "" if isinstance(q, dict) else str(q)
    gold = ""
    if isinstance(q, dict):
        gold = q.get("ground_truth_answer") or q.get("ground_truth_solution") or ""
    steps_data = (rec.get("label") or {}).get("steps") or []

    good_prefix: list[str] = []
    error_text: Optional[str] = None
    first_error: Optional[int] = None
    for i, step in enumerate(steps_data):
        comps = step.get("completions") or []
        neg = next(
            (c for c in comps if c.get("rating") == -1 and (c.get("text") or "").strip()),
            None,
        )
        if neg is not None:
            error_text = neg["text"].strip()
            first_error = i
            break
        # Follow the good path: a human correction if present, else the chosen completion.
        human = step.get("human_completion")
        chosen = step.get("chosen_completion")
        if isinstance(human, dict):
            text = human.get("text")
        elif isinstance(human, str):
            text = human
        elif chosen is not None and 0 <= chosen < len(comps):
            text = comps[chosen].get("text")
        else:
            text = None  # give-up / end of solution
        if not text or not text.strip():
            break
        good_prefix.append(text.strip())

    if error_text is not None:
        return problem, str(gold), good_prefix + [error_text], first_error, False
    return problem, str(gold), good_prefix, None, (True if good_prefix else None)


def load_prm800k(max_items: Optional[int] = None) -> list[dict]:
    """Load PRM800K (tasksource/PRM800K on HuggingFace) as a given-solution dataset.

    Uses the public tasksource mirror of the OpenAI PRM800K dataset. The repository
    stores raw JSONL files with nested per-step ratings, so we read the records
    directly and reconstruct a labeled reasoning chain per problem via
    `_prm800k_build_example` (see its docstring). Each example exposes the chain
    `solution_steps`, the gold `first_error_step`, and `solution_is_correct`.
    Deduplicates by problem text so each unique problem appears once.
    """
    console.print("[blue]Loading PRM800K…")
    try:
        records = []
        for filename in ("phase1_train.jsonl", "phase2_train.jsonl"):
            path = hf_hub_download(
                "tasksource/PRM800K",
                filename,
                repo_type="dataset",
            )
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load PRM800K: {e}\n"
            "Check https://huggingface.co/datasets/tasksource/PRM800K"
        ) from e

    seen: set[str] = set()
    results = []
    for ex in records:
        problem, gold, steps, first_error, is_correct = _prm800k_build_example(ex)
        if not problem or not steps or problem in seen:
            continue
        seen.add(problem)

        results.append(
            {
                "problem_id": f"prm800k_{len(results)}",
                "prompt": problem,
                "gold_answer": str(gold),
                "solution_steps": steps,
                "first_error_step": first_error,
                "solution_is_correct": is_correct,
                "metadata": {"source": "prm800k"},
            }
        )
        if max_items and len(results) >= max_items:
            break

    return results


# ---------------------------------------------------------------------------
# MATH (Hendrycks et al.)
# ---------------------------------------------------------------------------


def load_math(max_items: Optional[int] = None) -> list[dict]:
    """Load the MATH benchmark (EleutherAI/hendrycks_math).

    Loads all subject subsets and combines them.
    Gold answer is taken from the 'solution' field by extracting \\boxed{}.
    """
    console.print("[blue]Loading MATH dataset…")
    datasets_lib = _hf()
    subsets = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    all_rows = []
    for subset in subsets:
        try:
            part = datasets_lib.load_dataset(
                "EleutherAI/hendrycks_math", subset, split="test"
            )
            all_rows.extend(part)
        except Exception:
            continue

    if not all_rows:
        raise RuntimeError(
            "Could not load MATH dataset from EleutherAI/hendrycks_math."
        )
    console.print(f"  Loaded [cyan]{len(all_rows)}[/] problems from EleutherAI/hendrycks_math")
    ds = all_rows

    if max_items:
        ds = ds[:max_items]

    results = []
    for i, ex in enumerate(ds):
        problem = ex.get("problem") or ex.get("question") or ""
        gold = ex.get("answer") or ex.get("gold_answer") or ""
        if not gold and ex.get("solution"):
            gold = extract_model_answer(ex["solution"])
        results.append(
            {
                "problem_id": f"math_{i}",
                "prompt": problem,
                "gold_answer": str(gold),
                "metadata": {
                    "level": ex.get("level"),
                    "type": ex.get("type") or ex.get("subject"),
                },
            }
        )
    return results


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def load_dataset_by_name(name: str, max_items: Optional[int] = None) -> list[dict]:
    _loaders = {
        "gsm8k": load_gsm8k,
        "processbench": load_processbench,
        "prm800k": load_prm800k,
        "math": load_math,
    }
    if name not in _loaders:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(_loaders)}"
        )
    return _loaders[name](max_items)

from __future__ import annotations

from typing import Optional

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


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def load_gsm8k(max_items: Optional[int] = None) -> list[dict]:
    console.print("[blue]Loading GSM8K (test split)…")
    ds = _hf().load_dataset("gsm8k", "main", split="test")
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
    """Load Qwen/ProcessBench (test split).

    ProcessBench is a process-level evaluation benchmark. Each example contains
    a problem and a pre-written step-by-step solution (which may contain errors).
    The task is to identify the index of the first erroneous step (-1 = all correct).

    Fields populated:
      - prompt:           problem text + numbered steps + evaluation instruction
      - gold_answer:      str(label) where label is the first-error step index
                          (0-indexed; -1 means all steps are correct)
      - metadata.label:   same value as gold_answer, as int
      - metadata.steps:   the original pre-written steps list
      - metadata.source:  gsm8k / math / olympiad / omnimath
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
        steps: list[str] = ex.get("steps") or []
        problem = ex.get("problem") or ex.get("question") or ""

        # Build evaluation prompt: show pre-written steps and ask for the first error
        if steps:
            steps_text = "\n".join(f"Step {j}: {s}" for j, s in enumerate(steps))
            prompt = (
                f"{problem}\n\n"
                f"The following is a step-by-step solution. "
                f"Identify the index of the first step that contains an error. "
                f"Steps are indexed from 0. "
                f"If all steps are correct, answer -1.\n\n"
                f"{steps_text}\n\n"
                f"What is the index of the first erroneous step? "
                f"Answer with a single integer (-1 if all steps are correct). "
                f"Final answer:"
            )
        else:
            prompt = problem

        results.append(
            {
                "problem_id": f"processbench_{i}",
                "prompt": prompt,
                "gold_answer": str(label) if label is not None else "",
                "metadata": {
                    "label": label,
                    "steps": steps,
                    "source": ex.get("source"),
                },
            }
        )
    return results


# ---------------------------------------------------------------------------
# PRM800K
# ---------------------------------------------------------------------------


def load_prm800k(max_items: Optional[int] = None) -> list[dict]:
    """Load PRM800K (tasksource/PRM800K on HuggingFace).

    Uses the public tasksource mirror of the OpenAI PRM800K dataset.
    Deduplicates by problem text so each unique problem appears once.
    """
    console.print("[blue]Loading PRM800K…")
    try:
        ds = _hf().load_dataset("tasksource/PRM800K", split="train")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load PRM800K: {e}\n"
            "Check https://huggingface.co/datasets/tasksource/PRM800K"
        ) from e

    seen: set[str] = set()
    results = []
    for ex in ds:
        q = ex.get("question") or {}
        if isinstance(q, dict):
            problem = q.get("problem") or q.get("text") or str(q)
            gold = q.get("ground_truth_answer") or ""
        else:
            problem = str(q)
            gold = ex.get("gold_answer") or ""

        if not problem or problem in seen:
            continue
        seen.add(problem)

        results.append(
            {
                "problem_id": f"prm800k_{len(results)}",
                "prompt": problem,
                "gold_answer": str(gold),
                "metadata": {},
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

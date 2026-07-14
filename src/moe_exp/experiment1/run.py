"""Experiment 1 — Basic reasoning and failure taxonomy.

Usage
-----
    python -m moe_exp.experiment1.run \\
        --model allenai/OLMoE-1B-7B-0924-Instruct \\
        --datasets gsm8k processbench \\
        --max-items 20 \\
        --output-dir results/exp1

    # Multiple models:
    python -m moe_exp.experiment1.run \\
        --model allenai/OLMoE-1B-7B-0924-Instruct \\
        --model Qwen/Qwen1.5-MoE-A2.7B-Chat \\
        --datasets gsm8k \\
        --max-items 50 \\
        --output-dir results/exp1

Output
------
    results/exp1/
        <model-slug>/<dataset>/traces.jsonl   — one TraceRecord per example
        summary.json                          — aggregate taxonomy table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from moe_exp.analysis.classifier import classify_trace
from moe_exp.analysis.step_splitter import split_steps
from moe_exp.datasets.loaders import (
    AVAILABLE_DATASETS,
    GIVEN_SOLUTION_DATASETS,
    load_dataset_by_name,
)
from moe_exp.experiment1.taxonomy import build_row, build_summary
from moe_exp.models.inference import generate_cot, SYSTEM_PROMPT_SELFCHECK
from moe_exp.models.loader import QUANTIZATION_CHOICES, load_model_and_tokenizer
from moe_exp.schemas import TraceRecord
from moe_exp.utils import answers_match, extract_model_answer, read_json, write_json, write_jsonl

console = Console()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="exp1",
        description="Experiment 1 — Basic reasoning and failure taxonomy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        metavar="HF_MODEL_ID",
        help=(
            "HuggingFace model ID to evaluate. "
            "Repeat the flag to run multiple models sequentially."
        ),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k"],
        choices=AVAILABLE_DATASETS,
        metavar="DATASET",
        help=f"Dataset(s) to run. Choices: {AVAILABLE_DATASETS}. Default: gsm8k.",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        metavar="N",
        help="Cap on examples per dataset (smoke-test). Default: use all.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp1"),
        metavar="DIR",
        help="Root output directory. Default: results/exp1.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="'cuda' uses device_map=auto; 'cpu' loads to CPU. Default: cuda.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per example. Default: 1024.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Inference batch size. For large MoE models keep at 1 "
            "to avoid OOM. Default: 1."
        ),
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True to from_pretrained (some models need this).",
    )
    p.add_argument(
        "--quantization",
        default="none",
        choices=QUANTIZATION_CHOICES,
        metavar="METHOD",
        help=(
            "Quantization method for model loading. "
            f"Choices: {QUANTIZATION_CHOICES}. Default: none (bf16)."
        ),
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        default=False,
        help=(
            "Run each dataset with the self-checking prompt INSTEAD of the normal "
            "prompt. This prompt encourages the model to verify each step and "
            "correct errors. Results are saved under a separate dataset name "
            "(e.g. gsm8k_selfcheck). To get both variants, run twice: once "
            "without this flag and once with it. Given-solution datasets "
            "(ProcessBench, PRM800K) are skipped under --self-check: their "
            "chain is pre-written, so there is nothing to generate."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_slug(model_id: str) -> str:
    """Convert a HF model ID to a safe directory name."""
    return model_id.replace("/", "--")


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> None:
    all_rows: list[dict] = []

    # Build the list of (dataset_name, output_dataset_name, system_prompt) runs.
    # If --self-check, run ONLY the self-checking prompt variant (stored as
    # "<dataset>_selfcheck"). Otherwise, run with the normal prompt.
    run_configs: list[tuple[str, str, str | None]] = []
    if args.self_check:
        for ds in args.datasets:
            # Self-check only applies to datasets we generate CoT for. Given-solution
            # datasets (ProcessBench/PRM800K) analyze a pre-written chain, so the
            # self-checking generation prompt does not apply.
            if ds in GIVEN_SOLUTION_DATASETS:
                continue
            run_configs.append((ds, f"{ds}_selfcheck", SYSTEM_PROMPT_SELFCHECK))
    else:
        for ds in args.datasets:
            run_configs.append((ds, ds, None))  # normal prompt

    if not run_configs:
        raise ValueError("No runnable dataset configuration was selected.")

    for model_id in args.models:
        console.rule(f"[bold cyan]Model: {model_id}")
        # Lazily loaded: given-solution datasets (ProcessBench/PRM800K) need no
        # generation, so a run over only those never loads the model weights.
        model = None
        tokenizer = None

        for source_dataset, output_name, sys_prompt in run_configs:
            console.rule(f"  [cyan]Dataset: {output_name}")

            examples = load_dataset_by_name(source_dataset, max_items=args.max_items)
            # Drop examples that have no problem text
            examples = [ex for ex in examples if ex.get("prompt")]
            console.print(f"  {len(examples)} examples loaded.")

            traces: list[TraceRecord] = []

            # --- Given-solution examples: analyze the pre-written chain directly ---
            given = [ex for ex in examples if ex.get("solution_steps")]
            for ex in given:
                steps = [s for s in ex["solution_steps"] if s.strip()]
                if not steps:
                    continue
                cot_text = "\n\n".join(steps)
                step_labels = classify_trace(steps, cot_text)
                # Gold first-error step indexes `steps` directly (ProcessBench label
                # / reconstructed PRM800K chain), so it aligns with this trace.
                fe = ex.get("first_error_step")
                if fe is not None and 0 <= fe < len(steps):
                    step_labels.first_error_step = fe
                traces.append(
                    TraceRecord(
                        dataset=output_name,
                        problem_id=ex["problem_id"],
                        prompt=ex["prompt"],
                        system_prompt=sys_prompt,
                        gold_answer=ex.get("gold_answer", ""),
                        model_id=model_id,
                        model_answer=extract_model_answer(cot_text),
                        is_correct=ex.get("solution_is_correct"),
                        cot_text=cot_text,
                        steps=steps,
                        step_labels=step_labels,
                        task_type="reasoning",
                    )
                )

            # --- Generation examples: generate the model's own CoT ---
            to_generate = [ex for ex in examples if not ex.get("solution_steps")]
            if to_generate:
                if model is None:
                    model, tokenizer = load_model_and_tokenizer(
                        model_id,
                        device=args.device,
                        trust_remote_code=args.trust_remote_code,
                        quantization=args.quantization,
                    )
                problems = [ex["prompt"] for ex in to_generate]
                generated = generate_cot(
                    model,
                    tokenizer,
                    problems,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    system_prompt=sys_prompt,
                )
                for ex, cot_text in zip(to_generate, generated):
                    model_answer = extract_model_answer(cot_text)
                    is_correct = answers_match(model_answer, ex["gold_answer"])
                    steps = split_steps(cot_text)
                    step_labels = classify_trace(steps, cot_text)
                    traces.append(
                        TraceRecord(
                            dataset=output_name,
                            problem_id=ex["problem_id"],
                            prompt=ex["prompt"],
                            system_prompt=sys_prompt,
                            gold_answer=ex["gold_answer"],
                            model_id=model_id,
                            model_answer=model_answer,
                            is_correct=is_correct,
                            cot_text=cot_text,
                            steps=steps,
                            step_labels=step_labels,
                            task_type="reasoning",
                        )
                    )

            if not traces:
                raise RuntimeError(
                    f"Dataset '{source_dataset}' produced zero valid traces; "
                    "refusing to create an empty Experiment 1 output."
                )

            slug = _model_slug(model_id)
            trace_path = args.output_dir / slug / output_name / "traces.jsonl"
            write_jsonl(traces, trace_path)
            console.print(f"  Traces → [green]{trace_path}")

            row = build_row(model_id, output_name, traces)
            all_rows.append(row)
            console.print(
                f"  Correct traces: [bold]{row.get('pct_correct_traces')}[/]  "
                f"Backtracking: {row.get('pct_backtracking')}  "
                f"Contradiction: {row.get('pct_contradiction')}"
            )

        # Free model memory before loading the next one
        if model is not None:
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Write summary — merge with existing summary to preserve previous runs
    summary_path = args.output_dir / "summary.json"
    existing = read_json(summary_path)
    if existing and "rows" in existing:
        # Index old rows by (model, dataset) key
        merged = {(r["model"], r["dataset"]): r for r in existing["rows"]}
    else:
        merged = {}
    # New rows override old ones with matching keys
    for row in all_rows:
        merged[(row["model"], row["dataset"])] = row
    summary = build_summary(list(merged.values()))
    write_json(summary, summary_path)
    console.print(f"\n[bold green]Summary → {summary_path}")

    # Pretty-print the table to the terminal
    _print_table(summary)


def _print_table(summary: dict) -> None:
    cols = summary["columns"]
    table = Table(title="Experiment 1 — Failure Taxonomy", show_lines=True)
    for col in cols:
        table.add_column(col, overflow="fold")
    for row in summary["rows"]:
        table.add_row(*[str(row.get(c, "")) for c in cols])
    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()

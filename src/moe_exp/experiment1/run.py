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
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from moe_exp.analysis.classifier import classify_trace
from moe_exp.analysis.step_splitter import split_steps
from moe_exp.datasets.loaders import AVAILABLE_DATASETS, load_dataset_by_name
from moe_exp.experiment1.taxonomy import build_row, build_summary
from moe_exp.models.inference import generate_cot
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

    for model_id in args.models:
        console.rule(f"[bold cyan]Model: {model_id}")
        model, tokenizer = load_model_and_tokenizer(
            model_id,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            quantization=args.quantization,
        )

        for dataset_name in args.datasets:
            console.rule(f"  [cyan]Dataset: {dataset_name}")

            examples = load_dataset_by_name(dataset_name, max_items=args.max_items)
            # Drop examples that have no problem text
            examples = [ex for ex in examples if ex.get("prompt")]
            console.print(f"  {len(examples)} examples loaded.")

            problems = [ex["prompt"] for ex in examples]
            generated = generate_cot(
                model,
                tokenizer,
                problems,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
            )

            traces: list[TraceRecord] = []
            for ex, cot_text in zip(examples, generated):
                model_answer = extract_model_answer(cot_text)
                is_correct = answers_match(model_answer, ex["gold_answer"])
                steps = split_steps(cot_text)
                step_labels = classify_trace(steps, cot_text)

                # For ProcessBench, first_error_step comes from the gold metadata label
                if dataset_name == "processbench":
                    label = ex.get("metadata", {}).get("label")
                    if label is not None:
                        step_labels.first_error_step = label

                traces.append(
                    TraceRecord(
                        dataset=dataset_name,
                        problem_id=ex["problem_id"],
                        prompt=ex["prompt"],
                        gold_answer=ex["gold_answer"],
                        model_id=model_id,
                        model_answer=model_answer,
                        is_correct=is_correct,
                        cot_text=cot_text,
                        steps=steps,
                        step_labels=step_labels,
                    )
                )

            slug = _model_slug(model_id)
            trace_path = args.output_dir / slug / dataset_name / "traces.jsonl"
            write_jsonl(traces, trace_path)
            console.print(f"  Traces → [green]{trace_path}")

            row = build_row(model_id, dataset_name, traces)
            all_rows.append(row)
            console.print(
                f"  Accuracy: [bold]{row.get('accuracy')}[/]  "
                f"Backtracking: {row.get('pct_backtracking')}  "
                f"Contradiction: {row.get('pct_contradiction')}"
            )

        # Free model memory before loading the next one
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

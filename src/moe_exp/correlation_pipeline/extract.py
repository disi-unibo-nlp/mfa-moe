from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch

from moe_exp.correlation_pipeline.benchmarks import BENCHMARKS, DEFAULT_BENCHMARKS
from moe_exp.correlation_pipeline.defaults import DEFAULT_FORWARD_MODEL, DEFAULT_GENERATION_MODEL
from moe_exp.experiment2.run import process_file
from moe_exp.models.loader import QUANTIZATION_CHOICES, load_model_and_tokenizer

logger = logging.getLogger(__name__)
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _model_slug(model: str) -> str:
    return _SAFE_FILENAME.sub("--", model).strip("-")


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def extract_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    generation_root = args.generation_dir / _model_slug(args.generation_model)
    output_root = args.output_dir / _model_slug(args.model_id)
    inputs: list[tuple[str, Path, Path]] = []
    for dataset in args.datasets:
        input_path = generation_root / dataset / "traces.jsonl"
        if not input_path.is_file() or input_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing generated traces for {dataset}: {input_path}. "
                "Run the correlation generation stage first."
            )
        manifest_path = generation_root / dataset / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing generation manifest for {dataset}: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generated_target = manifest.get("target_model_id")
        if generated_target != args.model_id:
            raise ValueError(
                f"Generated {dataset} traces target {generated_target!r}, but forward extraction "
                f"requested {args.model_id!r}. Use the same target model."
            )
        output_path = output_root / dataset / "traces_with_routing.jsonl"
        inputs.append((dataset, input_path, output_path))

    logger.info("Loading forward-pass checkpoint %s once for %d datasets", args.model_id, len(inputs))
    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        quantization=args.quantization,
    )
    summaries: list[dict[str, Any]] = []
    try:
        for dataset, input_path, output_path in inputs:
            logger.info("Extracting %s", dataset)
            process_file(
                input_path=input_path,
                model_id=args.model_id,
                output_path=output_path,
                limit=args.limit,
                top_k=args.top_k,
                extract_hidden_states=not args.router_only,
                quantization=args.quantization,
                model=model,
                tokenizer=tokenizer,
                strict_top_k=True,
            )
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError(f"Forward extraction produced no output for {dataset}")
            trace_count = sum(
                1
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            summaries.append(
                {
                    "dataset": dataset,
                    "status": "complete",
                    "traces": trace_count,
                    "input": input_path.as_posix(),
                    "output": output_path.as_posix(),
                }
            )
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "status": "complete",
        "forward_model": args.model_id,
        "generation_model": args.generation_model,
        "quantization": args.quantization,
        "router_only": args.router_only,
        "datasets": summaries,
    }
    _write_json_atomic(summary, output_root / "summary.json")
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Teacher-force llama.cpp traces through a Hugging Face MoE checkpoint"
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_FORWARD_MODEL,
        help="Hugging Face target checkpoint matching the GGUF",
    )
    parser.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--datasets", nargs="+", choices=tuple(BENCHMARKS), default=DEFAULT_BENCHMARKS)
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=Path("results/correlation_pipeline/generation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/correlation_pipeline/forward"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit traces per dataset")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_CHOICES,
        default="unsloth-4bit",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--router-only",
        action="store_true",
        help="Skip hidden-state tensors; this disables hidden/router geometry correlations.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    extract_all(args)


if __name__ == "__main__":
    main()
"""CLI for gold boundary extraction and paper-matched episode probes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from moe_exp.probeTest.extract import (
    DEFAULT_MODEL_ID,
    DEFAULT_QUANTIZATION,
    extract_gold_corpus,
)
from moe_exp.probeTest.label import BENCHMARK_DATASETS, label_benchmark_examples
from moe_exp.probeTest.probe import train_layerwise_probes


def _add_extraction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument(
        "--quantization",
        choices=("gptq-4bit", "none", "bnb-4bit", "bnb-8bit"),
        default=DEFAULT_QUANTIZATION,
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message. The default uses only the released SAT instruction.",
    )
    parser.add_argument(
        "--include-think-boundary-units",
        action="store_true",
        help="Include the 38 released </think>/Final Answer units omitted by the paper count.",
    )
    parser.add_argument("--max-documents", type=int, default=None)


def _add_probe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--skip-plot", action="store_true")


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark-datasets",
        nargs="+",
        choices=BENCHMARK_DATASETS,
        default=list(BENCHMARK_DATASETS),
    )
    parser.add_argument("--examples-per-benchmark", type=int, default=20)
    parser.add_argument(
        "--max-benchmark-input-tokens",
        type=int,
        default=4096,
        help="Truncate very long traces after this many prompt+reasoning tokens.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="Forward the gold traces and save boundary activations"
    )
    _add_extraction_arguments(extract_parser)
    extract_parser.add_argument("--output-dir", type=Path, required=True)

    probe_parser = subparsers.add_parser(
        "probe", help="Train layer-wise probes from a completed activation manifest"
    )
    probe_parser.add_argument("--manifest", type=Path, required=True)
    probe_parser.add_argument("--output-dir", type=Path, required=True)
    _add_probe_arguments(probe_parser)

    label_parser = subparsers.add_parser(
        "label", help="Apply trained probes to benchmark reasoning for manual inspection"
    )
    label_parser.add_argument("--probe-results", type=Path, required=True)
    label_parser.add_argument("--output-dir", type=Path, required=True)
    label_parser.add_argument("--trust-remote-code", action="store_true")
    _add_benchmark_arguments(label_parser)

    all_parser = subparsers.add_parser(
        "all", help="Run extraction, train probes, then label benchmark examples"
    )
    _add_extraction_arguments(all_parser)
    _add_probe_arguments(all_parser)
    _add_benchmark_arguments(all_parser)
    all_parser.add_argument("--skip-benchmark-labeling", action="store_true")
    all_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run root; writes activations/ and probes/ below this directory",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "extract":
        manifest = extract_gold_corpus(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            model_id=args.model,
            revision=args.model_revision,
            quantization=args.quantization,
            trust_remote_code=args.trust_remote_code,
            system_prompt=args.system_prompt,
            include_think_boundary_units=args.include_think_boundary_units,
            max_documents=args.max_documents,
        )
        print(f"Activation manifest: {manifest}")
        return

    if args.command == "probe":
        results = train_layerwise_probes(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            test_size=args.test_size,
            seed=args.seed,
            max_iter=args.max_iter,
            make_plot=not args.skip_plot,
        )
        print(f"Probe results: {results}")
        return

    if args.command == "label":
        manifest = label_benchmark_examples(
            probe_results_path=args.probe_results,
            output_dir=args.output_dir,
            datasets=args.benchmark_datasets,
            examples_per_dataset=args.examples_per_benchmark,
            max_input_tokens=args.max_benchmark_input_tokens,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"Benchmark-label manifest: {manifest}")
        return

    activation_dir = args.output_dir / "activations"
    probe_dir = args.output_dir / "probes"
    manifest = extract_gold_corpus(
        dataset_dir=args.dataset_dir,
        output_dir=activation_dir,
        model_id=args.model,
        revision=args.model_revision,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
        system_prompt=args.system_prompt,
        include_think_boundary_units=args.include_think_boundary_units,
        max_documents=args.max_documents,
    )
    results = train_layerwise_probes(
        manifest_path=manifest,
        output_dir=probe_dir,
        test_size=args.test_size,
        seed=args.seed,
        max_iter=args.max_iter,
        make_plot=not args.skip_plot,
    )
    benchmark_manifest = None
    if not args.skip_benchmark_labeling:
        benchmark_manifest = label_benchmark_examples(
            probe_results_path=results,
            output_dir=args.output_dir / "benchmark_labels",
            datasets=args.benchmark_datasets,
            examples_per_dataset=args.examples_per_benchmark,
            max_input_tokens=args.max_benchmark_input_tokens,
            trust_remote_code=args.trust_remote_code,
        )
    print(f"Activation manifest: {manifest}")
    print(f"Probe results: {results}")
    if benchmark_manifest is not None:
        print(f"Benchmark-label manifest: {benchmark_manifest}")


if __name__ == "__main__":
    main()

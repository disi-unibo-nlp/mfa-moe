"""CLI for gold boundary extraction and paper-matched episode probes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from moe_exp.probeTest.extract import DEFAULT_MODEL_ID, extract_gold_corpus
from moe_exp.probeTest.probe import train_layerwise_probes


def _add_extraction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument(
        "--quantization",
        choices=("none", "bnb-4bit", "bnb-8bit"),
        default="bnb-4bit",
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

    all_parser = subparsers.add_parser("all", help="Run extraction and then all probes")
    _add_extraction_arguments(all_parser)
    _add_probe_arguments(all_parser)
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
    print(f"Activation manifest: {manifest}")
    print(f"Probe results: {results}")


if __name__ == "__main__":
    main()

"""MTP compatibility probe: one real batch of 16 against a running vLLM server."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from moe_exp.correlation_pipeline.batch_predictor import (
    classify_batch,
    load_program_and_adapter,
)
from moe_exp.correlation_pipeline.batch_probe import build_probe_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--judge-program", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:41800/v1")
    parser.add_argument("--api-key", default="local-vllm-key")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--reasoning-effort", default="medium")
    args = parser.parse_args()

    items = build_probe_items(args.traces, args.batch_size)
    _, predict, adapter = load_program_and_adapter(args.judge_program)

    start = time.perf_counter()
    labels = classify_batch(
        items,
        adapter=adapter,
        predict=predict,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )
    elapsed = time.perf_counter() - start

    if len(labels) != args.batch_size:
        raise SystemExit(f"expected {args.batch_size} labels, got {len(labels)}")
    if not all(isinstance(label, str) and label for label in labels):
        raise SystemExit("batch returned a non-string or empty label")

    print(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "elapsed_seconds": elapsed,
                "batch_generations_per_second": 1.0 / elapsed,
                "sentences_per_second": args.batch_size / elapsed,
                "labels": labels,
                "distinct_labels": sorted(set(labels)),
            },
            sort_keys=True,
        )
    )
    print(
        f"MTP_PROBE_OK batch_size={args.batch_size} labels={len(labels)} "
        f"elapsed_seconds={elapsed:.2f}"
    )


if __name__ == "__main__":
    main()

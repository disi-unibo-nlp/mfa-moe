"""Build a taxonomy summary from one or more already-labelled trace files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_exp.experiment1.taxonomy import build_row, build_summary
from moe_exp.schemas import TraceRecord


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for input_path in args.input:
        traces: list[TraceRecord] = []
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    traces.append(TraceRecord.model_validate_json(line))
        if not traces:
            continue
        model_ids = {trace.model_id for trace in traces}
        datasets = {trace.dataset for trace in traces}
        if len(model_ids) != 1 or len(datasets) != 1:
            raise ValueError(
                f"{input_path} must contain one model/dataset pair; "
                f"found models={sorted(model_ids)}, datasets={sorted(datasets)}"
            )
        rows.append(build_row(model_ids.pop(), datasets.pop(), traces))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(build_summary(rows), handle, indent=2)
    print(f"Wrote {len(rows)} taxonomy rows to {args.output}")


if __name__ == "__main__":
    main()

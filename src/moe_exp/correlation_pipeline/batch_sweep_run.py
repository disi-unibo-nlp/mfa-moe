"""Sweep runner: executes the cells reachable with the currently running server.

The sbatch wrapper starts one server per (mtp, max_num_seqs) group and invokes this
module with the matching filter, so server restarts happen only when a server-level
setting changes. Cell rows are appended to a shared scoreboard.jsonl.

The batch-1 / concurrency-1 / mtp-off cell is the control: its labels are stored and
every other cell's labels are compared against it. A cell whose labels differ is marked
label_mismatch and is disqualified by the selection rule via `failures`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_exp.correlation_pipeline.batch_benchmark import append_row, build_cells, run_cell
from moe_exp.correlation_pipeline.batch_predictor import load_program_and_adapter
from moe_exp.correlation_pipeline.batch_probe import build_probe_items

CONTROL_NAME = "control_labels.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--slice-size", type=int, default=256)
    parser.add_argument("--judge-program", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:41800/v1")
    parser.add_argument("--api-key", default="local-vllm-key")
    parser.add_argument("--scoreboard", type=Path, required=True)
    parser.add_argument("--mtp", action="store_true", help="Server currently has MTP enabled")
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="medium")
    args = parser.parse_args()

    per_file = max(1, args.slice_size // len(args.traces))
    items = []
    for path in args.traces:
        items.extend(build_probe_items(path, per_file))
    items = items[: args.slice_size]
    if len(items) < args.slice_size:
        raise SystemExit(f"slice has {len(items)} items, expected {args.slice_size}")
    print(f"slice_ready items={len(items)}")

    _, predict, adapter = load_program_and_adapter(args.judge_program)

    cells = [
        cell
        for cell in build_cells(
            batch_sizes=args.batch_sizes,
            concurrencies=args.concurrencies,
            mtp_modes=[args.mtp],
        )
        if cell["max_num_seqs"] == args.max_num_seqs
    ]
    if not cells:
        raise SystemExit("no cells match this server configuration")

    control_path = args.scoreboard.parent / CONTROL_NAME
    for cell in cells:
        result = run_cell(
            cell,
            items=items,
            adapter=adapter,
            predict=predict,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )
        row, labels = result["row"], result["labels"]

        is_control = (
            cell["batch_size"] == 1 and cell["concurrency"] == 1 and cell["mtp"] is False
        )
        if is_control and not control_path.is_file():
            control_path.parent.mkdir(parents=True, exist_ok=True)
            control_path.write_text(json.dumps({str(k): v for k, v in labels.items()}))
            row["label_mismatch"] = 0
            row["is_control"] = True
        elif control_path.is_file():
            control = {int(k): v for k, v in json.loads(control_path.read_text()).items()}
            shared = set(control) & set(labels)
            mismatch = sum(1 for i in shared if control[i] != labels[i])
            row["label_mismatch"] = mismatch
            row["label_compared"] = len(shared)
            row["is_control"] = False
            # Label drift is a correctness failure, not a performance note.
            row["failures"] = row["failures"] + mismatch
        else:
            row["label_mismatch"] = None
            row["is_control"] = False

        append_row(args.scoreboard, row)
        print(
            f"CELL_DONE batch={cell['batch_size']} conc={cell['concurrency']} "
            f"mtp={cell['mtp']} max_num_seqs={cell['max_num_seqs']} "
            f"tok_s={row['aggregate_tokens_per_second']:.1f} "
            f"p95_ms={row['latency_p95_ms']:.0f} failures={row['failures']} "
            f"mismatch={row.get('label_mismatch')}"
        )

    print(f"SWEEP_GROUP_OK mtp={args.mtp} max_num_seqs={args.max_num_seqs} cells={len(cells)}")


if __name__ == "__main__":
    main()

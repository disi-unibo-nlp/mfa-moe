"""D3c stage 1: labelled-token table of GPT cohort B (reserved; used only for frozen-score contrasts).

Reuses the D3b extraction (`d3b_tokens.extract_attempt` and driver) with the cohort-B task list; outcome columns are
never loaded.  python -m dynrt.d3c_tokens [--workers W]  -> results/B4/tokens_B/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from . import d3b_tokens as T
from .common import FAST, RESULTS, V3R2, sha256_file

OUT_B = RESULTS / "B4" / "tokens_B"


def gpt_b_tasks(limit: int | None = None) -> tuple[list[dict], dict]:
    """GPT cohort-B tasks (same fields as `d3b_tokens.gpt_tasks`; outcome columns are not loaded)."""
    from v3an.data import load_labels, read_location
    ready = json.loads((FAST / "manifests/gpt/B/ready.json").read_text())
    spec = ready["spec"]
    meta = dict(num_experts=int(spec["experts"]), num_layers=len(spec["layers"]), top_k=int(spec["top_k"]))
    att = pd.read_parquet(V3R2 / "gpt/B/attempts.parquet", columns=T.GPT_COLUMNS)
    att = att.sort_values("attempt_id").reset_index(drop=True)
    if limit:
        att = att.iloc[::max(1, len(att) // limit)].iloc[:limit].reset_index(drop=True)
    labels, label_prov = load_labels("gpt")
    tasks, missing = [], 0
    for r in att.itertuples(index=False):
        lab = labels.get((r.dataset, r.problem_id, int(r.sample_id), r.trace_sha256))
        if lab is None:
            missing += 1
            lab = []
        tasks.append(dict(attempt_id=r.attempt_id, question=r.question, dataset=r.dataset,
                          tensor_path=r.tensor_path, loader="gz",
                          trace=read_location(json.loads(r.source_location)), trace_sha256=r.trace_sha256,
                          labels=lab, completion_tokens=int(r.completion_tokens), split="", **meta))
    tasks.sort(key=lambda t: -Path(t["tensor_path"]).stat().st_size)
    prov = dict(dataset="gpt-B", spec=meta, n_tasks=len(tasks), attempts_without_labels=missing,
                label_files=label_prov, attempts_parquet_sha256=sha256_file(V3R2 / "gpt/B/attempts.parquet"),
                ready_sha256=sha256_file(FAST / "manifests/gpt/B/ready.json"))
    return tasks, prov


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT_B)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    a = p.parse_args(argv)
    T.gpt_tasks = gpt_b_tasks
    checks = T.run("gpt", a.out, workers=a.workers, limit=a.limit)
    return 1 if checks["failed_checks"] else 0


if __name__ == "__main__":
    sys.exit(main())

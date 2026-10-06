"""D3 feature freeze: cohort-B features L1 (bJSD) and L2 (margin) for the replication models.

Profiles and cells are fitted on each model's cohort-A attempts whose questions are absent from its cohort B
(`freeze_d2.b_dyn`, frozen D1/D2 definitions). No outcome is read. qwen35 is added for the descriptive report;
qwen36 dev+tune B features are those already frozen in D2.
CLI: python -m dynrt.freeze_d3  -> results/FEATURES_FROZEN_D3.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import transitions as T
from .c2 import verify_frozen_d2
from .common import RESULTS, digest, sha256_file
from .data import load_cohort
from .freeze_d2 import b_dyn, describe_pairs

REPLICATION = ("gemma", "glm", "nemotron", "qwen330b")
DESCRIPTIVE = ("qwen35",)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    d12 = verify_frozen_d2()
    log(f"D1/D2 freezes verified: {d12}")
    manifest: dict = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), verified=d12, replication_models=list(REPLICATION),
                          descriptive_models=list(DESCRIPTIVE) + ["qwen36 (dev+tune, D2 features)"], models={})
    for model in REPLICATION + DESCRIPTIVE:
        t0 = time.time()
        coh_a, coh_b = load_cohort(model, "A"), load_cohort(model, "B")
        pt_a, pt_b = T.build_pairs(coh_a), T.build_pairs(coh_b)
        df, binfo = b_dyn(coh_a, coh_b, pt_a, pt_b)
        df.to_parquet(RESULTS / model / "B" / "b_dyn.parquet")
        att_b = coh_b.attempts
        prov = {c: json.loads((RESULTS / model / c / "provenance.json").read_text()) for c in "AB"}
        info = dict(
            shape=dict(layers=coh_a.table.layers, experts=coh_a.table.experts),
            A=dict(attempts=int(len(coh_a.attempts)), sentences=int(len(coh_a.sent)),
                   failed_checks=prov["A"]["checks"]["failed_checks"]),
            B=dict(attempts=int(len(att_b)), questions=int(att_b["question"].nunique()),
                   capped=int(att_b["capped"].sum()), sentences=int(len(coh_b.sent)),
                   failed_checks=prov["B"]["checks"]["failed_checks"]),
            fit=dict(train_attempts=binfo["train_attempts"],
                     a_questions_in_b=int(coh_a.attempts["question"].isin(set(att_b["question"])).sum())),
            pairs_A=describe_pairs(coh_a, pt_a), pairs_B=describe_pairs(coh_b, pt_b),
            L1_bJSD_available=int(np.isfinite(df["bJSD"]).sum()), L2_margin_available=int(np.isfinite(df["margin"]).sum()),
            file_sha256=sha256_file(RESULTS / model / "B" / "b_dyn.parquet"))
        manifest["models"][model] = info
        log(f"{model}: B {info['B']['attempts']} attempts / {info['B']['questions']} questions, "
            f"{info['pairs_B']['pairs']} pairs, L1 available {info['L1_bJSD_available']}, {time.time() - t0:.0f}s")
    manifest["models"]["qwen36"] = dict(
        file_sha256=sha256_file(RESULTS / "qwen36" / "B" / "b_dyn.parquet"),
        note="frozen in D2 (qwen36 dev+tune cohort B, 16 attempts)")
    code = Path(__file__).resolve().parent
    manifest["code_sha256"] = {p.name: sha256_file(p) for p in sorted(code.glob("*.py"))
                               if p.name not in ("c2.py", "c6.py", "summary.py", "d3.py")}
    manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "time"})
    (RESULTS / "FEATURES_FROZEN_D3.json").write_text(json.dumps(manifest, indent=1, default=float))
    log("wrote FEATURES_FROZEN_D3.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
